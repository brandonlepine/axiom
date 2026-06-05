"""ResidualPatcher: per-token, per-layer residual-stream activation patching.

Question: does injecting the **stereotype** variant's residual state into the
**reference** run raise the probability the model assigns the (shared) stereotype
continuation -- and where (which layer / token) does that effect live?

    source = sent_x = the stereotype-associated identity (the activations we inject)
    target = sent_y = the reference identity (the run we patch into)

The two prompts differ only in the identity span, so they share a token prefix and
suffix; :func:`axiom.alignment.align_minimal_pair` maps each target position to its
aligned source position. For each (layer, target position) we copy the source
``resid_pre`` activation into the target run and re-score the continuation:

    bias_effect = patched_cont_avg_logp - reference_cont_avg_logp
    normalized_restoration = bias_effect / (stereotype_cont_avg_logp - reference_cont_avg_logp)

``bias_effect > 0`` => injecting the stereotype state raised the continuation's
probability. We also patch the whole identity span at once per layer
(``token_position = -1``, span ``identity_all``).

Direction is fixed and explicit: ``stereotype_into_reference`` (sent_x state into the
sent_y run), recorded in provenance.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from axiom.alignment import align_minimal_pair
from axiom.interventions.base import InterventionRunner, StepResult
from axiom.interventions.readout import continuation_logp
from axiom.metrics.bias import bias_effect as _bias_effect
from axiom.metrics.bias import normalized_restoration as _normalized_restoration
from axiom.paths import slugify
from axiom.provenance import ArtifactMetadata, InputArtifact

DIRECTION = "stereotype_into_reference"

RAW_COLUMNS = [
    "cohort_pair_id", "row_id", "identity_x", "identity_y", "predicate_label_provisional", "axis",
    "layer", "token_position", "span", "source_position", "is_identity_token",
    "token_text_target", "token_text_source",
    "reference_cont_avg_logp", "stereotype_cont_avg_logp", "patched_cont_avg_logp",
    "bias_effect", "normalized_restoration",
]


class ResidualPatcher(InterventionRunner):
    """Residual-stream activation patcher (see module docstring)."""

    step_name = "residual_patching"
    produced_by = "ResidualPatcher"

    @property
    def required_inputs(self) -> list[Path]:
        return [self.adapter.cohort_path()]

    # -- cohort / resume ----------------------------------------------------------
    def _load_pairs(self) -> pd.DataFrame:
        df = self.adapter.load_cohort(validate=True).reset_index(drop=True)
        pc = self.config.patching
        if pc is not None and pc.max_pairs is not None:
            df = df.head(pc.max_pairs).copy()
        return df

    def _resume_done_ids(self, raw_path: Path) -> set[int]:
        """Return cohort_pair_ids already complete; rewrite the raw to drop the last
        (possibly partial) pair so it is recomputed cleanly."""
        try:
            existing = pd.read_csv(raw_path)
        except Exception:
            return set()
        present = sorted(int(p) for p in existing.get("cohort_pair_id", pd.Series([], dtype=int)).dropna().unique())
        if not present:
            return set()
        done = set(present[:-1])  # redo the last pair (it may have been interrupted mid-write)
        existing[existing["cohort_pair_id"].isin(done)].reindex(columns=RAW_COLUMNS).to_csv(raw_path, index=False)
        return done

    # -- run ----------------------------------------------------------------------
    def run(self) -> StepResult:
        import torch
        from tqdm.auto import tqdm
        from transformer_lens import utils as tl_utils

        self.check_inputs()
        if self.loaded is None:
            raise ValueError("ResidualPatcher requires a loaded model.")
        pc = self.config.patching
        patch_bs = pc.patch_batch_size if pc else 32
        resume = pc.resume if pc else True

        model = self.loaded.model
        tokenizer = self.loaded.tokenizer
        device = self.loaded.device
        n_layers = self.loaded.n_layers
        layers = list(pc.layers) if (pc and pc.layers) else list(range(n_layers))
        id_x_col, id_y_col = self.adapter.identity_columns

        pairs = self._load_pairs()
        out_dir = self.layout.ensure()
        raw_name = f"{self.config.dataset}_{self._model_slug()}_residual_patching_raw.csv"
        raw_path = out_dir / raw_name

        done_ids: set[int] = self._resume_done_ids(raw_path) if (resume and raw_path.exists()) else set()
        mode = "a" if done_ids else "w"
        fh = raw_path.open(mode, newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        if mode == "w":
            writer.writeheader()

        skipped: list[tuple[int, str]] = []
        progress = tqdm(total=len(pairs) - len(done_ids), desc=f"patch {self.config.dataset}", unit="pair")
        try:
            for _, row in pairs.iterrows():
                pair_id = int(row["cohort_pair_id"])
                if pair_id in done_ids:
                    continue
                rows = self._patch_one_pair(
                    model, tokenizer, device, layers, tl_utils, torch, row, id_x_col, id_y_col
                )
                progress.update(1)
                if isinstance(rows, str):
                    skipped.append((pair_id, rows))
                    progress.set_postfix(skipped=len(skipped))
                    continue
                writer.writerows(rows)
                fh.flush()
                if device == "cuda":
                    torch.cuda.empty_cache()
        finally:
            progress.close()
            fh.close()

        return self._finalize(raw_path, raw_name, len(pairs), skipped)

    def _patch_one_pair(
        self, model, tokenizer, device, layers, tl_utils, torch, row, id_x_col, id_y_col
    ) -> list[dict[str, Any]] | str:
        """Patch every (layer, position) for one pair; return raw rows or a skip-reason string."""
        s_ids = tokenizer(str(row["sent_x"]), add_special_tokens=True)["input_ids"]
        t_ids = tokenizer(str(row["sent_y"]), add_special_tokens=True)["input_ids"]
        pref_ids = tokenizer(str(row["prefix_y"]), add_special_tokens=True)["input_ids"]
        aln, reason = align_minimal_pair(s_ids, t_ids, pref_ids)
        if aln is None:
            return reason

        s_t = torch.tensor([aln.source_ids], device=device)
        t_t = torch.tensor([aln.target_ids], device=device)
        t_flat = t_t[0]
        len_t = len(aln.target_ids)
        len_s = len(aln.source_ids)
        cont_start, cont_count = aln.cont_start, aln.cont_count

        # baselines + cached source resid_pre
        _, src_cache = model.run_with_cache(s_t, names_filter=lambda n: n.endswith("hook_resid_pre"))
        reference_avg = float(continuation_logp(model(t_t), t_flat, cont_start)[0].item()) / cont_count
        stereotype_avg = float(
            continuation_logp(model(s_t), s_t[0], len_s - cont_count)[0].item()
        ) / cont_count

        tok_text_target = [tokenizer.decode([tid]).replace("\n", "\\n") for tid in aln.target_ids]
        tok_text_source = [
            tokenizer.decode([aln.source_ids[aln.source_pos[c]]]).replace("\n", "\\n") for c in range(len_t)
        ]

        out_rows: list[dict[str, Any]] = []

        def record(layer: int, pos: int, span: str, src: int, patched_avg: float) -> None:
            be = _bias_effect(patched_avg, reference_avg)
            out_rows.append({
                "cohort_pair_id": int(row["cohort_pair_id"]), "row_id": row.get("row_id"),
                "identity_x": row.get(id_x_col), "identity_y": row.get(id_y_col),
                "predicate_label_provisional": row.get("predicate_label_provisional"),
                "axis": row.get("axis"),
                "layer": layer, "token_position": pos, "span": span, "source_position": src,
                "is_identity_token": int(span == "identity"),
                "token_text_target": tok_text_target[pos] if pos >= 0 else "",
                "token_text_source": tok_text_source[pos] if pos >= 0 else "",
                "reference_cont_avg_logp": reference_avg, "stereotype_cont_avg_logp": stereotype_avg,
                "patched_cont_avg_logp": patched_avg, "bias_effect": be,
                "normalized_restoration": _normalized_restoration(patched_avg, reference_avg, stereotype_avg),
            })

        pbs = max(1, int(self.config.patching.patch_batch_size)) if self.config.patching else 32
        for layer in layers:
            act_name = tl_utils.get_act_name("resid_pre", layer)
            qact = src_cache[act_name]  # [1, len_s, d]

            for chunk_start in range(0, len_t, pbs):
                chunk = list(range(chunk_start, min(chunk_start + pbs, len_t)))
                ctrl_pos = torch.tensor(chunk, device=device)
                src_pos = torch.tensor([aln.source_pos[c] for c in chunk], device=device)
                batched = t_t.repeat(len(chunk), 1)

                def hook(act, hook, ctrl_pos=ctrl_pos, src_pos=src_pos, qact=qact):
                    act = act.clone()
                    batch_rows = torch.arange(act.shape[0], device=act.device)
                    act[batch_rows, ctrl_pos, :] = qact[0, src_pos, :].to(act.dtype)
                    return act

                patched = model.run_with_hooks(batched, fwd_hooks=[(act_name, hook)])
                sums = continuation_logp(patched, t_flat, cont_start)
                for i, c in enumerate(chunk):
                    record(layer, c, aln.spans[c], aln.source_pos[c], float(sums[i].item()) / cont_count)

            if aln.identity_target_positions:
                ctrl_pos = torch.tensor(aln.identity_target_positions, device=device)
                src_pos = torch.tensor(aln.identity_source_positions, device=device)

                def hook_all(act, hook, ctrl_pos=ctrl_pos, src_pos=src_pos, qact=qact):
                    act = act.clone()
                    act[0, ctrl_pos, :] = qact[0, src_pos, :].to(act.dtype)
                    return act

                patched = model.run_with_hooks(t_t, fwd_hooks=[(act_name, hook_all)])
                avg = float(continuation_logp(patched, t_flat, cont_start)[0].item()) / cont_count
                record(layer, -1, "identity_all", -1, avg)

        del src_cache
        return out_rows

    # -- aggregation + provenance -------------------------------------------------
    def _finalize(self, raw_path: Path, raw_name: str, n_pairs: int, skipped: list[tuple[int, str]]) -> StepResult:
        from axiom.alignment import PATCH_SPANS
        from axiom.figures.patching import residual_span_heatmap

        raw_df = pd.read_csv(raw_path)
        self._write_sidecar(raw_path, raw_name, {"n_pairs": n_pairs, "n_skipped": len(skipped),
                                                 "direction": DIRECTION, "skipped": skipped[:50]})

        # span x layer mean bias_effect (the headline localization result)
        span_df = raw_df[raw_df["span"].isin(PATCH_SPANS)].copy()
        span_df["span"] = pd.Categorical(span_df["span"], categories=list(PATCH_SPANS), ordered=True)
        span_summary = (
            span_df.groupby(["layer", "span"], observed=True)["bias_effect"].mean()
            .reset_index().rename(columns={"bias_effect": "mean_bias_effect"})
        )
        span_name = f"{self.config.dataset}_{self._model_slug()}_residual_patching_span_summary.csv"
        self.write_dataframe(span_summary, span_name, self.produced_by, extra={"direction": DIRECTION})

        # whole-identity-span effect by layer
        id_all = (
            raw_df[raw_df["span"] == "identity_all"].groupby("layer")["bias_effect"].mean()
            .reset_index().rename(columns={"bias_effect": "mean_bias_effect"})
        )
        id_name = f"{self.config.dataset}_{self._model_slug()}_residual_patching_identity_by_layer.csv"
        self.write_dataframe(id_all, id_name, self.produced_by, extra={"direction": DIRECTION})

        fig_path = self.layout.artifact(
            f"{self.config.dataset}_{self._model_slug()}_residual_patching_span_heatmap"
        )
        figs = residual_span_heatmap(
            span_summary, fig_path,
            title=f"{self.config.dataset}: resid_pre patching ({DIRECTION}) by span",
        )

        artifacts = [raw_path, self.layout.artifact(span_name), self.layout.artifact(id_name), *figs]
        return StepResult(
            artifacts=artifacts,
            summary={
                "n_pairs": n_pairs, "n_skipped": len(skipped),
                "mean_identity_all_bias_effect": float(id_all["mean_bias_effect"].mean()) if len(id_all) else float("nan"),
                "peak_identity_layer": int(id_all.loc[id_all["mean_bias_effect"].idxmax(), "layer"]) if len(id_all) else -1,
            },
        )

    def _model_slug(self) -> str:
        return slugify(self.config.model.name)

    def _write_sidecar(self, path: Path, name: str, extra: dict[str, Any]) -> None:
        cfg = self.config.model_dump(mode="json")
        if self.loaded is not None:
            cfg["model_runtime"] = self.loaded.provenance()
        ArtifactMetadata(
            artifact=name, produced_by=self.produced_by, run_id=self.layout.run_id,
            dataset=self.config.dataset, model=self.config.model.name,
            input_artifacts=[InputArtifact.of(p) for p in self.required_inputs if Path(p).exists()],
            config=cfg, extra=extra,
        ).write(path)
