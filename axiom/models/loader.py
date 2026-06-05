"""Single shared interface for loading a model + tokenizer as a TransformerLens model.

Every model-bearing step constructs one :class:`ModelLoader` from a :class:`ModelConfig`
and calls :meth:`load`. This is the only place HuggingFace and TransformerLens loading
logic lives, so device/dtype/BOS/padding quirks are handled identically for the local
GPT-2 smoke path and the pod Llama path (CLAUDE.md, "Model loading is centralized").

The wrap policy matches the upstream analysis scripts: ``fold_ln=False`` and no
weight centering, so hooks see the model's true residual stream and head outputs
(centering would change what a patch/ablation actually does).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from axiom.config import ModelConfig
from axiom.device import effective_dtype_name, resolve_device, resolve_dtype

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


@dataclass
class LoadedModel:
    """A loaded model bundle plus the resolved runtime facts to record in provenance."""

    model: "HookedTransformer"
    tokenizer: Any
    device: str
    dtype_name: str
    n_layers: int
    n_heads: int
    d_model: int

    def provenance(self) -> dict[str, Any]:
        """Runtime facts about this load, for the artifact sidecar."""
        return {
            "device": self.device,
            "dtype": self.dtype_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
        }


class ModelLoader:
    """Loads a HuggingFace causal LM and wraps it with TransformerLens.

    Args:
        config: validated :class:`ModelConfig` (model id, hf_path, tl_name, device, dtype...).

    Example:
        >>> loader = ModelLoader(ModelConfig(name="gpt2", hf_path="gpt2",
        ...                                  tl_name="gpt2", family="gpt2"))
        >>> bundle = loader.load()  # doctest: +SKIP
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def load(self, verbose: bool = True) -> LoadedModel:
        import torch  # noqa: F401  (local import keeps non-model code / light tests torch-free)
        from pathlib import Path

        from transformer_lens import HookedTransformer

        cfg = self.config
        device = resolve_device(cfg.device)
        dtype = resolve_dtype(cfg.dtype, device)
        dtype_name = effective_dtype_name(cfg.dtype, device)

        # No weight processing: the hooks must see the model's true residual stream and
        # head outputs (folding LN / centering would change what a patch or ablation does).
        tl_kwargs = dict(
            device=device, dtype=dtype, fold_ln=False, center_writing_weights=False,
            center_unembed=False, default_prepend_bos=cfg.prepend_bos,
        )

        from_hub = not Path(cfg.hf_path).expanduser().is_dir()
        if verbose:
            src = "HF hub" if from_hub else f"local checkpoint {cfg.hf_path}"
            print(f"[ModelLoader] {cfg.name} | device={device} | dtype={dtype_name} | source={src}")
            print(f"[ModelLoader] tl_name={cfg.tl_name}")

        if from_hub:
            # Let TransformerLens download + convert the weights from HuggingFace directly.
            model = HookedTransformer.from_pretrained(cfg.tl_name, **tl_kwargs)
            tokenizer = model.tokenizer
        else:
            # Local HF checkpoint: load it explicitly and hand it to TransformerLens.
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.hf_path, use_fast=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.hf_path, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            model = HookedTransformer.from_pretrained(
                cfg.tl_name, hf_model=hf_model, tokenizer=tokenizer, **tl_kwargs
            )

        tokenizer.padding_side = cfg.padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype_name=dtype_name,
            n_layers=int(model.cfg.n_layers),
            n_heads=int(model.cfg.n_heads),
            d_model=int(model.cfg.d_model),
        )
