"""Provenance/atomicity and output-layout invariants (CLAUDE.md: metadata + naming)."""
import json

from axiom.paths import OutputLayout, slugify
from axiom.provenance import (
    ArtifactMetadata,
    atomic_write_text,
    new_run_id,
    sha256_file,
)


def test_slugify_model_names():
    assert slugify("meta-llama/Llama-3.1-8B") == "llama-3.1-8b"
    assert slugify("gpt2") == "gpt2"
    assert slugify("EleutherAI/pythia_70m") == "pythia-70m"


def test_run_id_format():
    rid = new_run_id()
    assert rid.endswith
    date, _, suffix = rid.partition("-")
    assert date.endswith("Z") and len(suffix) == 8


def test_output_layout_path(tmp_path):
    layout = OutputLayout(
        model="meta-llama/Llama-3.1-8B",
        dataset="winoqueer",
        step="residual_patching",
        run_id="20260605T000000Z-deadbeef",
        root=tmp_path,
    )
    expected = tmp_path / "outputs" / "llama-3.1-8b" / "winoqueer" / "residual_patching" / "20260605T000000Z-deadbeef"
    assert layout.dir == expected
    layout.ensure()
    assert expected.is_dir()
    assert layout.artifact("x.csv") == expected / "x.csv"


def test_atomic_write_and_sha(tmp_path):
    p = tmp_path / "sub" / "a.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    # sha256 is stable + matches a re-hash
    assert sha256_file(p) == sha256_file(p)
    # no leftover temp files in the directory
    assert [q.name for q in p.parent.iterdir()] == ["a.txt"]


def test_metadata_sidecar_written(tmp_path):
    art = tmp_path / "winoqueer_gpt2_scoring_raw.csv"
    atomic_write_text(art, "row_id,bias_score\n1,0.5\n")
    meta = ArtifactMetadata(
        artifact=art.name,
        produced_by="BiasScorer",
        run_id="20260605T000000Z-deadbeef",
        dataset="winoqueer",
        model="gpt2",
        input_artifacts=[],
        config={"seed": 0},
    )
    meta_path = meta.write(art)
    assert meta_path.name == art.name + ".meta.json"
    loaded = json.loads(meta_path.read_text())
    assert loaded["produced_by"] == "BiasScorer"
    assert loaded["run_id"] == "20260605T000000Z-deadbeef"
    assert loaded["schema_version"] == "v1"
    assert "environment" in loaded
