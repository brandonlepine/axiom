"""Config validation invariants (CLAUDE.md: config is data, validated before use)."""
import pytest
from pydantic import ValidationError

from axiom.config import ModelConfig, RunConfig, load_model_registry, load_run_config


def test_model_config_minimal():
    m = ModelConfig(name="gpt2", hf_path="gpt2", tl_name="gpt2", family="gpt2")
    assert m.device == "auto" and m.dtype == "float16" and m.prepend_bos is True


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(name="gpt2", hf_path="gpt2", tl_name="gpt2", typo=True)


def test_run_config_requires_nonempty_dataset():
    with pytest.raises(ValidationError):
        RunConfig(step="scoring", dataset="  ", model=ModelConfig(name="g", hf_path="g", tl_name="g"))


def test_load_registry_and_run_config_from_repo(tmp_path):
    reg_yaml = tmp_path / "models.yaml"
    reg_yaml.write_text(
        "gpt2:\n  name: gpt2\n  hf_path: gpt2\n  tl_name: gpt2\n  family: gpt2\n  dtype: float32\n"
    )
    reg = load_model_registry(reg_yaml)
    assert reg["gpt2"].dtype == "float32"

    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(
        "step: scoring\ndataset: winoqueer\nseed: 1\n"
        "model:\n  name: gpt2\n  hf_path: gpt2\n  tl_name: gpt2\n  family: gpt2\n"
        "scoring:\n  batch_size: 4\n  max_pairs: 5\n"
    )
    cfg = load_run_config(run_yaml)
    assert cfg.dataset == "winoqueer" and cfg.scoring.max_pairs == 5 and cfg.seed == 1
