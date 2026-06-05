"""Provenance, atomic IO, and metadata sidecars.

Every artifact this pipeline writes must be traceable back to a dataset, model,
configuration, and exact run. This module is the single place that:

  * mints a ``run_id`` that threads through a whole pipeline run,
  * captures the git SHA and environment,
  * writes files atomically (temp file + ``os.replace``) so an interrupted run never
    leaves a half-written CSV/checkpoint behind,
  * computes input checksums,
  * writes the ``<artifact>.meta.json`` sidecar required for every artifact.

See CLAUDE.md ("Provenance and reproducibility").
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v1"


def new_run_id() -> str:
    """Mint a run id: UTC date + short uuid, sortable and human-skimmable.

    Example: ``20260605T142233Z-a1b2c3d4``. Time is injected via :func:`utc_now_iso`'s
    clock; callers that need determinism in tests can pass a fixed id downstream.
    """
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    """Current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def git_sha(repo: Path | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable.

    Never raises: provenance capture must not break a run on a machine without git or
    outside a repository (e.g. a fresh pod checkout of a tarball).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_dirty(repo: Path | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes; ``None`` if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (1 MiB chunks; safe for large CSVs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes atomically: temp file in the same dir, fsync, then ``os.replace``.

    Same-directory temp guarantees ``os.replace`` is atomic (no cross-filesystem move).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically (UTF-8)."""
    atomic_write_bytes(path, text.encode("utf-8"))


def env_fingerprint() -> dict[str, Any]:
    """Capture the runtime environment relevant to numerical reproducibility."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:  # torch optional for non-model steps / light tests
        pass
    return info


@dataclass
class InputArtifact:
    """A consumed input, recorded with its checksum for downstream auditing."""

    path: str
    sha256: str

    @classmethod
    def of(cls, path: str | Path) -> "InputArtifact":
        return cls(path=str(path), sha256=sha256_file(path))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass
class ArtifactMetadata:
    """The ``<artifact>.meta.json`` sidecar required for every written artifact.

    Mirrors the minimum schema in CLAUDE.md and adds an environment fingerprint.
    """

    artifact: str
    produced_by: str
    run_id: str
    dataset: str
    model: str
    produced_at: str = field(default_factory=utc_now_iso)
    git_sha: str | None = field(default_factory=git_sha)
    git_dirty: bool | None = field(default_factory=git_dirty)
    dataset_manifest: str | None = None
    input_artifacts: list[InputArtifact] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    environment: dict[str, Any] = field(default_factory=env_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "dataset": self.dataset,
            "dataset_manifest": self.dataset_manifest,
            "model": self.model,
            "input_artifacts": [a.to_dict() for a in self.input_artifacts],
            "config": self.config,
            "schema_version": self.schema_version,
            "environment": self.environment,
            **({"extra": self.extra} if self.extra else {}),
        }

    def write(self, artifact_path: str | Path) -> Path:
        """Write the sidecar next to ``artifact_path`` as ``<artifact>.meta.json``."""
        meta_path = Path(str(artifact_path) + ".meta.json")
        atomic_write_text(meta_path, json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return meta_path
