"""Versioned adapter-artifact store with atomic WRITING -> READY manifests.

A public adapter version is only exposed once its manifest is READY. ``create_v0``
publishes a concrete initial artifact at job creation so disaggregated inference
has something to load. Torch-free (json/filesystem); object-store backends slot in
later behind the same interface.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .schemas import TrainingJobSpec, lora_config_hash


@dataclass
class AdapterArtifactManifest:
    job_id: str
    version: int
    status: Literal["WRITING", "READY", "FAILED"]
    base_model: str
    base_model_revision: str | None
    lora_config_hash: str | None
    rank: int
    alpha: int
    target_modules: list[str]
    files: list[str] = field(default_factory=list)
    is_base: bool = False
    trained_steps: int = 0
    trained_tokens: int = 0
    created_at: float = field(default_factory=time.time)


class ArtifactStore:
    """Local-filesystem implementation. URIs are directories in the MVP."""

    def _version_dir(self, output_uri: str, version: int) -> Path:
        return Path(output_uri) / "versions" / f"v{version:06d}"

    def _write_manifest(self, version_dir: Path, manifest: AdapterArtifactManifest) -> str:
        version_dir.mkdir(parents=True, exist_ok=True)
        path = version_dir / "manifest.json"
        path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def create_v0(self, spec: TrainingJobSpec) -> str:
        """Publish the initial (base / no-adapter) artifact, READY immediately."""
        version_dir = self._version_dir(spec.output_uri, 0)
        manifest = AdapterArtifactManifest(
            job_id=spec.job_id,
            version=0,
            status="READY",
            base_model=spec.base_model,
            base_model_revision=spec.base_model_revision,
            lora_config_hash=lora_config_hash(spec.adapter),
            rank=spec.adapter.rank,
            alpha=spec.adapter.alpha,
            target_modules=list(spec.adapter.target_modules),
            files=[],
            is_base=True,
        )
        return self._write_manifest(version_dir, manifest)

    def finalize_manifest(
        self,
        spec: TrainingJobSpec,
        version: int,
        files: list[str],
        *,
        trained_steps: int = 0,
        trained_tokens: int = 0,
    ) -> str:
        """Write a READY manifest for a trained version after files exist on disk."""
        version_dir = self._version_dir(spec.output_uri, version)
        missing = [f for f in files if not os.path.exists(f)]
        status: Literal["READY", "FAILED"] = "FAILED" if missing else "READY"
        manifest = AdapterArtifactManifest(
            job_id=spec.job_id,
            version=version,
            status=status,
            base_model=spec.base_model,
            base_model_revision=spec.base_model_revision,
            lora_config_hash=lora_config_hash(spec.adapter),
            rank=spec.adapter.rank,
            alpha=spec.adapter.alpha,
            target_modules=list(spec.adapter.target_modules),
            files=list(files),
            trained_steps=trained_steps,
            trained_tokens=trained_tokens,
        )
        uri = self._write_manifest(version_dir, manifest)
        if status == "FAILED":
            raise RuntimeError(f"artifact files missing for {spec.job_id} v{version}: {missing}")
        return uri
