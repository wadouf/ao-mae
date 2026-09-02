from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CHECKPOINT_PATTERN = "{method}_loco_{city}_K{k}_seed{seed:02d}.npz"


class AdapterResolutionError(RuntimeError):
    pass


@dataclass
class ResolvedMethod:
    name: str
    kind: str
    module: str | None = None
    callable_name: str | None = None
    repository: Path | None = None
    checkpoint_root: Path | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def callable_path(self) -> str:
        return f"{self.module}:{self.callable_name}"

    @property
    def available(self) -> bool:
        return not self.missing


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    methods = cfg.get("methods") if isinstance(cfg, dict) else None
    if not isinstance(methods, dict):
        raise ValueError("Adapter configuration must contain a methods mapping")
    return methods


def resolve_method(name: str, spec: dict[str, Any]) -> ResolvedMethod:
    kind = spec.get("adapter", "internal_reference")
    if kind == "internal_reference":
        return ResolvedMethod(name=name, kind=kind)
    if kind != "python_module":
        return ResolvedMethod(name=name, kind=kind, missing=[f"unsupported adapter type {kind}"])

    missing: list[str] = []
    repository_env = spec.get("repository_env")
    checkpoint_env = spec.get("checkpoint_root_env")

    repository = None
    if repository_env:
        value = os.environ.get(repository_env, "").strip()
        if not value:
            missing.append(f"environment variable {repository_env} is not set")
        else:
            repository = Path(value).expanduser().resolve()
            if not repository.exists():
                missing.append(f"{repository_env} points at a missing path: {repository}")

    checkpoint_root = None
    if checkpoint_env:
        value = os.environ.get(checkpoint_env, "").strip()
        if not value:
            missing.append(f"environment variable {checkpoint_env} is not set")
        else:
            checkpoint_root = Path(value).expanduser().resolve()
            if not checkpoint_root.exists():
                missing.append(f"{checkpoint_env} points at a missing path: {checkpoint_root}")

    return ResolvedMethod(
        name=name,
        kind=kind,
        module=spec.get("module"),
        callable_name=spec.get("callable", "predict_batch"),
        repository=repository,
        checkpoint_root=checkpoint_root,
        missing=missing,
    )


def resolve_all(adapter_config_path: str | Path) -> dict[str, ResolvedMethod]:
    methods = load_adapter_config(adapter_config_path)
    return {name: resolve_method(name, spec or {}) for name, spec in methods.items()}


def register_repository(method: ResolvedMethod) -> None:
    if method.repository is None:
        return
    entry = str(method.repository)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def checkpoint_path(method: ResolvedMethod, city: str, k: int, seed: int) -> Path:
    if method.checkpoint_root is None:
        raise AdapterResolutionError(f"{method.name} has no checkpoint root")
    name = CHECKPOINT_PATTERN.format(method=method.name.lower(), city=city, k=k, seed=seed)
    return method.checkpoint_root / name
