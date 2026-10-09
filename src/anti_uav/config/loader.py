"""Configuration loading.

All YAML under ``configs/`` is funnelled through here so that:

* validation happens exactly once, at load time, with a message naming the file
  and the offending path;
* resolution is memoised, so a FastAPI request touching eight configs does not
  re-read eight files;
* environment variables can override a field without editing YAML - necessary
  because the same repo runs on a Pascal box and a Blackwell box with the same
  checked-in config.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..utils.io import read_json
from ..utils.logging import get_logger
from ..utils.paths import project_root, subdir
from .schema import (
    AppSettings,
    CoverageMap,
    DatasetRegistry,
    DatasetSpec,
    DroneRules,
    ExperimentMatrix,
    GpuProfile,
    ProfileOverride,
    TrainRecipe,
)

log = get_logger(__name__)

M = TypeVar("M", bound=BaseModel)

_ENV_PREFIX = "ANTI_UAV_"


class ConfigError(RuntimeError):
    """Raised when a config file is missing or fails validation."""


def config_dir() -> Path:
    return subdir("configs")


def config_path(*parts: str) -> Path:
    return config_dir().joinpath(*parts)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(
            f"config file not found: {path}\n"
            f"expected it under {project_root() / 'configs'}"
        )
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Raw YAML read, no validation. For list-shaped files."""
    return _read_yaml(Path(path))


@functools.lru_cache(maxsize=128)
def _load_cached(path_str: str, model_name: str) -> BaseModel:
    models: dict[str, type[BaseModel]] = {
        "AppSettings": AppSettings,
        "CoverageMap": CoverageMap,
        "DatasetRegistry": DatasetRegistry,
        "DroneRules": DroneRules,
        "ExperimentMatrix": ExperimentMatrix,
        "ProfileOverride": ProfileOverride,
        "TrainRecipe": TrainRecipe,
    }
    model = models[model_name]
    data = _read_yaml(Path(path_str))
    data = _apply_env_overrides(data, path=Path(path_str))
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, Path(path_str))) from exc


def load(path: str | Path, model: type[M]) -> M:
    """Load and validate a config file into ``model``.

    Results are memoised by ``(path, model name)``. Call :func:`invalidate` after
    editing a config while the API is running.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        candidate = config_path(*Path(path).parts)
        resolved = candidate if candidate.is_file() else resolved
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")

    loaded = _load_cached(str(resolved), model.__name__)
    if not isinstance(loaded, model):  # pragma: no cover - cache key mismatch guard
        raise ConfigError(f"cached {model.__name__} but requested {model.__name__}")
    return loaded  # type: ignore[return-value]


def invalidate() -> None:
    """Drop the memo cache. Used by the API's config-reload endpoint."""
    _load_cached.cache_clear()
    log.info("config cache cleared")


def _format_validation_error(exc: ValidationError, path: Path) -> str:
    lines = [f"invalid config: {path}"]
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {location}: {err['msg']}")
        if err.get("input") not in (None, {}, []):
            lines.append(f"    got: {err['input']!r}")
    return "\n".join(lines)


def _apply_env_overrides(data: dict[str, Any], *, path: Path) -> dict[str, Any]:
    """Apply ``ANTI_UAV_<SECTION>__<FIELD>`` overrides.

    Example::

        ANTI_UAV_APP__DEVICE=cuda:0
        ANTI_UAV_APP__IMGSZ=1024

    Only sections that already exist in the file are considered, so a stray
    variable cannot invent a key. The section name comes from the file's parent
    directory (``configs/app.yaml`` -> ``APP``, ``configs/rules/*`` -> ``RULES``).
    """
    section = path.parent.name.upper()
    if not section:
        return data

    prefix = f"{_ENV_PREFIX}{section}__"
    overrides = {
        key: value for key, value in os.environ.items() if key.startswith(prefix)
    }
    if not overrides:
        return data

    for key, raw in overrides.items():
        parts = [p.lower() for p in key[len(prefix) :].split("__") if p]
        if not parts:
            continue
        cursor: dict[str, Any] = data
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                break
            cursor = nxt
        else:
            cursor[parts[-1]] = _coerce_env(raw)

    log.debug("applied env overrides", extra={"config": str(path), "count": len(overrides)})
    return data


def _coerce_env(raw: str) -> Any:
    """Best-effort type coercion so ``IMGSZ=1024`` is an int, not a string."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# --------------------------------------------------------------------------- #
# named accessors
# --------------------------------------------------------------------------- #


def load_registry() -> DatasetRegistry:
    return load(config_path("datasets", "registry.yaml"), DatasetRegistry)


def load_dataset(alias: str) -> DatasetSpec:
    """Load one dataset spec, preferring a per-dataset file when present.

    ``configs/datasets/<alias>.yaml`` may hold a single ``DatasetSpec``; the
    registry is the fallback and the source of truth for shared fields.
    """
    per_dataset = config_path("datasets", f"{alias}.yaml")
    if per_dataset.is_file():
        return load(per_dataset, DatasetSpec)

    registry = load_registry()
    if alias not in registry.datasets:
        known = ", ".join(registry.aliases)
        raise ConfigError(f"unknown dataset {alias!r}; known datasets: {known}")
    return registry.get(alias)


def load_recipe(family: str) -> TrainRecipe:
    """``configs/train/<family>.yaml``."""
    safe = family.lower().replace("-", "_")
    path = config_path("train", f"{safe}.yaml")
    if not path.is_file():
        known = ", ".join(p.stem for p in config_path("train").glob("*.yaml"))
        raise ConfigError(f"unknown training recipe {family!r}; available: {known}")
    return load(path, TrainRecipe)


def load_profile_override(profile: GpuProfile | str) -> ProfileOverride:
    """``configs/train/overrides/<profile>.yaml``, with a safe default.

    ``auto`` resolves to the detected profile rather than falling back to a
    default, so an unspecified config still lands on the right GPU tier.
    """
    if profile is GpuProfile.AUTO or str(profile).lower() == GpuProfile.AUTO.value:
        profile = detect_profile()

    value = profile.value if isinstance(profile, GpuProfile) else str(profile).lower()
    path = config_path("train", "overrides", f"{value}.yaml")
    if not path.is_file():
        return ProfileOverride(profile=_coerce_profile(value))
    return load(path, ProfileOverride)


def _coerce_profile(value: str) -> GpuProfile:
    try:
        return GpuProfile(value.lower())
    except ValueError as exc:
        known = ", ".join(p.value for p in GpuProfile)
        raise ConfigError(f"unknown GPU profile {value!r}; available: {known}") from exc

def load_matrix() -> ExperimentMatrix:
    return load(config_path("matrix.yaml"), ExperimentMatrix)


def load_rules() -> DroneRules:
    return load(config_path("rules", "drone_rules.yaml"), DroneRules)


def load_coverage_map() -> CoverageMap:
    return load(config_path("coverage", "coverage_map.yaml"), CoverageMap)


def load_settings() -> AppSettings:
    """``configs/app.yaml``, falling back to defaults when absent."""
    path = config_path("app.yaml")
    if not path.is_file():
        return AppSettings()
    return load(path, AppSettings)


def settings_with(**overrides: Any) -> AppSettings:
    """Settings with ad-hoc overrides - used by the CLI and the API."""
    base = load_settings().model_dump()
    base.update({k: v for k, v in overrides.items() if v is not None})
    return AppSettings.model_validate(base)


def detect_profile() -> GpuProfile:
    """Infer the GPU profile from the installed torch build and the live device.

    This is what lets one repo work on a GTX 1070 (sm_61) and a 5090 (sm_120)
    without editing config. ``ANTI_UAV_PROFILE`` overrides the detection.
    """
    override = os.environ.get(f"{_ENV_PREFIX}PROFILE")
    if override:
        return _coerce_profile(override)

    try:
        import torch
    except ImportError:
        return GpuProfile.CPU

    if not torch.cuda.is_available():
        return GpuProfile.CPU

    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if capability < 70:
        return GpuProfile.PASCAL
    if capability < 75:
        return GpuProfile.VOLTA  # sm_70: fp16 tensor cores, no bf16
    if capability < 80:
        return GpuProfile.TURING  # sm_75: fp16 tensor cores, the Kaggle T4
    if capability < 90:
        return GpuProfile.AMPERE
    if capability < 100:
        return GpuProfile.ADA
    return GpuProfile.BLACKWELL


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Read an ultralytics ``args.yaml`` written next to a run's weights."""
    return read_json(path) if str(path).endswith(".json") else _read_yaml(Path(path))


__all__ = [
    "ConfigError",
    "config_dir",
    "config_path",
    "detect_profile",
    "invalidate",
    "load",
    "load_coverage_map",
    "load_dataset",
    "load_matrix",
    "load_profile_override",
    "load_recipe",
    "load_registry",
    "load_rules",
    "load_settings",
    "load_yaml",
    "settings_with",
]
