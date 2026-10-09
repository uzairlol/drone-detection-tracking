"""Environment verification.

The failure this exists to prevent is quiet and expensive. A torch wheel
installed by default ``pip install torch`` on a Pascal card has **no sm_61
kernels** - CUDA 12.8 removed Maxwell and Pascal outright. It imports fine,
reports ``torch.cuda.is_available() == True``, and then fails hours into a training
run with ``no kernel image is available for execution on the device``.

So this checks the arch list, not just availability, and it is run before any
training. Three profiles matter here:

============  =========  =========================================
GPU           capability  torch build needed
============  =========  =========================================
GTX 1070      sm_61       2.6.x + cu126 (or older). NOT 2.7+, NOT cu128
T4 / V100     sm_75/70    anything from 2.0+ with cu118+
RTX 4090      sm_89       2.7+ + cu126 or cu128
RTX 5090      sm_120      2.7+ + cu128 or cu130. cu126 will NOT work
RTX 3090      sm_86       anything from 2.0+
============  =========  =========================================
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from .config.schema import GpuProfile
from .utils.logging import get_logger

log = get_logger(__name__)

#: profile -> the NEWEST CUDA runtime line that still ships kernels for it.
#: Read as a CEILING, not a floor: a build *above* this value is the problem,
#: because CUDA dropped the architecture rather than the wheel being too old.
#: CUDA 12.8 removed Maxwell/Pascal (sm_50-sm_62); CUDA 13.x removed Volta
#: (sm_70) too. Turing (sm_75) and newer survive into CUDA 13.
#:
#: Only the two profiles whose architecture CUDA actually deleted are enforced -
#: for everything else this is documentation, and the arch-kernels check against
#: ``torch.cuda.get_arch_list()`` is the real test.
_PROFILE_CUDA: dict[GpuProfile, str] = {
    GpuProfile.PASCAL: "12.6",
    GpuProfile.VOLTA: "12.9",
    GpuProfile.TURING: "13.9",
    GpuProfile.AMPERE: "13.9",
    GpuProfile.ADA: "13.9",
    GpuProfile.BLACKWELL: "12.8",  # floor, not ceiling - see docs/ENVIRONMENT.md
    GpuProfile.CPU: "",
}

#: Profiles where a build ABOVE the value in _PROFILE_CUDA is the failure.
#: Blackwell is the exception: it needs a build at or above its floor, and the
#: arch-kernels check is what catches it.
_PROFILE_CUDA_CEILING: frozenset[GpuProfile] = frozenset(
    {GpuProfile.PASCAL, GpuProfile.VOLTA}
)


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    #: ``error`` blocks training; ``warn`` does not.
    level: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "level": self.level}


@dataclass(slots=True)
class VerifyReport:
    checks: list[Check] = field(default_factory=list)
    profile: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.level == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile": self.profile,
            "environment": self.environment,
            "checks": [c.to_dict() for c in self.checks],
        }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def verify(profile: GpuProfile | str | None = None) -> VerifyReport:
    """Run every environment check."""
    from .config.loader import _coerce_profile, detect_profile

    report = VerifyReport()
    report.environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    _check_python(report)
    _check_packages(report)
    _check_torch(report, _coerce_profile(profile) if profile is not None else detect_profile())
    _check_ultralytics(report)
    _check_base_weights(report)
    _check_optional(report)
    _check_disk(report)

    return report


#: Interpreter range this project supports. Kept in step with
#: ``requires-python`` in pyproject.toml by a test, because a guard that drifts
#: from the declared range blocks working environments for no reason - which is
#: exactly what happened when this said < 3.13 while Kaggle shipped 3.13.
#:
#: The floor is a hard requirement: the code and every pinned dependency are
#: guaranteed there. The ceiling is a statement about what has been *tested*,
#: not about what exists - most pinned deps publish wheels a release or two
#: behind, so 3.14 is refused until it is exercised.
#:
#: This deliberately does NOT assert anything about specific packages having or
#: lacking wheels for a given version. Those claims rot: lapx and opencv-python
#: both lacked 3.13 wheels once and have them now, and a stale assertion is worse
#: than no assertion because it is believed. The load-bearing question - "can
#: the dependencies actually run here?" - is answered by _check_packages and by
#: the fact that this module imported at all.
PYTHON_MIN = (3, 11)
PYTHON_MAX = (3, 14)


def _check_python(report: VerifyReport) -> None:
    version = sys.version_info
    current = (version.major, version.minor)
    too_old = current < PYTHON_MIN
    too_new = current >= PYTHON_MAX
    ok = not (too_old or too_new)
    detail = f"{version.major}.{version.minor}.{version.micro}"
    if too_old:
        detail += (
            f" - below the supported floor {PYTHON_MIN[0]}.{PYTHON_MIN[1]}; "
            f"use 3.11 or newer"
        )
    elif too_new:
        detail += (
            f" - at or above the tested ceiling {PYTHON_MAX[0]}.{PYTHON_MAX[1]}; "
            f"most pinned deps publish wheels a release or two behind, so this "
            f"version is untested. Use 3.11-3.13."
        )
    report.checks.append(
        Check("python", ok, detail, level="error" if not ok else "info")
    )


def _check_packages(report: VerifyReport) -> None:
    required = {
        "numpy": "1.24",
        "pydantic": "2.6",
        "PyYAML": "6.0",
        "lapx": "0.5",
        "ultralytics": "8.3",
    }
    missing: list[str] = []
    for name, minimum in required.items():
        version = package_version(name)
        if version is None:
            missing.append(f"{name}>={minimum}")
    if missing:
        report.checks.append(
            Check(
                "packages",
                False,
                f"missing: {', '.join(missing)}",
                level="error",
            )
        )
        report.checks[-1].detail += "\n    pip install " + " ".join(f"'{m}'" for m in missing)
    else:
        report.checks.append(
            Check(
                "packages",
                True,
                ", ".join(
                    f"{n}=={package_version(n)}"
                    for n in ("numpy", "pydantic", "ultralytics", "lapx")
                    if package_version(n)
                ),
            )
        )


def _check_torch(report: VerifyReport, profile: GpuProfile) -> None:
    report.profile = profile.value
    try:
        import torch
    except ImportError as exc:
        report.checks.append(Check("torch", False, f"not installed: {exc}"))
        return

    report.environment["torch"] = torch.__version__
    report.environment["torch_cuda_build"] = torch.version.cuda
    report.checks.append(Check("torch", True, f"{torch.__version__}"))

    if profile is GpuProfile.CPU:
        if torch.cuda.is_available():
            report.checks.append(
                Check(
                    "profile",
                    False,
                    "profile forced to cpu but CUDA is available; drop --profile to use it",
                    level="warn",
                )
            )
        else:
            report.checks.append(
                Check(
                    "profile",
                    True,
                    "cpu profile: training and GPU inference unavailable here (expected on a "
                    "dev box - data prep, the API, the UI and the test suite all work)",
                    level="info",
                )
            )
        return

    if not torch.cuda.is_available():
        report.checks.append(
            Check(
                "cuda",
                False,
                f"profile {profile.value} but torch.cuda.is_available() is False",
            )
        )
        report.checks[-1].detail += _install_hint(profile, torch.__version__, None)
        return

    report.checks.append(Check("cuda", True, f"CUDA build {torch.version.cuda}"))

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"sm_{props.major}{props.minor}",
                "vram_gb": round(props.total_memory / 1024**3, 1),
            }
        )
    report.environment["devices"] = devices

    if not devices:
        report.checks.append(Check("devices", False, "no CUDA devices reported"))
        return

    for device in devices:
        arch = device["capability"]
        report.checks.append(
            Check("device", True, f"{device['name']} ({arch}, {device['vram_gb']} GB)")
        )

        arch_list = torch.cuda.get_arch_list()
        if arch_list and arch not in arch_list:
            report.checks.append(
                Check(
                    "arch-kernels",
                    False,
                    f"{device['name']} needs {arch}, but this torch build has {arch_list}",
                )
            )
            report.checks[-1].detail += _install_hint(profile, torch.__version__, torch.version.cuda)
        elif arch_list:
            report.checks.append(Check("arch-kernels", True, f"{arch} present"))

    required = _PROFILE_CUDA.get(profile, "")
    build = torch.version.cuda or ""
    if required and build:
        def as_tuple(value: str) -> tuple[int, ...]:
            return tuple(int(p) for p in value.split(".")[:2] if p.isdigit())

        try:
            too_new = as_tuple(build) > as_tuple(required)
            needs_floor = profile in {GpuProfile.BLACKWELL}
            too_old = as_tuple(build) < as_tuple(required)
            if profile in _PROFILE_CUDA_CEILING and too_new:
                # CUDA removed this architecture. The wheel imports fine, CUDA
                # reports available, and the run dies at the first kernel launch
                # with "no kernel image is available for execution on the device".
                report.checks.append(
                    Check(
                        "cuda-version",
                        False,
                        f"CUDA {build} is newer than {required}, which is the last line "
                        f"supporting {profile.value}",
                    )
                )
                report.checks[-1].detail += _install_hint(profile, torch.__version__, build)
            elif needs_floor and too_old:
                report.checks.append(
                    Check(
                        "cuda-version",
                        False,
                        f"CUDA {build} is older than {required}, which {profile.value} needs",
                    )
                )
                report.checks[-1].detail += _install_hint(profile, torch.__version__, build)
            else:
                report.checks.append(
                    Check("cuda-version", True, f"CUDA {build} is compatible with {profile.value}")
                )
        except ValueError:
            report.checks.append(
                Check("cuda-version", True, f"CUDA {build} (unparsed)", level="warn")
            )


def _install_hint(profile: GpuProfile, torch_version: str, cuda: str | None) -> str:
    """The exact command that fixes an incompatible wheel."""
    if profile is GpuProfile.PASCAL:
        return (
            "\n    CUDA 12.8+ removed Maxwell/Pascal kernels. Install the last build that has "
            "them:\n      pip install --upgrade 'torch==2.6.0' 'torchvision==0.21.0' "
            "--index-url https://download.pytorch.org/whl/cu126"
        )
    if profile is GpuProfile.VOLTA:
        return (
            "\n    CUDA 13.x removed Volta (sm_70). Install the last 12.x line that has it:\n"
            "      pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu126"
        )
    if profile is GpuProfile.BLACKWELL:
        return (
            "\n    sm_120 needs CUDA 12.8+:\n      pip install --upgrade torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    return (
        f"\n    current: torch {torch_version}, CUDA {cuda}\n"
        f"    see docs/ENVIRONMENT.md for the compatibility matrix"
    )


def _check_ultralytics(report: VerifyReport) -> None:
    version = package_version("ultralytics")
    if version is None:
        report.checks.append(
            Check(
                "ultralytics",
                False,
                "not installed - training and inference unavailable",
                level="warn",
            )
        )
        report.checks[-1].detail += "\n    pip install 'anti-uav[train]'"
        return
    report.checks.append(Check("ultralytics", True, version))


def _check_base_weights(report: VerifyReport) -> None:
    """Report the recipes' starting checkpoints.

    Deliberately ``warn``, never ``error``. A missing checkpoint is a
    five-minute download and the machine is still perfectly capable of doing data
    prep, evaluation, the API and the test suite without it - so it must not
    block those. But it *will* block training, and finding that out after the
    dataset is built is the expensive way to find out, hence a check.
    """
    from .detection.weights import targets as weight_targets

    try:
        found = weight_targets()
    except Exception as exc:  # a malformed recipe should not break verify-env
        report.checks.append(
            Check("base-weights", False, f"could not read recipes: {exc}", level="warn")
        )
        return

    if not found:
        return

    absent = [t for t in found if not t.present]
    if not absent:
        report.checks.append(
            Check(
                "base-weights",
                True,
                ", ".join(f"{t.name} ({t.size_mb:.0f} MB)" for t in found),
                level="info",
            )
        )
        return

    have = [t.name for t in found if t.present]
    report.checks.append(
        Check(
            "base-weights",
            False,
            "missing " + ", ".join(t.name for t in absent)
            + (f"; have {', '.join(have)}" if have else ""),
            level="warn",
        )
    )
    report.checks[-1].detail += (
        "\n    training cannot start until these are on disk."
        "\n    anti-uav fetch-weights"
    )


def _check_optional(report: VerifyReport) -> None:
    optional = {
        "gdown": ("download gdrive (Anti-UAV300)", "warn"),
        "kaggle": ("download kaggle (drone-vs-bird)", "warn"),
        "modelscope": ("download modelscope (Anti-UAV600)", "warn"),
        "onnx": ("ONNX export", "warn"),
        "onnxruntime": ("ONNX graph inspection", "warn"),
        "tensorrt": ("TensorRT engine build on this host", "info"),
    }
    missing = [
        f"{name} ({why})"
        for name, (why, _level) in optional.items()
        if package_version(name) is None
    ]
    present = [
        f"{name}=={package_version(name)}"
        for name in optional
        if package_version(name) is not None
    ]
    if missing:
        report.checks.append(
            Check("optional", False, "absent: " + "; ".join(missing), level="info")
        )
    if present:
        report.checks.append(Check("optional-present", True, ", ".join(present), level="info"))


def _check_disk(report: VerifyReport) -> None:
    from .utils.paths import subdir

    try:
        free = shutil.disk_usage(subdir("data")).free
    except OSError as exc:  # pragma: no cover
        report.checks.append(Check("disk", True, f"unknown ({exc})", level="info"))
        return

    gib = free / 1024**3
    ok = gib >= 20
    report.checks.append(
        Check(
            "disk",
            ok,
            f"{gib:.1f} GB free on the data volume",
            level="error" if not ok else "info",
        )
    )
    if not ok:
        report.checks[-1].detail += (
            "\n    MM-UAV is ~400 GB extracted. The recommended subset is "
            "train/0001-0150/{rgb_frame,gt_rgb}\n    at roughly 12 GB - see docs/DATASETS.md."
        )


def format_report(report: VerifyReport) -> str:
    lines = [
        f"python     {report.environment.get('python')}  ({report.environment.get('platform', '')[:48]})",
        f"profile    {report.profile}",
    ]
    for device in report.environment.get("devices", []) or []:
        lines.append(
            f"gpu        {device['index']}: {device['name']}  {device['capability']}  "
            f"{device['vram_gb']} GB"
        )
    if not report.environment.get("devices") and report.profile == "cpu":
        lines.append("gpu        none (CPU profile)")

    lines.append("")
    for check in report.checks:
        marker = "ok  " if check.ok else ("WARN" if check.level != "error" else "FAIL")
        lines.append(f"  [{marker}] {check.name:<14} {check.detail}")

    lines.append("")
    if report.ok:
        lines.append(
            "environment OK"
            + (
                "  (cpu profile: use a GPU box for training)"
                if report.profile == "cpu"
                else "  ready to train"
            )
        )
    else:
        lines.append(f"{len(report.errors)} blocking problem(s). Fix before training.")
    return "\n".join(lines)


def run(*, profile: str | None = None, console: Any = None) -> int:
    """Entry point used by the CLI. Returns a process exit code."""
    resolved: GpuProfile | None
    if profile:
        try:
            resolved = GpuProfile(profile.lower())
        except ValueError as exc:
            resolved = None
            if console:
                console.print(f"[red]unknown profile {profile!r}:[/] {exc}")
    else:
        resolved = None

    report = verify(resolved)
    text = format_report(report)

    if console is None:
        print(text)
    else:
        console.print(text, markup=False)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
