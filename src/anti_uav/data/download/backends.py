"""Dataset acquisition.

One dispatcher, several backends, because the four datasets are distributed four
different ways:

    dvb       Kaggle           needs an account + ~/.kaggle/kaggle.json
    mavvid    Bitbucket        public git, no auth
    antiuav   Google Drive     password-protected archive (gdown)
              ModelScope       public fallback, no auth
              Baidu Pan        IR-only variants, account required
    mmuav     Baidu Pan        account + captcha, 400 GB

Design points that matter in practice:

* ``--dry-run`` first, always. Every backend reports the plan and the size before
  touching the network, because two of these are large enough to fill a disk.
* Downloads are resumable where the backend allows it, and always land in a
  ``.part`` file that is only promoted to its final name after the size check
  passes. An interrupted transfer therefore never looks like a complete one.
* Whatever lands gets a SHA-256 recorded in ``data/manifests/``. Downstream
  conversion never re-reads a byte it did not hash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config.schema import DatasetSpec, DownloadSource, Modality, SourceKind
from ...utils.io import dir_size_bytes, extract_any, human_bytes, require_free_space, sha256_file
from ...utils.logging import get_logger
from ...utils.paths import dataset_root

log = get_logger(__name__)


@dataclass(slots=True)
class DownloadPlan:
    """What a download is about to do, before it does any of it."""

    dataset: str
    variant: str
    source_label: str
    kind: SourceKind
    url: str
    destination: Path
    approx_size_gb: float
    needs_credentials: bool
    notes: str = ""
    #: Extraction code for password-protected archives (Anti-UAV uses `sagx`).
    #: Carried on the plan rather than re-read from the registry so that a plan
    #: is fully self-describing: `describe()` can print it and a backend can use
    #: it without needing the spec back.
    password: str | None = None

    @property
    def approx_bytes(self) -> int:
        return int(self.approx_size_gb * (1024**3))

    def describe(self) -> str:
        creds = " [NEEDS CREDENTIALS]" if self.needs_credentials else ""
        code = f"\n    code     : {self.password}" if self.password else ""
        return (
            f"{self.dataset}/{self.variant} <- {self.source_label}\n"
            f"    kind     : {self.kind.value}\n"
            f"    url      : {self.url}\n"
            f"    into     : {self.destination}\n"
            f"    approx   : {human_bytes(self.approx_bytes)}{creds}{code}\n"
            f"    notes    : {self.notes.strip() or '-'}"
        )


@dataclass(slots=True)
class DownloadResult:
    dataset: str
    variant: str
    source: str
    status: str  # completed | skipped | failed | needs_credentials
    path: Path | None = None
    bytes_downloaded: int = 0
    sha256: str | None = None
    duration_s: float = 0.0
    error: str = ""
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "skipped"}


class DownloadBackend(ABC):
    """One way of fetching one dataset."""

    kind: SourceKind

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """``(can_run_now, reason_if_not)``. Checked before every attempt so the
        failure message names the missing credential instead of a stack trace."""

    @abstractmethod
    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        ...

    # -- shared helpers ----------------------------------------------------- #

    @staticmethod
    def _promote(part: Path, final: Path) -> Path:
        """Rename a completed ``.part`` file into place."""
        final.parent.mkdir(parents=True, exist_ok=True)
        part.replace(final)
        return final

    @staticmethod
    def _already_have(plan: DownloadPlan, pattern: str = "*") -> Path | None:
        """Return an existing non-empty payload if one is already present."""
        matches = [p for p in plan.destination.glob(pattern) if p.is_file() and p.stat().st_size > 0]
        if matches:
            return max(matches, key=lambda p: p.stat().st_size)
        return None


# --------------------------------------------------------------------------- #
# git backends (Bitbucket / GitHub style)
# --------------------------------------------------------------------------- #


class GitBackend(DownloadBackend):
    """Shallow clone. MAV-VID's Bitbucket mirror is a plain git repo of images."""

    def available(self) -> tuple[bool, str]:
        if shutil.which("git") is None:
            return False, "git is not on PATH"
        return True, ""

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        started = time.monotonic()
        target = plan.destination / "repo"
        if (target / ".git").is_dir() or any(target.iterdir() if target.exists() else []):
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                path=target, duration_s=0.0,
                error="repo directory already populated",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", plan.url, str(target)]
        log.info("cloning", extra={"dataset": plan.dataset, "url": plan.url})
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if completed.returncode != 0:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started,
                error=f"git clone exited {completed.returncode}: {completed.stderr[-400:]}",
            )
        size = dir_size_bytes(target)
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=target, bytes_downloaded=size, duration_s=time.monotonic() - started,
        )


class KaggleBackend(DownloadBackend):
    """Kaggle CLI. Requires an accepted rules file, not just a token.

    The CLI accepts several credential shapes and which one you have depends on
    how old your account is and which SDK it ships. This backend gates on all of
    them rather than a subset, because gating on the wrong subset reports
    "unauthenticated" on machines that are in fact perfectly able to download.

    Read straight off ``kaggle.api.KaggleApi._load_config``, whose precedence is:

    1. ``KAGGLE_API_TOKEN`` (a token, or a path to a file holding one)
    2. ``~/.kaggle/access_token`` / ``.txt``
    3. inside a Kaggle notebook, ``KAGGLE_API_V1_TOKEN`` via the data proxy
    4. legacy API key: ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``, or ``kaggle.json``
    5. anonymous, for commands that allow it

    Two of those were missing here and both bite in practice. ``KAGGLE_USERNAME``
    and ``KAGGLE_KEY`` are the pair the Kaggle UI hands you and the pair a notebook
    gets from its Secrets panel; the CLI folds every ``KAGGLE_*`` variable into
    its config and authenticates from them, while this gate ignored them and
    refused a download that was about to work. The notebook-proxy path matters
    because that is the one credential a Kaggle kernel already has.
    """

    kind = SourceKind.KAGGLE

    def credential_source(self) -> str:
        """Name the credential this machine has, or ``""`` if it has none.

        Reporting *which* shape was found matters: "unauthenticated" on a box
        where the CLI would have worked sends you looking in the wrong place.

        The order mirrors the CLI, and it is not arbitrary. The CLI reads its
        config file first and then overlays environment variables, so an env var
        beats a file for the same key - and it tries access-token auth before the
        legacy API key. Checking the files first would therefore report a stale
        ``kaggle.json`` when a fresh ``KAGGLE_USERNAME``/``KAGGLE_KEY`` pair was
        right there in the environment, which is the exact failure mode this
        method exists to stop.
        """
        if os.environ.get("KAGGLE_API_TOKEN"):
            return "KAGGLE_API_TOKEN"

        kaggle_dir = Path.home().joinpath(".kaggle")
        for name in ("access_token", "access_token.txt"):
            if (kaggle_dir / name).is_file():
                return f"~/.kaggle/{name}"

        # Set by Kaggle itself inside a kernel; no user setup required.
        if os.environ.get("KAGGLE_API_V1_TOKEN"):
            return "KAGGLE_API_V1_TOKEN (Kaggle notebook proxy)"

        # The CLI folds every KAGGLE_* variable into its config, so this pair is
        # the legacy API-key credential and works from a notebook's Secrets panel.
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return "KAGGLE_USERNAME + KAGGLE_KEY"

        if (kaggle_dir / "kaggle.json").is_file():
            return "~/.kaggle/kaggle.json"

        return ""

    def _credentials_present(self) -> bool:
        return bool(self.credential_source())

    def available(self) -> tuple[bool, str]:
        if shutil.which("kaggle") is None:
            return False, "kaggle CLI not found - pip install 'anti-uav[download]'"
        source = self.credential_source()
        if source:
            return True, ""
        return False, (
            "no Kaggle credentials found. Any ONE of these works:\n"
            "    export KAGGLE_API_TOKEN=<token>            (Account > Settings > API)\n"
            "    export KAGGLE_USERNAME=<user>              # legacy pair, both required\n"
            "    export KAGGLE_KEY=<key>\n"
            "    kaggle login --accept-dictionary            # writes ~/.kaggle/kaggle.json\n"
            "Note that Kaggle also requires you to accept the dataset's rules in the "
            "browser before an authenticated API call will return data; that step has "
            "no API equivalent."
        )

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        started = time.monotonic()
        plan.destination.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", plan.url, "-p", str(plan.destination), "--unzip"]
        log.info("kaggle download", extra={"dataset": plan.dataset, "ref": plan.url})
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if completed.returncode != 0:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started,
                error=f"kaggle exited {completed.returncode}: {completed.stderr[-400:]}",
            )
        size = dir_size_bytes(plan.destination)
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=plan.destination, bytes_downloaded=size, duration_s=time.monotonic() - started,
        )


class GdriveBackend(DownloadBackend):
    """Google Drive via gdown. Handles the password-protected archive."""

    kind = SourceKind.GDRIVE

    def available(self) -> tuple[bool, str]:
        try:
            import gdown  # noqa: F401
        except ImportError:
            return False, "gdown not installed - pip install 'anti-uav[download]'"
        return True, ""

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        import gdown

        started = time.monotonic()
        plan.destination.mkdir(parents=True, exist_ok=True)
        existing = self._already_have(plan, "*.zip")
        if existing is not None:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                path=existing, bytes_downloaded=existing.stat().st_size,
                duration_s=0.0, error="archive already present",
            )

        require_free_space(plan.destination, plan.approx_bytes, context=plan.source_label)

        part = plan.destination / (Path(plan.url).name or "download.zip.part")
        final = part.with_suffix("").with_suffix(".zip")

        url = plan.url
        if not url.startswith("http"):
            url = f"https://drive.google.com/uc?id={url}"

        log.info("gdown fetch", extra={"dataset": plan.dataset, "password": bool(plan.password)})
        try:
            # gdown 6.x has no `password=` or `fuzzy=` parameter. The Drive file
            # itself is not encrypted - it is a plain zip - so the code below is
            # only for Baidu-style "extraction code" prompts and is reported in
            # describe()/the notes rather than passed to gdown. Passing it raised
            # TypeError on every call.
            path = gdown.download(
                url,
                str(part),
                quiet=False,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001 - gdown raises bare exceptions
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        if path is None:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started,
                error="gdown returned nothing - check the file id and the archive password",
            )

        # gdown returns a str path here, but its type also admits BinaryIO and
        # GoogleDriveFileToDownload for other call styles. `part` is what we asked
        # it to write, so fall back to it rather than coercing a file object.
        landed = path if isinstance(path, (str, Path)) else part
        promoted = self._promote(Path(landed), final)
        digest = sha256_file(promoted)
        size = promoted.stat().st_size
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=promoted, bytes_downloaded=size, sha256=digest,
            duration_s=time.monotonic() - started,
        )


class ModelScopeBackend(DownloadBackend):
    """ModelScope. No auth for Anti-UAV600 - the easiest large download here."""

    kind = SourceKind.MODELSCOPE

    def available(self) -> tuple[bool, str]:
        try:
            import modelscope  # noqa: F401
        except ImportError:
            return False, "modelscope not installed - pip install 'anti-uav[download]'"
        return True, ""

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        started = time.monotonic()
        plan.destination.mkdir(parents=True, exist_ok=True)
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=str(exc),
            )

        log.info("modelscope snapshot", extra={"dataset": plan.dataset, "repo": plan.url})
        try:
            path = snapshot_download(plan.url, local_dir=str(plan.destination))
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=Path(path), bytes_downloaded=dir_size_bytes(path),
            duration_s=time.monotonic() - started,
        )


class BaiduBackend(DownloadBackend):
    """Baidu Pan.

    Requires a logged-in ``BDUSS`` cookie AND usually an interactive captcha, so
    this backend deliberately does not attempt to solve the captcha. Instead it
    checks for the cookie, reports the exact next step, and refuses cleanly.

    Getting the cookie (once, in a normal browser):
      1. Sign in at pan.baidu.com
      2. Open the share link, accept the code
      3. F12 -> Network -> any request -> Request Headers -> copy `BDUSS=...`
      4. Save it to   data/raw/.baidu_cookies.json   (gitignored)
         as {"BDUSS": "the-value"}
      5. Re-run this command. It will list the files and sizes so you can
         download them with the Baidu client or aria2, then re-run with
         --skip-download to register the local copy.

    A captcha will still stop the transfer. Batch small file groups instead of one
    400 GB folder, and expect to resume more than once - Baidu throttles hard.
    """

    kind = SourceKind.BAIDU

    COOKIE_FILE = ".baidu_cookies.json"

    def cookie_path(self) -> Path:
        return dataset_root("_credentials") / self.COOKIE_FILE

    def available(self) -> tuple[bool, str]:
        path = self.cookie_path()
        if path.is_file():
            return True, ""
        return False, (
            f"Baidu cookie not found at {path}.\n"
            "  Baidu Pan has no anonymous download API, so this cannot be automated.\n"
            "  See the 'MM-UAV: acquisition notes' section of docs/DATASETS.md for the\n"
            "  cookie + captcha walkthrough, and the subsetting recommendation\n"
            "  (RGB frames + gt_rgb for ~150 sequences is ~12 GB instead of 400 GB)."
        )

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        started = time.monotonic()
        ok, reason = self.available()
        if not ok:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "needs_credentials",
                duration_s=0.0, error=reason,
            )

        # Even with a cookie, multi-GB Baidu transfers hit an interactive
        # captcha. Rather than half-implement an API that will fail, we report
        # the plan and let the operator fetch with the official client.
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "needs_credentials",
            path=None, duration_s=time.monotonic() - started,
            error=(
                "Authenticated, but Baidu will still require an interactive captcha.\n"
                f"  Share link : {plan.url}\n"
                f"  Extraction code: {plan.password or '-'}\n"
                f"  Target     : {plan.destination}\n"
                f"  Pull only  : {plan.source_label}\n"
                "  Recommended subset: train/0001..0150/{rgb_frame,gt_rgb}/ (~12 GB)\n"
                "  Then re-run: anti-uav download --dataset mmuav --skip-download"
            ),
            extras={"share_url": plan.url, "code": plan.password, "dest": str(plan.destination)},
        )


class ManualBackend(DownloadBackend):
    """A source we deliberately do not automate.

    The Drone-vs-Bird challenge videos are distributed on request by the
    organisers; the public GitHub repo carries annotations only. Reporting this
    as a distinct status beats letting the command appear to succeed.
    """

    kind = SourceKind.MANUAL

    def available(self) -> tuple[bool, str]:
        return True, ""

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "needs_credentials",
            error=(
                f"Manual acquisition required. {plan.notes}\n"
                f"  Reference  : {plan.url}\n"
                f"  Target     : {plan.destination}\n"
                "  Then re-run: anti-uav download --dataset <alias> --skip-download"
            ),
        )


def _looks_like_html(prefix: bytes) -> bool:
    """Sniff a leading chunk for a doctype/``<html``, as a backstop.

    Some mirrors serve a page with a generic content-type, so the header alone
    is not enough. Takes bytes rather than the response, deliberately: an earlier
    version pulled the first chunk from the response and then handed the same
    iterator to the write loop, so a successful download wrote *nothing* after a
    clean sniff. Peeking at bytes the caller already holds cannot do that.
    """
    if not prefix:
        return False
    stripped = prefix[:512].lstrip().lower()
    return (
        stripped.startswith(b"<!doctype html")
        or stripped.startswith(b"<html")
        or b"<html" in stripped[:256]
    )


class HttpBackend(DownloadBackend):
    """Plain HTTP(S) with resume. No auth.

    For a single file at a stable URL - a tarball, an archive, a raw blob. NOT
    for a repository or directory URL: those return the hosting site's HTML
    landing page, which this backend now refuses rather than saving.
    """

    kind = SourceKind.HTTP

    def available(self) -> tuple[bool, str]:
        return True, ""

    def fetch(self, plan: DownloadPlan) -> DownloadResult:
        import time

        import requests  # type: ignore[import-untyped]

        started = time.monotonic()
        plan.destination.mkdir(parents=True, exist_ok=True)
        name = plan.url.rstrip("/").split("/")[-1] or "download.bin"
        part = plan.destination / f"{name}.part"
        final = plan.destination / name

        existing = self._already_have(plan, name)
        if existing is not None:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                path=existing, bytes_downloaded=existing.stat().st_size,
                duration_s=0.0, error="file already present",
            )

        require_free_space(plan.destination, plan.approx_bytes, context=plan.source_label)

        try:
            with requests.get(plan.url, stream=True, timeout=60, allow_redirects=True) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                mode = "ab" if part.exists() else "wb"
                written = part.stat().st_size if mode == "ab" else 0

                # Pull the first chunk before opening the file, so the HTML check
                # runs before anything is written and the chunk is still ours to
                # write if the payload turns out to be genuine.
                chunks = response.iter_content(chunk_size=1 << 20)
                first = next(chunks, b"")

                # A repo or directory URL returns the hosting site's HTML landing
                # page, not the artefact. Saving that and reporting `completed` is
                # worse than failing: it is a success message for zero data, and
                # the junk file lands in the dataset root where a converter may
                # later mistake it for input.
                content_type = response.headers.get("content-type", "").lower()
                if "html" in content_type or _looks_like_html(first):
                    part.unlink(missing_ok=True)
                    return DownloadResult(
                        plan.dataset, plan.variant, plan.source_label, "failed",
                        duration_s=time.monotonic() - started,
                        error=(
                            f"{plan.url} returned an HTML page (content-type "
                            f"{content_type or 'unset'}), not a downloadable artefact. "
                            f"This URL is a web page or repository, not a file. If it is "
                            f"a git repository, declare the source `kind: bitbucket` "
                            f"(clones it) or `kind: manual` (for reference material). "
                            f"Nothing was written."
                        ),
                    )

                with part.open(mode) as handle:
                    if first:
                        handle.write(first)
                        written += len(first)
                    for chunk in chunks:
                        handle.write(chunk)
                        written += len(chunk)
        except Exception as exc:  # noqa: BLE001
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        promoted = self._promote(part, final)
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=promoted, bytes_downloaded=promoted.stat().st_size,
            sha256=sha256_file(promoted), duration_s=time.monotonic() - started,
            extras={"declared_bytes": total},
        )


_BACKENDS: dict[SourceKind, type[DownloadBackend]] = {
    SourceKind.BITBUCKET: GitBackend,
    SourceKind.KAGGLE: KaggleBackend,
    SourceKind.GDRIVE: GdriveBackend,
    SourceKind.MODELSCOPE: ModelScopeBackend,
    SourceKind.BAIDU: BaiduBackend,
    SourceKind.MANUAL: ManualBackend,
    SourceKind.HTTP: HttpBackend,
    SourceKind.HF: HttpBackend,  # huggingface_hub handled specially below
}


def backend_for(kind: SourceKind) -> DownloadBackend:
    cls = _BACKENDS.get(kind)
    if cls is None:
        raise ValueError(f"no backend registered for {kind}")
    return cls()


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def plan_for(
    spec: DatasetSpec,
    *,
    variant: str | None = None,
    source_label: str | None = None,
    max_sequences: int | None = None,
    modalities: list[Modality] | None = None,
) -> list[DownloadPlan]:
    """Build the download plan for a dataset.

    One plan per selectable source. Sources are filtered by ``source_label`` when
    given, which is how you pick a specific mirror without reordering the YAML.
    """
    chosen_variant = variant or spec.default_variant or (spec.variants[0] if spec.variants else "full")
    if spec.variants and chosen_variant not in spec.variants and chosen_variant != "full":
        raise ValueError(
            f"{spec.alias}: variant {chosen_variant!r} not in {spec.variants}"
        )

    plans: list[DownloadPlan] = []
    for source in spec.sources:
        if source_label and source.label != source_label:
            continue
        notes = source.notes
        if max_sequences:
            notes += (
                f" Subsetting to ~{max_sequences} sequences "
                f"(modalities: {', '.join(m.value for m in (modalities or spec.modalities))})."
            )
        plans.append(
            DownloadPlan(
                dataset=spec.alias,
                variant=chosen_variant,
                source_label=source.label,
                kind=source.kind,
                url=source.url,
                destination=dataset_root(spec.alias) / chosen_variant,
                approx_size_gb=source.approx_size_gb,
                needs_credentials=source.needs_credentials,
                notes=notes,
                password=source.password,
            )
        )
    return plans


def download(
    spec: DatasetSpec,
    *,
    variant: str | None = None,
    source_label: str | None = None,
    extract: bool = True,
    dry_run: bool = False,
    max_sequences: int | None = None,
    modalities: list[Modality] | None = None,
) -> Iterator[DownloadResult]:
    """Fetch a dataset from every matching source, yielding one result each."""
    plans = plan_for(
        spec,
        variant=variant,
        source_label=source_label,
        max_sequences=max_sequences,
        modalities=modalities,
    )
    if not plans:
        raise ValueError(
            f"{spec.alias}: no source matched"
            + (f" label={source_label!r}" if source_label else "")
        )

    for plan in plans:
        backend = backend_for(plan.kind)
        can_run, reason = backend.available()

        if not can_run:
            yield DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "needs_credentials",
                error=reason, extras={"plan": plan.describe()},
            )
            continue

        if dry_run:
            yield DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                error="dry run", extras={"plan": plan.describe()},
            )
            continue

        result = backend.fetch(plan)
        if result.ok and extract and result.path is not None and result.path.is_file():
            dest = plan.destination / "extracted"
            try:
                extract_any(result.path, dest)
                result.extras["extracted_to"] = str(dest)
            except Exception as exc:  # noqa: BLE001
                result.status = "completed"
                result.extras["extract_error"] = f"{type(exc).__name__}: {exc}"
        yield result


def verify_payload(
    spec: DatasetSpec,
    *,
    variant: str | None = None,
) -> dict[str, object]:
    """Hash whatever is already on disk for a dataset.

    Run this before converting. The manifest it produces is what the conversion
    report cites, so a report can always be traced back to the exact bytes that
    were read.
    """
    chosen = variant or spec.default_variant or "full"
    root = dataset_root(spec.alias) / chosen
    if not root.exists():
        return {"dataset": spec.alias, "variant": chosen, "exists": False, "files": []}

    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        # Hashing 400 GB of frames one at a time is not useful. Hash archives and
        # metadata; for bulk frame trees record a manifest digest instead.
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"} or path.stat().st_size < 4 << 20:
            files.append(
                {
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": None})

    return {
        "dataset": spec.alias,
        "variant": chosen,
        "exists": True,
        "root": str(root),
        "total_bytes": dir_size_bytes(root),
        "file_count": len(files),
        "files": files,
    }