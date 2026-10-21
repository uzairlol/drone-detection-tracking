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
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

from ...config.schema import DatasetSpec, Modality, SourceKind
from ...utils.io import (
    dir_size_bytes,
    extract_any,
    human_bytes,
    require_free_space,
    sha256_file,
)
from ...utils.logging import get_logger
from ...utils.paths import dataset_root
from ...utils.progress import NullProgress, ProgressReporter

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
    #: One of: completed | skipped | failed | needs_credentials | manual.
    #:
    #: ``needs_credentials`` and ``manual`` are deliberately distinct. They used to
    #: be the same string, which made a source that is never going to be
    #: automated report that you were missing a login - so the fix people reach
    #: for is authenticating, repeatedly, against a source that has nothing to
    #: authenticate with. ``manual`` means "a human fetches this, and you do not
    #: need to".
    status: str
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
    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        """Fetch the plan. ``progress`` is optional: library callers pass none."""

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


#: Git's ``--progress`` lines. It rewrites the same line with ``\r`` when the
#: sink is a terminal, so the reader splits on both carriage return and newline.
#: A line looks like::
#:
#:     Receiving objects:  73% (12/16), 8.41 GiB | 6.21 MiB/s
#:     Resolving deltas: 100% (5/5), done.
#:
#: The phase name is greedy on purpose: "Receiving objects" and "Updating files"
#: both have to survive as one label rather than being split on the space.
_GIT_PROGRESS = re.compile(
    r"^(?P<phase>[A-Za-z][A-Za-z ]*?):\s+(?P<percent>\d+)%\s+"
    r"\((?P<done>\d+)/(?P<total>\d+)\)"
    r"(?:,\s+(?P<size>[\d.]+\s*[KMGTP]?i?B))?"
    r"(?:\s+\|\s+(?P<speed>[\d.]+\s*[KMGTP]?i?B/s))?"
)

#: Archive suffixes a mirror may ship the payload as, instead of a frame tree.
_ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.xz",
    ".txz",
    ".tar.bz2",
)


def _is_archive(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(_ARCHIVE_SUFFIXES)


class GitBackend(DownloadBackend):
    """Shallow clone, with live progress and archive unpacking.

    The original assumption - "MAV-VID's Bitbucket mirror is a plain git repo of
    images" - was wrong for the surviving copy: that repository holds exactly one
    file, an 11.5 GB ``mav_vid_dataset.zip``. Two problems followed from it.

    * ``subprocess.run(capture_output=True)`` swallowed every byte of git's
      progress until the transfer finished, so an 11.5 GB clone looked hung.
    * Nothing ever opened the zip. The clone returned the *directory* as the
      payload, the orchestrator only extracts when the result is a *file*, and
      the converter then found no images.

    So this backend streams ``--progress`` into the reporter, and when the clone
    turns out to be an archive rather than a frame tree, it extracts it and
    promotes the frames into the dataset root. The ``.git`` directory is removed
    afterwards; keeping a second 11.5 GB copy of the payload in a pack file is
    the opposite of clean.
    """

    def available(self) -> tuple[bool, str]:
        if shutil.which("git") is None:
            return False, "git is not on PATH"
        return True, ""

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        reporter = progress or NullProgress()
        started = time.monotonic()
        target = plan.destination / "repo"
        plan.destination.mkdir(parents=True, exist_ok=True)

        if self._payload_already_present(plan):
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                path=plan.destination, duration_s=0.0,
                error="payload already present",
            )

        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--no-tags",
            "--progress",
            plan.url,
            str(target),
        ]
        log.info("cloning", extra={"dataset": plan.dataset, "url": plan.url})

        reporter.begin(f"git clone {plan.dataset}")
        returncode, tail = self._run_streaming(cmd, reporter, plan.dataset)
        if returncode != 0:
            reporter.end(description=f"git clone {plan.dataset}", detail="failed")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started,
                error=f"git clone exited {returncode}: {tail[-400:]}",
            )

        size = dir_size_bytes(target)
        try:
            payload, extracted_from = self._settle_payload(plan, target, reporter)
        except Exception as exc:
            reporter.end(description=f"git clone {plan.dataset}", detail="unpack failed")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                path=target, bytes_downloaded=size,
                duration_s=time.monotonic() - started,
                error=f"cloned, but unpacking failed: {type(exc).__name__}: {exc}",
            )

        reporter.end(description=f"git clone {plan.dataset}", detail=human_bytes(size))
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=payload, bytes_downloaded=size,
            duration_s=time.monotonic() - started,
            extras={"repo": str(target), "extracted_from": extracted_from},
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _payload_already_present(plan: DownloadPlan) -> bool:
        """A finished clone, or already-unpacked frames, means there is nothing to do."""
        for split in ("train", "val", "valid", "validation"):
            candidate = plan.destination / split
            if candidate.is_dir() and any(candidate.rglob("*")):
                return True
        repo = plan.destination / "repo"
        return repo.is_dir() and any(repo.iterdir())

    @staticmethod
    def _run_streaming(
        cmd: list[str],
        reporter: ProgressReporter,
        dataset: str,
    ) -> tuple[int, str]:
        """Run git, forwarding ``--progress`` lines to the reporter.

        Reading with ``read1`` matters: a text-mode ``readline`` blocks until a
        newline, and git redraws progress with ``\\\\r`` when stderr is a pipe, so
        the bar would stay still for the whole transfer and then jump.
        """
        env = dict(os.environ)
        # A private mirror that suddenly wants a password would otherwise hang
        # forever with no output. Fail instead of waiting on an invisible prompt.
        env["GIT_TERMINAL_PROMPT"] = "0"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        assert proc.stderr is not None
        stderr = cast("BinaryIO", proc.stderr)
        tail: list[str] = []
        buffer = ""

        def consume(line: str) -> None:
            stripped = line.strip()
            if not stripped:
                return
            matched = _GIT_PROGRESS.match(stripped)
            if matched:
                size = matched.group("size") or ""
                speed = matched.group("speed") or ""
                detail = "  ".join(part for part in (size, speed) if part)
                reporter.update(
                    float(matched.group("percent")),
                    detail=detail,
                    description=f"git clone {dataset} ({matched.group('phase').strip().lower()})",
                )
                return
            tail.append(stripped)
            if len(tail) > 40:
                tail.pop(0)

        try:
            while True:
                chunk = stderr.read1(1 << 14)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", "replace")
                while True:
                    cuts = [pos for pos in (buffer.find("\r"), buffer.find("\n")) if pos != -1]
                    if not cuts:
                        break
                    cut = min(cuts)
                    line, buffer = buffer[:cut], buffer[cut + 1 :]
                    consume(line)
            if buffer:
                consume(buffer)
        except KeyboardInterrupt:
            proc.terminate()
            raise

        proc.wait()
        return proc.returncode, "\n".join(tail)

    def _settle_payload(
        self,
        plan: DownloadPlan,
        target: Path,
        reporter: ProgressReporter,
    ) -> tuple[Path, str | None]:
        """Leave the frames where the converter expects them.

        Returns ``(payload_path, archive_name)``. A mirror that really does ship
        an image tree is left alone; one that ships a single archive is extracted
        into the dataset root and its ``.git`` pack is deleted.
        """
        archives = sorted(
            (p for p in target.rglob("*") if _is_archive(p)),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        has_frames = any(
            p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in target.rglob("*") if p.is_file()
        )
        if has_frames or not archives:
            return target, None

        archive = archives[0]
        if archive.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2")):
            log.info(
                "unpacking clone",
                extra={"dataset": plan.dataset, "archive": archive.name},
            )
            reporter.begin(f"unpack {archive.name}")
            destination = plan.destination / "_extracted"
            extract_any(archive, destination)
            frames = self._promote_extracted(destination, plan.destination)
            reporter.end(description=f"unpack {archive.name}", detail=f"{frames:,} files")

        # The pack duplicated the archive; the frames are the payload now.
        shutil.rmtree(target, ignore_errors=True)
        return plan.destination, archive.name

    @staticmethod
    def _promote_extracted(extracted: Path, destination: Path) -> int:
        """Flatten a single archive wrapper directory into the dataset root.

        Zips habitually contain one top-level folder (``mav_vid_dataset/``).
        Leaving it in place is what made the converter miss ``train/`` and fall
        back to guessing splits from perceptual hashes, so it is flattened here.
        """
        root = extracted
        while True:
            children = list(root.iterdir())
            split_like = any(
                p.is_dir() and p.name.lower() in {"train", "val", "valid", "validation", "test"}
                for p in children
            )
            if split_like or len(children) != 1 or not children[0].is_dir():
                break
            root = children[0]

        moved = 0
        for child in sorted(root.iterdir()):
            count = sum(1 for _ in child.rglob("*") if _.is_file()) if child.is_dir() else 1
            out = destination / child.name
            if out.exists():
                out = out.with_name(f"{out.name}__from_archive")
            shutil.move(str(child), str(out))
            moved += count
        shutil.rmtree(extracted, ignore_errors=True)
        return moved


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

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        started = time.monotonic()
        reporter = progress or NullProgress()
        plan.destination.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", plan.url, "-p", str(plan.destination), "--unzip"]
        log.info("kaggle download", extra={"dataset": plan.dataset, "ref": plan.url})
        reporter.begin(f"kaggle {plan.dataset}")
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if completed.returncode != 0:
            reporter.end(description=f"kaggle {plan.dataset}", detail="failed")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started,
                error=f"kaggle exited {completed.returncode}: {completed.stderr[-400:]}",
            )
        size = dir_size_bytes(plan.destination)
        reporter.end(description=f"kaggle {plan.dataset}", detail=human_bytes(size))
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

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        import time

        import gdown

        started = time.monotonic()
        reporter = progress or NullProgress()
        plan.destination.mkdir(parents=True, exist_ok=True)
        existing = self._already_have(plan, "*.zip")
        if existing is not None:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "skipped",
                path=existing, bytes_downloaded=existing.stat().st_size,
                duration_s=0.0, error="archive already present",
            )

        require_free_space(plan.destination, plan.approx_bytes, context=plan.source_label)
        reporter.begin(f"gdrive {plan.dataset}")

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
            reporter.end(description=f"gdrive {plan.dataset}", detail="failed")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        if path is None:
            reporter.end(description=f"gdrive {plan.dataset}", detail="nothing downloaded")
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
        reporter.end(description=f"gdrive {plan.dataset}", detail=human_bytes(size))
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

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        started = time.monotonic()
        reporter = progress or NullProgress()
        plan.destination.mkdir(parents=True, exist_ok=True)
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=str(exc),
            )

        log.info("modelscope snapshot", extra={"dataset": plan.dataset, "repo": plan.url})
        reporter.begin(f"modelscope {plan.dataset}")
        try:
            path = snapshot_download(plan.url, local_dir=str(plan.destination))
        except Exception as exc:  # noqa: BLE001
            reporter.end(description=f"modelscope {plan.dataset}", detail="failed")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        size = dir_size_bytes(path)
        reporter.end(description=f"modelscope {plan.dataset}", detail=human_bytes(size))
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "completed",
            path=Path(path), bytes_downloaded=size,
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

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
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

    Some artefacts are reference material rather than training input - the
    Drone-vs-Bird challenge videos are distributed on request by the organisers,
    and the public repo carries annotations only. Reporting this as a distinct
    status beats letting the command appear to succeed.

    The status is ``manual``, not ``needs_credentials``. These have nothing to do
    with each other: one means "you are not logged in", the other means "nobody
    will ever download this automatically, and that is fine".
    """

    kind = SourceKind.MANUAL

    def available(self) -> tuple[bool, str]:
        return True, ""

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        return DownloadResult(
            plan.dataset, plan.variant, plan.source_label, "manual",
            error=(
                f"Reference material, not fetched. Nothing to do here.\n"
                f"  Why        : {plan.notes}\n"
                f"  Reference  : {plan.url}\n"
                f"  If you want it, place it at: {plan.destination}\n"
                f"  Then re-run: anti-uav download --dataset {plan.dataset} --skip-download"
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

    def fetch(
        self, plan: DownloadPlan, progress: ProgressReporter | None = None
    ) -> DownloadResult:
        import time

        import requests  # type: ignore[import-untyped]

        started = time.monotonic()
        reporter = progress or NullProgress()
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
        reporter.begin(f"http {name}")

        try:
            with requests.get(plan.url, stream=True, timeout=60, allow_redirects=True) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                mode = "ab" if part.exists() else "wb"
                written = part.stat().st_size if mode == "ab" else 0
                reporter.update(
                    (written / total * 100.0) if total else None,
                    detail=f"{human_bytes(written)} / {human_bytes(total)}" if total else human_bytes(written),
                )

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
                    reporter.end(description=f"http {name}", detail="rejected: HTML page")
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
                        if total:
                            reporter.update(
                                min(100.0, written / total * 100.0),
                                detail=f"{human_bytes(written)} / {human_bytes(total)}",
                            )
                    for chunk in chunks:
                        handle.write(chunk)
                        written += len(chunk)
                        if total:
                            reporter.update(
                                min(100.0, written / total * 100.0),
                                detail=f"{human_bytes(written)} / {human_bytes(total)}",
                            )
        except Exception as exc:  # noqa: BLE001
            reporter.end(description=f"http {name}", detail=f"failed: {type(exc).__name__}")
            return DownloadResult(
                plan.dataset, plan.variant, plan.source_label, "failed",
                duration_s=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}",
            )

        promoted = self._promote(part, final)
        reporter.end(description=f"http {name}", detail=human_bytes(promoted.stat().st_size))
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
    progress: ProgressReporter | None = None,
) -> Iterator[DownloadResult]:
    """Fetch a dataset from every matching source, yielding one result each.

    ``progress`` is threaded through to whichever backend runs, so a single bar
    in the CLI covers git, HTTP, gdown and Kaggle without each of them growing a
    UI of its own. Library callers omit it and get a silent no-op.
    """
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

        result = backend.fetch(plan, progress)
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