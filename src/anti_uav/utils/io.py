"""Filesystem, hashing and serialisation helpers.

Dataset work is long-running and interruptible, so the recurring pattern is
"write to a temp file, then atomically replace". A killed ingest job must never
leave a half-written ``labels/0001.txt`` behind, because the next run would
silently train on truncated ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from .logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_CHUNK = 1 << 20  # 1 MiB


def human_bytes(num: float | int) -> str:
    """``1536`` -> ``'1.5 KB'``. Used in download dry-runs and disk reports."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(size) < 1024.0 or unit == "PB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:,.1f} PB"


def sha256_file(path: str | Path, *, chunk: int = _CHUNK) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Atomically write JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    tmp.replace(target)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, text: str, *, newline: str = "\n") -> Path:
    """Atomically write UTF-8 text with normalised line endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)
    tmp.replace(target)
    return target


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one JSON object per line. Used for per-frame tracking output."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return target


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return
    with p.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed jsonl line", extra={"path": str(p), "line": lineno})


def safe_extract_zip(archive: str | Path, dest: str | Path, *, strip_prefix: str | None = None) -> Path:
    """Extract a zip, refusing path traversal (``../``) and absolute members.

    Public dataset mirrors are not always trustworthy; a zip with ``../../..``
    members would overwrite arbitrary files. This walks the resolved target and
    asserts containment before writing anything.
    """
    archive_path = Path(archive)
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        if strip_prefix:
            members = [m for m in members if m.startswith(strip_prefix)]
        for name in members:
            if not name:
                continue
            target = (destination / name).resolve()
            if not str(target).startswith(str(destination)):
                raise ValueError(f"unsafe zip member escapes destination: {name!r}")
            zf.extract(name, destination)
    log.info("extracted archive", extra={"archive": archive_path.name, "dest": str(destination)})
    return destination


def safe_extract_tar(archive: str | Path, dest: str | Path) -> Path:
    """Extract a tar with the Python 3.12+ ``filter='data'`` guard when present."""
    archive_path = Path(archive)
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    with tarfile.open(archive_path) as tf:
        try:
            tf.extractall(destination, filter="data")  # type: ignore[call-arg]
        except TypeError as exc:  # pragma: no cover - Python 3.11 fallback
            for member in tf.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination)):
                    raise ValueError(
                        f"unsafe tar member escapes destination: {member.name!r}"
                    ) from exc
                tf.extract(member, destination)
    log.info("extracted archive", extra={"archive": archive_path.name, "dest": str(destination)})
    return destination


def extract_any(archive: str | Path, dest: str | Path) -> Path:
    """Dispatch on file extension: zip / tar / tar.gz / tgz / txz / nothing-to-do."""
    p = Path(archive)
    name = p.name.lower()
    if name.endswith(".zip"):
        return safe_extract_zip(p, dest)
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2")):
        return safe_extract_tar(p, dest)
    log.info("no extraction needed", extra={"file": p.name})
    return Path(dest)


def iter_files(root: str | Path, patterns: Iterable[str] = ("*",), *, recursive: bool = True) -> Iterator[Path]:
    """Yield files under ``root`` matching any glob in ``patterns``, sorted.

    Sorted order matters: frame extraction and split assignment must be
    reproducible across machines.
    """
    base = Path(root)
    if not base.exists():
        return
    seen: set[Path] = set()
    for pattern in patterns:
        globber = base.rglob if recursive else base.glob
        for path in sorted(globber(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def count_files(root: str | Path, patterns: Iterable[str] = ("*",)) -> int:
    return sum(1 for _ in iter_files(root, patterns))


def dir_size_bytes(root: str | Path) -> int:
    """Recursive size in bytes. 0 for a missing directory."""
    base = Path(root)
    if not base.exists():
        return 0
    if base.is_file():
        return base.stat().st_size
    return sum(f.stat().st_size for f in base.rglob("*") if f.is_file())


def free_disk_bytes(path: str | Path) -> int:
    """Free bytes on the volume holding ``path``."""
    usage = shutil.disk_usage(Path(path))
    return int(usage.free)


def require_free_space(path: str | Path, needed: int, *, context: str = "") -> None:
    """Abort early rather than filling the disk mid-extract."""
    free = free_disk_bytes(path)
    if free < needed:
        raise OSError(
            f"insufficient disk space{(' for ' + context) if context else ''}: "
            f"need {human_bytes(needed)}, only {human_bytes(free)} free on {Path(path).anchor}"
        )


def unique_path(path: str | Path) -> Path:
    """``out.txt`` -> ``out (1).txt`` when the target exists."""
    p = Path(path)
    if not p.exists():
        return p
    stem, suffix, parent = p.stem, p.suffix, p.parent
    for n in range(1, 1000):
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename next to {p}")


def copy_into_tree(src: str | Path, dest: str | Path, *, overwrite: bool = False) -> int:
    """Copy a file tree, returning the number of files copied.

    Used by the combo builder, which hard-links when possible to avoid
    duplicating 100k JPEGs across 7 combo datasets.
    """
    source = Path(src)
    target = Path(dest)
    copied = 0
    for path in iter_files(source):
        rel = path.relative_to(source)
        out = target / rel
        if out.exists() and not overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        copied += 1
    return copied


def link_into_tree(src: str | Path, dest: str | Path) -> tuple[int, int]:
    """Hard-link a file tree into ``dest``; fall back to copy across volumes.

    Returns ``(linked, copied)``. On Windows a hardlink needs no extra admin
    rights and saves the most space, which matters a lot when the same 100k
    frames appear in seven combo datasets.
    """
    source = Path(src)
    target = Path(dest)
    linked = copied = 0
    for path in iter_files(source):
        rel = path.relative_to(source)
        out = target / rel
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, out)
            linked += 1
        except OSError:
            shutil.copy2(path, out)
            copied += 1
    return linked, copied


def move_dir_contents(src: str | Path, dest: str | Path, *, flatten: bool = False) -> int:
    """Move everything inside ``src`` into ``dest``; returns the file count."""
    source = Path(src)
    target = Path(dest)
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in sorted(source.iterdir()):
        destination = target / child.name if flatten else target / child.stem
        destination = destination / child.name if not flatten else destination
        shutil.move(str(child), str(destination))
        moved += 1 if child.is_file() else sum(1 for _ in child.rglob("*") if _.is_file())
    return moved


def chunks(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Batch an iterable. Guards against loading 100k paths at once."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def first_existing(*candidates: str | Path) -> Path | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    return None
