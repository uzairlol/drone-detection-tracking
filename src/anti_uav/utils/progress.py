"""Progress reporting for long transfers.

The download backends are library code. They run under pytest, inside the Kaggle
notebook and from the CLI, and only one of those has a terminal worth drawing on.
So a backend never imports a UI itself: it accepts an optional
:class:`ProgressReporter` and pushes updates into it. The CLI hands it the rich
one; every other caller gets :class:`NullProgress` and pays nothing.

The contract is deliberately tiny and percent-based. Git reports object counts,
HTTP reports bytes and gdown reports its own percentage; all three collapse to
"how far along is this" plus "the one line of detail worth printing", which is
exactly what a bar needs. Keeping the contract this narrow is what lets one bar
render all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console


@runtime_checkable
class ProgressReporter(Protocol):
    """What a backend is allowed to assume about a progress sink."""

    def begin(self, description: str) -> None:
        """Start (or restart) the displayed task."""

    def update(
        self,
        percent: float | None,
        *,
        detail: str = "",
        description: str | None = None,
    ) -> None:
        """Report an absolute percentage. ``None`` means "still running"."""

    def end(self, *, description: str | None = None, detail: str = "") -> None:
        """Finish the task, snapping the bar to 100%."""

    def stop(self) -> None:
        """Tear down any live display. Safe to call more than once."""


class NullProgress:
    """The default sink. Every method is a no-op, so backends never branch."""

    __slots__ = ()

    def begin(self, description: str) -> None:
        return None

    def update(
        self,
        percent: float | None,
        *,
        detail: str = "",
        description: str | None = None,
    ) -> None:
        return None

    def end(self, *, description: str | None = None, detail: str = "") -> None:
        return None

    def stop(self) -> None:
        return None


def _clamp(percent: float | None) -> float | None:
    if percent is None:
        return None
    return max(0.0, min(100.0, float(percent)))


class RichProgress:
    """One live bar on a rich ``Console``, reused across sequential transfers.

    A single task, not a stack: downloads run one at a time, and a pile of
    finished bars buries the only one that is actually moving. The bar is not
    transient so the final frame stays on screen after ``end``.
    """

    def __init__(self, console: Console | None = None, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._console = console
        self._progress: object | None = None
        self._task: object | None = None

    def _ensure(self) -> None:
        if not self._enabled or self._progress is not None:
            return
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("{task.fields[detail]}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
            refresh_per_second=8,
        )

    def begin(self, description: str) -> None:
        if not self._enabled:
            return
        self._ensure()
        assert self._progress is not None
        self._progress.start()  # type: ignore[attr-defined]
        if self._task is None:
            self._task = self._progress.add_task(  # type: ignore[attr-defined]
                description, total=100.0, detail=""
            )
        else:
            self._progress.update(  # type: ignore[attr-defined]
                self._task, description=description, completed=0.0, total=100.0, detail=""
            )

    def update(
        self,
        percent: float | None,
        *,
        detail: str = "",
        description: str | None = None,
    ) -> None:
        if not self._enabled or self._task is None:
            return
        assert self._progress is not None
        fields: dict[str, object] = {"detail": detail}
        if description is not None:
            fields["description"] = description
        value = _clamp(percent)
        if value is not None:
            fields["completed"] = value
        self._progress.update(self._task, **fields)  # type: ignore[attr-defined]

    def end(self, *, description: str | None = None, detail: str = "") -> None:
        if not self._enabled:
            return
        if self._task is not None:
            assert self._progress is not None
            fields: dict[str, object] = {"completed": 100.0, "detail": detail}
            if description is not None:
                fields["description"] = description
            self._progress.update(self._task, **fields)  # type: ignore[attr-defined]
        self.stop()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()  # type: ignore[attr-defined]
