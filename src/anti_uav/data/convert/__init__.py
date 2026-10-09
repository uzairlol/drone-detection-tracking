"""Converter registry and dispatch.

One entry point - :func:`convert_dataset` - that picks the right converter,
detects the on-disk layout when a dataset ships more than one shape, and returns
a uniform :class:`ConvertReport`.
"""

from __future__ import annotations

from pathlib import Path

from ...config.schema import DatasetSpec
from ...utils.logging import get_logger
from ..frameindex import ConvertReport
from .antiuav import AntiUavConverter
from .base import ConversionAborted, Converter
from .dvb import DvbConverter, DvbFlatConverter, detect_layout as detect_dvb_layout
from .mavvid import MavVidConverter
from .mmuav import MmUavConverter

log = get_logger(__name__)

CONVERTERS: dict[str, type[Converter]] = {
    "antiuav": AntiUavConverter,
    "dvb": DvbConverter,
    "mavvid": MavVidConverter,
    "mmuav": MmUavConverter,
}

#: Datasets whose published layout has changed shape across mirrors. The
#: converter for these is chosen from what actually landed on disk.
_LAYOUT_SENSITIVE = {"dvb"}


def converter_class(alias: str, raw_root: Path | None = None) -> type[Converter]:
    """The converter to use for a dataset, given what is on disk."""
    if alias == "dvb" and raw_root is not None and detect_dvb_layout(raw_root) == "flat":
        return DvbFlatConverter
    try:
        return CONVERTERS[alias]
    except KeyError as exc:
        raise KeyError(
            f"no converter registered for {alias!r}; available: {sorted(CONVERTERS)}"
        ) from exc


def convert_dataset(
    spec: DatasetSpec,
    raw_root: Path,
    *,
    variant: str | None = None,
    strict_labels: bool = True,
    copy_images: bool = True,
    **kwargs,
) -> ConvertReport:
    """Run the right converter over ``raw_root``.

    Extra keyword arguments (``modalities``, ``max_sequences`` for MM-UAV) are
    passed through to the converter, which ignores what it does not use.
    """
    cls = converter_class(spec.alias, raw_root)
    converter = cls(
        spec,
        raw_root,
        variant=variant,
        strict_labels=strict_labels,
        copy_images=copy_images,
        **kwargs,
    )
    log.info(
        "converting",
        extra={
            "dataset": spec.alias,
            "converter": cls.__name__,
            "variant": converter.variant,
            "raw_root": str(raw_root),
        },
    )
    return converter.run()


__all__ = [
    "CONVERTERS",
    "AntiUavConverter",
    "ConversionAborted",
    "Converter",
    "DvbConverter",
    "DvbFlatConverter",
    "MavVidConverter",
    "MmUavConverter",
    "convert_dataset",
    "converter_class",
]