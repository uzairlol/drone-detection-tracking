"""Augmentations ultralytics does not implement, wired through its callback hook.

Two knobs in ``configs/train/*.yaml`` describe augmentations this project depends
on and ultralytics does not provide:

``cutout``
    Random erasing. Ultralytics used to expose ``cutout`` as a
    ``RandomPerspective`` argument and removed it; the installed version's
    signature is ``(degrees, translate, scale, shear, perspective, size,
    preserve_obb)``. Birds and cloud edges are the false-positive source in this
    data, and erasing patches is the cheapest way to teach the backbone not to
    respond to a small high-contrast blob.

``ir_grayscale_probability``
    Desaturate a fraction of the batch. Combos that mix visible RGB with thermal
    IR (``dvb+mavvid+antiuav`` and ``all4``) hand the backbone a palette cue it
    can key on instead of learning shape: thermal frames have no colour at all.
    Collapsing colour on some visible frames removes the shortcut without
    discarding the RGB data.

Both are applied to ``batch["img"]`` in the ``on_train_batch_start`` callback,
which ultralytics fires on the preprocessed batch before the forward pass. This
is a documented extension point (``model.add_callback``) and works for both the
YOLO and the RT-DETR loader, so it does not fork the two code paths apart.

Why not albumentations: ``Albumentations`` is already in ultralytics' own
``v8_transforms``, but the package is an optional extra and is not installed, so
routing these through ``hyp.augmentations`` would silently no-op on a bare
install. A 60-line tensor op cannot.

Everything is driven by a ``torch.Generator`` seeded from the run seed, so a run
with ``seed: 0`` reproduces exactly as ``run_metadata.json`` claims it will.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..config.schema import TrainRecipe
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Rec. 601 luma coefficients, matching cv2's ``COLOR_BGR2GRAY``.
_LUMA = (0.299, 0.587, 0.114)

#: Fraction of each image's area one cutout hole covers. Ultralytics' removed
#: implementation used 0.5; it is not in any config file, so it is a constant.
_CUTOUT_RATIO = 0.5


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def cutout_batch(img: Tensor, holes: int, gen: torch.Generator, ratio: float = _CUTOUT_RATIO) -> Tensor:
    """Erase ``holes`` random rectangles per image, filling with zeros.

    ``img`` is ``(B, 3, H, W)`` already normalised, so zero is the dataset mean —
    the patch reads as uninformative rather than as a black rectangle, which
    would itself be a cue.

    The tensor is modified in place and returned for chaining.
    """
    if holes <= 0:
        return img
    batch, _channels, height, width = img.shape
    hole_area = max(1, int(height * width * ratio))

    for b in range(batch):
        for _ in range(holes):
            # Rectangles of fixed area but random aspect ratio, as in the original.
            aspect = 0.3 + 0.7 * torch.rand(1, generator=gen).item()
            h = round((hole_area * aspect) ** 0.5)
            w = round((hole_area / max(aspect, 1e-6)) ** 0.5)
            h = max(1, min(h, height))
            w = max(1, min(w, width))
            y = int(torch.randint(0, height - h + 1, (1,), generator=gen).item())
            x = int(torch.randint(0, width - w + 1, (1,), generator=gen).item())
            img[b, :, y : y + h, x : x + w] = 0.0
    return img


def grayscale_batch(img: Tensor, probability: float, gen: torch.Generator) -> Tensor:
    """Collapse a random fraction of the batch to luma, broadcast back to 3 channels.

    Thermal IR frames are genuinely single-channel, so this produces the same
    3-channel-tensor-of-a-grey-image shape the IR converter emits, rather than
    inventing a distribution the IR data does not have.
    """
    if probability <= 0.0:
        return img
    batch = img.shape[0]
    weights = torch.tensor(_LUMA, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    for b in range(batch):
        if float(torch.rand(1, generator=gen).item()) < probability:
            luma = (img[b : b + 1] * weights).sum(dim=1, keepdim=True)
            img[b : b + 1] = luma.expand_as(img[b : b + 1])
    return img


def augmentation_callback(recipe: TrainRecipe, seed: int) -> Callable[[Any], None] | None:
    """Build the ``on_train_batch_start`` callback for a recipe, or ``None``.

    Returns ``None`` when there is nothing to do, so the caller can skip
    registering a no-op callback.
    """
    holes = recipe.augmentation.cutout
    gray_p = recipe.augmentation.ir_grayscale_probability
    if holes <= 0 and gray_p <= 0.0:
        return None

    # A dedicated generator, so these draws neither consume nor depend on
    # ultralytics' own global RNG stream. Otherwise enabling cutout would shift
    # every other random decision and silently change the run's reproducibility.
    gen = _generator(seed)

    def _on_train_batch_start(trainer: Any) -> None:
        # ultralytics passes the trainer as the callback argument and hangs the
        # preprocessed batch off `trainer.batch` as a dict. Accept a bare tensor
        # too so the callback can be exercised directly in a test.
        batch = getattr(trainer, "batch", None)
        if isinstance(batch, Tensor):
            img: Any = batch
        elif isinstance(batch, dict):
            img = batch.get("img")
        else:
            return
        if not isinstance(img, Tensor) or img.ndim != 4:
            return
        # Guard against non-float tensors: the op only ever writes zeros or a
        # weighted sum of the same tensor, but dtype safety is worth one check.
        if not img.is_floating_point():
            return
        if gray_p > 0.0:
            grayscale_batch(img, gray_p, gen)
        if holes > 0:
            cutout_batch(img, holes, gen)

    log.debug(
        "registered tensor augmentations",
        extra={"cutout": holes, "ir_grayscale_probability": gray_p},
    )
    return _on_train_batch_start


def register(
    model: Any,
    recipe: TrainRecipe,
    seed: int,
) -> bool:
    """Attach the callback to a loaded ultralytics model. Returns True if added."""
    callback = augmentation_callback(recipe, seed)
    if callback is None:
        return False
    model.add_callback("on_train_batch_start", callback)
    return True
