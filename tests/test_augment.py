"""The two augmentations ultralytics does not provide.

``cutout`` was removed from ultralytics' ``RandomPerspective`` signature, and
there is no IR-grayscale knob at all. Routing them through albumentations would
mean depending on an optional extra that is not installed, so they run from a
project-owned callback instead. These tests exist because a silent no-op is the
failure mode that matters here: a config that says ``cutout: 6`` while the code
does nothing is worse than no config at all.
"""

from __future__ import annotations

import torch

from anti_uav.config.loader import load_recipe
from anti_uav.detection import augment


class _FakeTrainer:
    """Minimal stand-in: ultralytics hangs the preprocessed batch off ``.batch``."""

    def __init__(self, batch: object) -> None:
        self.batch = batch


def _recipe(cutout: int, ir_gray: float):  # type: ignore[no-untyped-def]
    recipe = load_recipe("yolo11n").model_copy(deep=True)
    recipe.augmentation.cutout = cutout
    recipe.augmentation.ir_grayscale_probability = ir_gray
    return recipe


class TestGrayscale:
    def test_collapses_channels_to_luma(self) -> None:
        img = torch.rand(4, 3, 16, 16)
        augment.grayscale_batch(img, 1.0, augment._generator(0))
        assert torch.equal(img[:, 0], img[:, 1])
        assert torch.equal(img[:, 1], img[:, 2])

    def test_probability_zero_is_a_no_op(self) -> None:
        img = torch.rand(4, 3, 16, 16)
        before = img.clone()
        augment.grayscale_batch(img, 0.0, augment._generator(0))
        assert torch.equal(before, img)

    def test_preserves_dtype_and_shape(self) -> None:
        img = torch.rand(2, 3, 8, 8, dtype=torch.float32)
        augment.grayscale_batch(img, 1.0, augment._generator(0))
        assert img.shape == (2, 3, 8, 8)
        assert img.dtype == torch.float32


class TestCutout:
    def test_erases_pixels(self) -> None:
        img = torch.ones(2, 3, 64, 64)
        augment.cutout_batch(img, 6, augment._generator(0))
        assert (img == 0).any()
        # Not the whole image: a fully erased batch carries no signal at all.
        assert (img != 0).any()

    def test_zero_holes_is_a_no_op(self) -> None:
        img = torch.ones(1, 3, 8, 8)
        augment.cutout_batch(img, 0, augment._generator(0))
        assert bool((img == 1).all())

    def test_never_writes_outside_the_tensor(self) -> None:
        """Small images must not produce negative-index slicing."""
        img = torch.ones(1, 3, 4, 4)
        augment.cutout_batch(img, 5, augment._generator(3))
        assert img.shape == (1, 3, 4, 4)


class TestCallback:
    def test_mutates_the_batch_in_place(self) -> None:
        callback = augment.augmentation_callback(_recipe(cutout=6, ir_gray=0.15), seed=0)
        assert callback is not None
        original = torch.rand(6, 3, 32, 32)
        trainer = _FakeTrainer({"img": original.clone()})
        callback(trainer)
        assert not torch.equal(original, trainer.batch["img"])

    def test_returns_none_when_there_is_nothing_to_do(self) -> None:
        assert augment.augmentation_callback(_recipe(0, 0.0), seed=0) is None

    def test_tolerates_a_missing_or_malformed_batch(self) -> None:
        callback = augment.augmentation_callback(_recipe(6, 0.15), seed=0)
        assert callback is not None
        for batch in (
            None,
            {"no_img_key": 1},
            torch.rand(2, 3, 8, 8),
            torch.randint(0, 4, (2, 3, 8, 8)),
            "not a batch",
        ):
            callback(_FakeTrainer(batch))
        callback(object())


class TestReproducibility:
    """A run must reproduce from its seed, which is the whole promise of `seed:`."""

    def test_same_seed_gives_an_identical_batch(self) -> None:
        base = torch.rand(4, 3, 32, 32)
        out = []
        for _ in range(2):
            img = base.clone()
            gen = augment._generator(7)
            augment.grayscale_batch(img, 0.5, gen)
            augment.cutout_batch(img, 3, gen)
            out.append(img)
        assert torch.equal(out[0], out[1])

    def test_different_seeds_diverge(self) -> None:
        base = torch.rand(4, 3, 32, 32)
        a, b = base.clone(), base.clone()
        augment.cutout_batch(a, 3, augment._generator(1))
        augment.cutout_batch(b, 3, augment._generator(2))
        assert not torch.equal(a, b)

    def test_does_not_disturb_the_global_torch_stream(self) -> None:
        """Enabling cutout must not shift every other random decision in the run."""
        base = torch.rand(2, 3, 8, 8)
        torch.manual_seed(1234)
        first = torch.rand(3)
        torch.manual_seed(1234)
        gen = augment._generator(0)
        img = base.clone()
        augment.grayscale_batch(img, 0.5, gen)
        augment.cutout_batch(img, 2, gen)
        second = torch.rand(3)
        assert torch.equal(first, second)
        assert not torch.equal(img, base)


class TestRecipeWiring:
    """The config must actually reach the callback."""

    def test_both_recipes_carry_the_knobs(self) -> None:
        for family in ("yolo11n", "rtdetr_x2"):
            recipe = load_recipe(family)
            assert recipe.augmentation.cutout > 0, family
            assert recipe.augmentation.ir_grayscale_probability > 0.0, family
