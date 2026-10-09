"""The ReID backbone used by the appearance gate.

Small on purpose. The appearance gate runs per detection inside a DeepStream
pipeline that is already batch-inferring 13 streams, on hardware where the block
diagram budgets ~3 ms for detection alone. A ReID model large enough to be
sophisticated would eat that budget for a marginal gain on targets that are 15 px
tall to begin with.

So the design goal is "cheap enough to always run, good enough to beat IoU", not
"state of the art re-identification".

Backbones
---------
``color``  a 4-conv stem on the input size, no pretrained weights. Trains in
           minutes on a laptop GPU and gives a working appearance gate. The right
           starting point.
``resnet18`` ImageNet-pretrained trunk with the classifier replaced. Better
           invariances out of the box, and :func:`torchvision_models` fetches the
           weights automatically.
``mobilenet_v3_small`` ImageNet-pretrained and ~4x cheaper than ResNet-18. The
           best speed/accuracy point for a Jetson.

All of them expose :attr:`ReIdNet.features`, the pooled vector the tracker
actually compares. The classification head exists only to *train* the features:
making crops of the same target share a label is a stronger signal than any
contrastive loss we could write here for a few hundred lines of code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from ..utils.logging import get_logger

log = get_logger(__name__)

#: Architectures this module can build.
ARCHITECTURES = ("color", "resnet18", "mobilenet_v3_small", "tiny_cnn")


class ReIdNet(nn.Module):
    """Embedding network with an optional identity-classification head.

    Wraps a backbone so that :meth:`forward` can be called in two modes:

    * ``mode="embed"`` (default) - returns the unit-norm pooled feature. What the
      tracker uses.
    * ``mode="classify"`` - returns identity logits plus the feature. What training
      uses.

    Keeping both in one module means a checkpoint saved after training loads
    directly into the inference path with no graph surgery. Being a real
    ``nn.Module`` (rather than a wrapper holding submodules) is what lets
    ``reid_train.py`` move it to a device, hand it to an optimiser, and get a
    ``state_dict`` out of ``torch.save`` without reimplementing any of that.
    """

    def __init__(
        self,
        *,
        arch: str = "color",
        num_classes: int = 0,
        embedding_dim: int = 512,
        input_size: int = 64,
    ) -> None:
        super().__init__()

        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {arch!r}; available: {ARCHITECTURES}")

        self.arch = arch
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        self.backbone, feature_dim = _build_backbone(arch, embedding_dim)
        self.project = (
            nn.Linear(feature_dim, embedding_dim)
            if feature_dim != embedding_dim
            else nn.Identity()
        )
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

    # -- graph -------------------------------------------------------------- #

    @property
    def features(self) -> nn.Module:
        """The pooled feature path. The tracker reads this."""
        return nn.Sequential(self.backbone, self.project, self.bn)

    def embed(self, tensor: Any) -> Any:
        """Unit-norm embedding for an ``(N, 3, H, W)`` batch."""
        pooled = self.backbone(tensor)
        projected = self.bn(self.project(pooled))
        return torch_functional.normalize(projected, dim=1)

    def forward(self, tensor: Any, *, mode: str = "embed") -> Any:
        if mode == "embed":
            return self.embed(tensor)

        if mode != "classify":
            raise ValueError(f"mode must be 'embed' or 'classify', got {mode!r}")

        pooled = self.backbone(tensor)
        projected = self.bn(self.project(pooled))
        if self.classifier is None:
            raise RuntimeError(
                "this ReIdNet was built without a classification head; it cannot be trained"
            )
        return self.classifier(projected), projected

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_backbone(arch: str, embedding_dim: int) -> tuple[nn.Module, int]:
    """Return ``(backbone, feature_dim)`` where backbone maps NCHW -> (N, feature_dim)."""
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the ImageNet head. GAP + Linear(512) is the standard pooling.
        model.fc = nn.Linear(512, embedding_dim)
        return model, embedding_dim

    if arch == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        model.classifier = nn.Linear(model.classifier[0].in_features, embedding_dim)
        return model, embedding_dim

    if arch == "tiny_cnn":
        return _tiny_cnn(embedding_dim, width=64)

    # "color": a plain conv stack. Trains in minutes, no download, and good enough
    # to make the appearance gate meaningful.
    return _tiny_cnn(embedding_dim, width=32)


def _tiny_cnn(embedding_dim: int, *, width: int) -> tuple[nn.Module, int]:
    """A compact CNN: four stride-2 blocks then global average pooling."""

    def block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    layers: list[nn.Module] = [block(3, width), block(width, width * 2), block(width * 2, width * 4)]
    if width >= 64:
        layers.append(block(width * 4, width * 8))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten()), width * (8 if width >= 64 else 4)


def count_identities_from_index(dataset: str, variant: str, modality: str = "rgb") -> int:
    """How many distinct identities exist, for sizing the classification head.

    Anti-UAV is single-target, so this returns ~1 and a classification head is
    useless - appearance learning needs multi-identity data, which is exactly why
    MM-UAV (multi-object MOT with identity preserved) is the right training source
    even though its targets are the smallest in the project.
    """
    from ..data.frameindex import load_index

    records = [
        r for r in load_index(dataset, variant)
        if r.modality == modality and r.box_count > 0
    ]
    return len({r.sequence_id for r in records})


def batch_from_crops(
    crops: list[np.ndarray], *, input_size: int = 64
) -> Any:
    """Stack crops into a normalised NCHW batch."""

    from .appearance import _to_tensor

    if not crops:
        return torch.zeros((0, 3, input_size, input_size))
    return torch.cat([_to_tensor(crop, input_size) for crop in crops], dim=0)


def summarise_model(model: ReIdNet) -> str:
    params = model.parameter_count()
    return (
        f"arch={model.arch} embedding_dim={model.embedding_dim} "
        f"classes={model.num_classes or '-'} params={params:,} "
        f"({params / 1e6:.2f} M)"
    )
