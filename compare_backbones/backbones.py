"""Common `.encode(x) -> (B, D)` interface over both backbones, so the same
episodic ProtoNet training loop can drive either one.

ASSUMPTION (flagged, not guessed silently): your real PECT/HROTFSL head does
local optimal-transport matching over per-patch/per-cell TOKENS, not a single
pooled vector. For a first fair A/B on raw-vs-image input, this file uses a
plain global-pooled embedding + Prototypical Network head (see protonet.py).
That isolates "does the backbone's representation help" from "does the OT
matching head help" -- it is intentionally the SIMPLEST classifier that is
identical for both backbones. Swap in your real UOT head later once you've
confirmed the backbone itself is worth it (see the experiment matrix in the
final message).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mamba1d_backbone import Mamba1DConfig, Mamba1DEncoder
from resnet12_reference import ResNet12


class Backbone(nn.Module):
    """Uniform wrapper: forward(x) -> (B, out_dim) pooled embedding."""

    out_dim: int
    input_kind: str  # "image" | "raw1d", used by the dataset loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class ResNet12Backbone(Backbone):
    def __init__(self, drop_rate: float = 0.0, dropblock_size: int = 5) -> None:
        super().__init__()
        self.net = ResNet12(drop_rate=drop_rate, dropblock_size=dropblock_size)
        self.out_dim = self.net.out_channels
        self.input_kind = "image"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.net(x)                # (B, C, H, W)
        return feat_map.mean(dim=(2, 3))       # global average pool -> (B, C)


class Mamba1DBackboneWrapper(Backbone):
    def __init__(self, config: Mamba1DConfig | None = None) -> None:
        super().__init__()
        self.net = Mamba1DEncoder(config or Mamba1DConfig())
        self.out_dim = self.net.out_channels
        self.input_kind = "raw1d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # already pooled (B, dim)


@dataclass(frozen=True)
class BackboneChoice:
    name: str
    input_kind: str


BACKBONE_REGISTRY = {
    "resnet12": BackboneChoice("resnet12", "image"),
    "mamba1d": BackboneChoice("mamba1d", "raw1d"),
}


def build_backbone(name: str, **kwargs) -> Backbone:
    if name == "resnet12":
        return ResNet12Backbone(**kwargs)
    if name == "mamba1d":
        cfg_keys = {"patch_size", "dim", "expand_dim", "state_dim", "depth", "use_kan", "in_channels", "pool"}
        cfg = Mamba1DConfig(**{k: v for k, v in kwargs.items() if k in cfg_keys})
        return Mamba1DBackboneWrapper(cfg)
    raise ValueError(f"Unknown backbone: {name!r}. Choices: {list(BACKBONE_REGISTRY)}")
