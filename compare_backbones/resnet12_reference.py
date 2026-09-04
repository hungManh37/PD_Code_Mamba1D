"""Standard few-shot ResNet12 (image backbone), for side-by-side comparison
with `Mamba1DEncoder`.

NOTE: this is the canonical few-shot-learning ResNet12 (4 residual stages,
DropBlock, 3x3 convs) used across most FSL papers (Lee et al. 2019, etc.).
It is NOT necessarily byte-identical to whatever lives in your private
`tim_2026/model/_reference/encoders/smnet_conv64f_encoder.py` (that file was
not part of the files you uploaded, so I could not diff against it). Treat
this file as the reference point for the architectural comparison below,
and swap it out for your real encoder when you wire things into your repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock(nn.Module):
    """Standard DropBlock2D used in FSL ResNet12 implementations."""

    def __init__(self, block_size: int = 5) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        if not self.training or gamma <= 0.0:
            return x
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()
        mask = F.max_pool2d(
            mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2
        )
        mask = 1 - mask
        keep = mask.sum() / mask.numel()
        return x * mask * (mask.numel() / (mask.sum() + 1e-8)) if keep > 0 else x


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, drop_rate: float, dropblock_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)
        )
        self.pool = nn.MaxPool2d(2)
        self.drop_rate = drop_rate
        self.dropblock = DropBlock(dropblock_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + identity)
        out = self.pool(out)
        if self.drop_rate > 0.0:
            out = self.dropblock(out, gamma=self.drop_rate)
        return out


class ResNet12(nn.Module):
    """4-stage ResNet12, channel width 64-160-320-640 (the FSL-standard config)."""

    def __init__(self, drop_rate: float = 0.0, dropblock_size: int = 5) -> None:
        super().__init__()
        channels = [64, 160, 320, 640]
        in_c = 3
        blocks = []
        for out_c in channels:
            blocks.append(BasicBlock(in_c, out_c, drop_rate, dropblock_size))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)  # (B, 640, H/16, W/16), e.g. 84x84 -> 5x5
