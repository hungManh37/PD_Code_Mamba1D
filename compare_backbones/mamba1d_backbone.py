"""Mamba1D backbone: a bidirectional Mamba (SSM) encoder for RAW 1D signals.

Motivation
----------
Your current few-shot pipeline (`BaseConv64FewShotModel` in `fewshot_common.py`)
always builds an *image* backbone via `build_resnet12_family_encoder(...)` and
feeds it 3x84x84 CWT-scalogram PNGs. ResNet12 therefore never sees the raw
pulse; it sees a 2D time-frequency picture of it.

VibrMamba (Yi et al., Measurement 2025) instead classifies directly from the
1D vibration trace: patchify -> position embedding -> bidirectional SSM
(Mamba) blocks -> classification head. No CWT step is needed.

This module gives you a standalone, framework-agnostic building block that
does the same thing for your PD pulses:

    raw pulse (B, L) or (B, C, L)
        -> patch + linear/KAN projection + position embedding
        -> depth x VibrMambaBlock (bidirectional selective SSM, KAN in/out)
        -> pooled embedding (B, D)   [also exposes token sequence (B, T, D)]

It is deliberately dependency-light (pure PyTorch, no custom CUDA scan) so it
runs anywhere, at the cost of being slower than the official `mamba-ssm`
kernel. Swap `selective_scan` below for `mamba_ssm.ops.selective_scan_fn` if
that package is installed in your environment and you want the fast kernel.

Nothing here overwrites your `_reference/ours.py` or `hrot_fsl.py`. It is a
new, independent encoder you can wire in wherever you currently choose a
backbone (see the bottom of this file / the accompanying README section for
the two integration options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. KAN layer (Kolmogorov-Arnold linear), replacing the paper's MLP/Linear
# --------------------------------------------------------------------------- #
class KANLinear(nn.Module):
    """Efficient KAN layer: y = w_basic * SiLU(x) @ W_b + w_spline * spline(x) @ W_s.

    This follows the "efficient-kan" formulation used in most KAN
    reimplementations (Liu et al. 2024, Eq. 12 in the VibrMamba paper):
    each edge activation phi(x) = w_basic*b(x) + w_spline*sum_i c_i B_i(x),
    with b(x) = SiLU(x) as the paper specifies, and B_i the B-spline basis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_weight_init: float = 1.0,
        spline_weight_init: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)  # (in_features, grid_size + 2*order + 1)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_scale = nn.Parameter(torch.tensor(float(base_weight_init)))
        self.spline_scale = nn.Parameter(torch.tensor(float(spline_weight_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., in_features, grid_size + spline_order)
        grid = self.grid  # (in_features, G)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases  # (..., in_features, grid_size + spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_out = self.base_scale * F.linear(F.silu(x), self.base_weight)
        spline_bases = self.b_splines(x)  # (N, in, G)
        spline_out = self.spline_scale * torch.einsum(
            "nig,oig->no", spline_bases, self.spline_weight
        )
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


# --------------------------------------------------------------------------- #
# 2. Selective scan (pure PyTorch reference; swap for mamba_ssm kernel if
#    you have it installed and want speed).
# --------------------------------------------------------------------------- #
def selective_scan(delta_a: torch.Tensor, delta_bx: torch.Tensor) -> torch.Tensor:
    """Sequential recurrence h_t = deltaA_t * h_{t-1} + deltaBx_t.

    delta_a, delta_bx: (B, L, E, N) -> returns hidden states h: (B, L, E, N)
    """
    b, length, e, n = delta_a.shape
    h = delta_a.new_zeros(b, e, n)
    out = []
    for t in range(length):
        h = delta_a[:, t] * h + delta_bx[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


class SSMDirection(nn.Module):
    """One directional selective SSM, Table 1 lines 7-19 of the VibrMamba paper."""

    def __init__(self, expand_dim: int, state_dim: int) -> None:
        super().__init__()
        self.expand_dim = expand_dim
        self.state_dim = state_dim

        self.in_proj_b = nn.Linear(expand_dim, state_dim, bias=False)
        self.in_proj_c = nn.Linear(expand_dim, state_dim, bias=False)
        self.in_proj_delta = nn.Linear(expand_dim, expand_dim, bias=True)

        # HiPPO-inspired negative log init for A (per-channel, per-state), so
        # that A = -exp(A_log) is initialized as a stable decaying matrix.
        a_init = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(expand_dim, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(expand_dim))

        self.conv = nn.Conv1d(
            expand_dim, expand_dim, kernel_size=3, padding=1, groups=expand_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, E)
        b, length, e = x.shape
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        B_t = self.in_proj_b(x_conv)  # (B, L, N)
        C_t = self.in_proj_c(x_conv)  # (B, L, N)
        delta = F.softplus(self.in_proj_delta(x_conv))  # (B, L, E)

        A = -torch.exp(self.A_log)  # (E, N)
        delta_a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,E,N)
        delta_bx = (
            delta.unsqueeze(-1) * B_t.unsqueeze(2) * x_conv.unsqueeze(-1)
        )  # (B,L,E,N)

        h = selective_scan(delta_a, delta_bx)  # (B, L, E, N)
        y = torch.einsum("blen,bln->ble", h, C_t) + self.D * x_conv
        return y  # (B, L, E)


class VibrMambaBlock(nn.Module):
    """Bidirectional SSM block with KAN in/out projections (paper Table 1 / Fig. 4)."""

    def __init__(self, dim: int, expand_dim: int, state_dim: int, use_kan: bool = True) -> None:
        super().__init__()
        proj = KANLinear if use_kan else nn.Linear
        self.norm = nn.LayerNorm(dim)
        self.in_proj_x = proj(dim, expand_dim)
        self.in_proj_z = proj(dim, expand_dim)
        self.forward_ssm = SSMDirection(expand_dim, state_dim)
        self.backward_ssm = SSMDirection(expand_dim, state_dim)
        self.out_proj = proj(expand_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x_n = self.norm(x)
        x_in = self.in_proj_x(x_n)
        z = self.in_proj_z(x_n)

        y_fwd = self.forward_ssm(x_in)
        y_bwd = torch.flip(self.backward_ssm(torch.flip(x_in, dims=[1])), dims=[1])

        gate = F.silu(z)
        y = y_fwd * gate + y_bwd * gate
        return self.out_proj(y) + residual


# --------------------------------------------------------------------------- #
# 3. Full encoder: raw 1D signal -> patch/pos-embed -> depth x VibrMambaBlock
# --------------------------------------------------------------------------- #
@dataclass
class Mamba1DConfig:
    patch_size: int = 16          # ps in the paper
    dim: int = 192                # token embedding dim
    expand_dim: int = 384         # inner SSM width (dim_inner)
    state_dim: int = 16           # SSM state size N
    depth: int = 6                # number of VibrMambaBlocks
    use_kan: bool = True          # KAN vs. plain Linear projections
    in_channels: int = 1          # number of raw signal channels (>=1)
    pool: str = "mean"            # "mean" | "cls"


class Mamba1DEncoder(nn.Module):
    """Standalone encoder. Plays the same role ResNet12 plays for images,
    but consumes the RAW 1D signal instead of a CWT scalogram.

    Input:  x of shape (B, L) or (B, C_in, L)
    Output: pooled embedding (B, dim); also exposes `forward_tokens` for the
            full (B, T, dim) sequence if you want to feed a few-shot local
            matching head (e.g. PECT/UOT) with per-patch tokens instead of a
            single pooled vector.
    """

    def __init__(self, config: Mamba1DConfig) -> None:
        super().__init__()
        self.config = config
        self.out_channels = config.dim  # keeps naming parity with the ResNet12 encoder

        patch_dim = config.in_channels * config.patch_size
        proj = KANLinear if config.use_kan else nn.Linear
        self.patch_embed = proj(patch_dim, config.dim)

        # generous max length; position embedding is sliced to the actual
        # number of patches at forward time (works for variable pulse length)
        self.max_patches = 4096
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, config.dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if config.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                VibrMambaBlock(config.dim, config.expand_dim, config.state_dim, config.use_kan)
                for _ in range(config.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(config.dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, T, C*patch_size), T = L // patch_size
        b, c, length = x.shape
        ps = self.config.patch_size
        usable = (length // ps) * ps
        if usable != length:
            x = x[..., :usable]
        x = x.reshape(b, c, usable // ps, ps)          # (B, C, T, ps)
        x = x.permute(0, 2, 1, 3).reshape(b, usable // ps, c * ps)  # (B, T, C*ps)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
        if x.dim() != 3:
            raise ValueError(f"Expected (B, L) or (B, C, L), got {tuple(x.shape)}")

        patches = self._patchify(x)          # (B, T, C*ps)
        tokens = self.patch_embed(patches)   # (B, T, dim)

        if self.config.pool == "cls":
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([tokens, cls], dim=1)

        t = tokens.shape[1]
        if t > self.max_patches:
            raise ValueError(
                f"Sequence has {t} patches > max_patches={self.max_patches}; "
                "raise Mamba1DEncoder.max_patches or increase patch_size."
            )
        tokens = tokens + self.pos_embed[:, :t, :]

        for block in self.blocks:
            tokens = block(tokens)
        return self.norm_out(tokens)  # (B, T, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        if self.config.pool == "cls":
            return tokens[:, -1, :]
        return tokens.mean(dim=1)


def build_mamba1d_encoder(**overrides) -> Mamba1DEncoder:
    return Mamba1DEncoder(Mamba1DConfig(**overrides))


# --------------------------------------------------------------------------- #
# 4. Minimal classifier wrapper, mirroring how your project turns a backbone
#    into a trainable model for a plain (non few-shot) comparison run.
# --------------------------------------------------------------------------- #
class Mamba1DClassifier(nn.Module):
    def __init__(self, num_classes: int, config: Mamba1DConfig | None = None) -> None:
        super().__init__()
        self.encoder = Mamba1DEncoder(config or Mamba1DConfig())
        self.head = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.head(feat)
