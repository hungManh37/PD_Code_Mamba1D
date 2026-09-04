"""Prototypical Network episode head — deliberately the SAME simple head for
both backbones, so the only variable between runs is ResNet12 vs Mamba1D.

logits = -||query_embedding - class_prototype||^2
prototype_c = mean(support_embeddings of class c)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_prototypes(support_embeddings: torch.Tensor, way: int, shot: int) -> torch.Tensor:
    # support_embeddings: (way*shot, D), ordered class-major (class0 x shot, class1 x shot, ...)
    dim = support_embeddings.shape[-1]
    return support_embeddings.reshape(way, shot, dim).mean(dim=1)  # (way, D)


def episode_logits(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    # (Q, D), (way, D) -> (Q, way) negative squared euclidean distance
    dists = torch.cdist(query_embeddings, prototypes, p=2) ** 2
    return -dists


def episode_loss_and_acc(
    backbone: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    way: int,
    shot: int,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    support_embed = backbone(support_x)
    query_embed = backbone(query_x)

    prototypes = compute_prototypes(support_embed, way, shot)
    logits = episode_logits(query_embed, prototypes)

    loss = F.cross_entropy(logits, query_y)
    preds = logits.argmax(dim=-1)
    acc = (preds == query_y).float().mean().item()
    return loss, acc, preds
