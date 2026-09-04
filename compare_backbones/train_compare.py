"""Train + evaluate ResNet12(image) or Mamba1D(raw-1D) under the SAME
episodic protocol, for a controlled A/B comparison.

Everything except the backbone + input representation is held fixed:
way-num, shot-num, query-num, episodes/epoch, optimizer, LR schedule, seeds.
This mirrors the "Prefer option B" fairness rule: use one shared protocol so
any accuracy delta is attributable to the backbone, not the training recipe.

Usage
-----
    # Baseline: ResNet12 on CWT scalogram images
    python train_compare.py --dataset-path /path/to/scalograms_root \\
        --backbone resnet12 --shot 1 --epochs 100 --run-name baseline_resnet12

    # New: Mamba1D on raw pulses (bypasses the CWT step entirely)
    python train_compare.py --dataset-path /path/to/pulses_root \\
        --backbone mamba1d --shot 1 --epochs 100 --run-name mamba1d_raw

Both commands must point at dataset roots with the SAME underlying pulses
and the SAME train/val/test split (just represented differently: images vs.
raw arrays) for the comparison to be meaningful. If you only have one
folder of raw pulses today, generate the scalogram folder from the exact
same split first — do not re-split randomly for each backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from backbones import build_backbone
from episodic_dataset import ClassFolderEpisodicDataset, load_image_scalogram, load_raw_signal_mat
from protonet import episode_loss_and_acc

# Reuse your existing logging/visualization utilities if the `tim_2026`
# package is importable in this environment; otherwise fall back to small
# local equivalents so this script still runs standalone.
try:
    from tim_2026.logging import append_summary, write_key_values
    from tim_2026.visualization import save_confusion_matrix, save_tsne
    HAVE_TIM_2026 = True
except ImportError:
    HAVE_TIM_2026 = False

    def write_key_values(path: Path, values: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}: {value}\n")

    def append_summary(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save_confusion_matrix(*args, **kwargs) -> None:
        print("  [skip] tim_2026.visualization not importable; no confusion matrix saved.")

    def save_tsne(*args, **kwargs) -> None:
        print("  [skip] tim_2026.visualization not importable; no t-SNE saved.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at_epoch(epoch: int, args: argparse.Namespace) -> float:
    if epoch < args.warmup_epochs:
        frac = (epoch + 1) / max(1, args.warmup_epochs)
        start = args.warmup_start_factor
        return args.lr * (start + (1 - start) * frac)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def make_loader(args: argparse.Namespace):
    if args.backbone == "resnet12":
        return lambda p: load_image_scalogram(p, image_size=args.image_size)
    return lambda p: load_raw_signal_mat(p, target_length=args.raw_length, mat_key=args.mat_key)


def run_episodes(
    backbone: torch.nn.Module,
    dataset: ClassFolderEpisodicDataset,
    n_episodes: int,
    way: int,
    shot: int,
    query: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    collect_predictions: bool = False,
):
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels, all_class_names = [], [], []

    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y, class_names = dataset.sample_episode(way, shot, query)
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        if optimizer is not None:
            backbone.train()
            optimizer.zero_grad()
            loss, acc, preds = episode_loss_and_acc(backbone, support_x, support_y, query_x, query_y, way, shot)
            loss.backward()
            optimizer.step()
        else:
            backbone.eval()
            with torch.no_grad():
                loss, acc, preds = episode_loss_and_acc(backbone, support_x, support_y, query_x, query_y, way, shot)

        total_loss += loss.item()
        total_acc += acc
        if collect_predictions:
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(query_y.cpu().tolist())
            all_class_names.append(class_names)

    return total_loss / n_episodes, total_acc / n_episodes, all_preds, all_labels, all_class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--backbone", choices=("resnet12", "mamba1d"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_compare"))
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--way", type=int, default=4)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=130)
    parser.add_argument("--val-episodes", type=int, default=150)
    parser.add_argument("--test-episodes", type=int, default=150)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-test-seed", type=int, default=200042)
    parser.add_argument("--gpu", type=int, default=0)

    # ResNet12 (image) specific
    parser.add_argument("--image-size", type=int, default=84)
    # Mamba1D (raw) specific
    parser.add_argument("--raw-length", type=int, default=2048)
    parser.add_argument("--mat-key", default=None)
    parser.add_argument("--mamba-dim", type=int, default=192)
    parser.add_argument("--mamba-expand-dim", type=int, default=384)
    parser.add_argument("--mamba-state-dim", type=int, default=16)
    parser.add_argument("--mamba-depth", type=int, default=6)
    parser.add_argument("--mamba-patch-size", type=int, default=16)

    args = parser.parse_args()

    run_name = args.run_name or f"{args.backbone}_{args.shot}shot_seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "args.json").open("w") as f:
        json.dump(vars(args) | {"pop_dataset_path": str(args.dataset_path)}, f, indent=2, default=str)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    loader = make_loader(args)
    train_set = ClassFolderEpisodicDataset(args.dataset_path, "train", loader, seed=args.seed)
    val_set = ClassFolderEpisodicDataset(args.dataset_path, "val", loader, seed=args.seed + 1)

    if args.backbone == "resnet12":
        backbone = build_backbone("resnet12").to(device)
    else:
        backbone = build_backbone(
            "mamba1d",
            in_channels=1,
            patch_size=args.mamba_patch_size,
            dim=args.mamba_dim,
            expand_dim=args.mamba_expand_dim,
            state_dim=args.mamba_state_dim,
            depth=args.mamba_depth,
        ).to(device)

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[{run_name}] backbone={args.backbone} params={sum(p.numel() for p in backbone.parameters()):,}")
    best_val_acc = -1.0
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group["lr"] = lr_at_epoch(epoch, args)

        train_loss, train_acc, *_ = run_episodes(
            backbone, train_set, args.train_episodes, args.way, args.shot, args.query, device, optimizer
        )
        val_loss, val_acc, *_ = run_episodes(
            backbone, val_set, args.val_episodes, args.way, args.shot, args.query, device, optimizer=None
        )

        append_summary(
            run_dir / "epoch_log.csv",
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            },
        )
        print(
            f"  epoch {epoch+1:3d}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(backbone.state_dict(), run_dir / "best.pt")

    # ---- final test, fixed seed for reproducibility across backbones ----
    backbone.load_state_dict(torch.load(run_dir / "best.pt"))
    test_set = ClassFolderEpisodicDataset(args.dataset_path, "test", loader, seed=args.final_test_seed)
    test_loss, test_acc, preds, labels, episode_classes = run_episodes(
        backbone, test_set, args.test_episodes, args.way, args.shot, args.query, device,
        optimizer=None, collect_predictions=True,
    )

    write_key_values(
        run_dir / "test_result.txt",
        {"backbone": args.backbone, "test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val_acc},
    )
    print(f"[{run_name}] TEST acc={test_acc:.4f} (best val acc={best_val_acc:.4f})")

    # confusion matrix is only meaningful if every episode used the SAME
    # class set, i.e. num_dataset_classes == way
    if len(test_set.class_names) == args.way:
        save_confusion_matrix(
            np.array(labels), np.array(preds), test_set.class_names, run_dir / "confusion_matrix"
        )
    else:
        print("  [skip] confusion matrix: dataset has more classes than `way`, "
              "episodes sample different class subsets each time.")


if __name__ == "__main__":
    main()
