"""Compare ResNet12 vs Mamba1D on characteristics OTHER than accuracy:
parameter count, on-disk model size, inference latency/throughput, peak GPU
memory, and convergence speed (epochs to reach best val acc).

Reuses each run's saved `args.json` (written by train_compare.py) to rebuild
the exact backbone config, and `best.pt` for the trained weights. Does NOT
require re-running training or touching the dataset.

Usage
-----
    python benchmark_extra.py \\
        --run-dir artifacts_compare/baseline_resnet12 \\
        --run-dir artifacts_compare/mamba1d_raw \\
        --output-dir artifacts_compare/_benchmark_extra

If you only ran Mamba1D, pass a single --run-dir; the script still reports
its standalone characteristics (just no side-by-side table).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from backbones import build_backbone


def load_backbone_from_run(run_dir: Path, device: torch.device):
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"{args_path} not found — was this run produced by train_compare.py?")
    with args_path.open() as f:
        run_args = json.load(f)

    backbone_name = run_args["backbone"]
    if backbone_name == "resnet12":
        backbone = build_backbone("resnet12")
        dummy_input = torch.randn(1, 3, run_args.get("image_size", 84), run_args.get("image_size", 84))
    elif backbone_name == "mamba1d":
        backbone = build_backbone(
            "mamba1d",
            in_channels=1,
            patch_size=run_args.get("mamba_patch_size", 16),
            dim=run_args.get("mamba_dim", 192),
            expand_dim=run_args.get("mamba_expand_dim", 384),
            state_dim=run_args.get("mamba_state_dim", 16),
            depth=run_args.get("mamba_depth", 6),
        )
        dummy_input = torch.randn(1, 1, run_args.get("raw_length", 2048))
    else:
        raise ValueError(f"Unknown backbone in {args_path}: {backbone_name!r}")

    ckpt_path = run_dir / "best.pt"
    if ckpt_path.exists():
        backbone.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        print(f"  [warn] {ckpt_path} not found; using randomly-initialized weights "
              f"(params/latency numbers are still valid, but this is not the trained model).")

    backbone = backbone.to(device).eval()
    return backbone_name, backbone, dummy_input.to(device), run_args


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


@torch.no_grad()
def benchmark_inference(
    backbone: torch.nn.Module,
    dummy_input: torch.Tensor,
    device: torch.device,
    batch_size: int,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> dict:
    x = dummy_input.repeat(batch_size, *([1] * (dummy_input.dim() - 1)))

    for _ in range(n_warmup):
        backbone(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(n_iters):
        backbone(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms_per_batch = (elapsed / n_iters) * 1000
    throughput_samples_per_sec = (n_iters * batch_size) / elapsed
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
    )
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": round(latency_ms_per_batch, 4),
        "latency_ms_per_sample": round(latency_ms_per_batch / batch_size, 4),
        "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
        "peak_gpu_mem_mb": round(peak_mem_mb, 2) if peak_mem_mb is not None else None,
    }


def convergence_stats(run_dir: Path) -> dict:
    log_path = run_dir / "epoch_log.csv"
    if not log_path.exists():
        return {"epochs_logged": 0, "best_epoch": None, "best_val_acc": None}

    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"epochs_logged": 0, "best_epoch": None, "best_val_acc": None}

    best_row = max(rows, key=lambda r: float(r["val_acc"]))
    return {
        "epochs_logged": len(rows),
        "best_epoch": int(best_row["epoch"]) + 1,  # 1-indexed for readability
        "best_val_acc": round(float(best_row["val_acc"]), 4),
        "final_train_acc": round(float(rows[-1]["train_acc"]), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                         help="Path to a train_compare.py run dir (repeatable). "
                              "e.g. --run-dir artifacts_compare/baseline_resnet12 --run-dir artifacts_compare/mamba1d_raw")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_compare/_benchmark_extra"))
    parser.add_argument("--batch-size", type=int, default=15,
                         help="Batch size for inference timing (default 15 = way4*(shot1+query...)-ish episode-sized batch).")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for run_dir in args.run_dir:
        print(f"\n=== {run_dir} ===")
        backbone_name, backbone, dummy_input, run_args = load_backbone_from_run(run_dir, device)

        total_params, trainable_params = count_params(backbone)
        size_mb = model_size_mb(run_dir / "best.pt")
        inf_stats = benchmark_inference(
            backbone, dummy_input, device, args.batch_size, args.n_warmup, args.n_iters
        )
        conv_stats = convergence_stats(run_dir)

        row = {
            "run_dir": str(run_dir),
            "backbone": backbone_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(size_mb, 3) if size_mb is not None else None,
            **inf_stats,
            **conv_stats,
        }
        results.append(row)

        print(f"  Params           : {total_params:,} total ({trainable_params:,} trainable)")
        print(f"  Model size       : {row['model_size_mb']} MB")
        print(f"  Latency          : {inf_stats['latency_ms_per_sample']} ms/sample "
              f"(batch={inf_stats['batch_size']})")
        print(f"  Throughput       : {inf_stats['throughput_samples_per_sec']} samples/sec")
        if inf_stats["peak_gpu_mem_mb"] is not None:
            print(f"  Peak GPU memory  : {inf_stats['peak_gpu_mem_mb']} MB")
        print(f"  Convergence      : best val_acc={conv_stats['best_val_acc']} "
              f"at epoch {conv_stats['best_epoch']}/{conv_stats['epochs_logged']}")

    # write CSV summary
    csv_path = args.output_dir / "extra_characteristics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved comparison table: {csv_path}")

    if len(results) >= 2:
        print("\n" + "=" * 78)
        print(f"{'Metric':<28}" + "".join(f"{r['backbone']:>22}" for r in results))
        print("-" * 78)

        def fmt(v):
            return "-" if v is None else v

        rows_to_show = [
            ("Total params", "total_params"),
            ("Trainable params", "trainable_params"),
            ("Model size (MB)", "model_size_mb"),
            ("Latency (ms/sample)", "latency_ms_per_sample"),
            ("Throughput (samples/s)", "throughput_samples_per_sec"),
            ("Peak GPU mem (MB)", "peak_gpu_mem_mb"),
            ("Best val acc", "best_val_acc"),
            ("Best epoch", "best_epoch"),
        ]
        for label, key in rows_to_show:
            print(f"{label:<28}" + "".join(f"{fmt(r[key]):>22}" for r in results))


if __name__ == "__main__":
    main()
