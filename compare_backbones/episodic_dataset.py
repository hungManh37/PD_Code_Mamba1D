"""Generic N-way K-shot episodic sampler over a class-folder dataset.

ASSUMED layout (confirm/adjust `_list_samples` below — this was NOT visible
in the files you uploaded, so it is a placeholder, not a verified fact):

    <dataset_path>/<split>/<class_name>/<sample_id>.<ext>

    e.g. dataset_path/train/IR007/0001.mat
         dataset_path/train/IR007/0002.mat
         dataset_path/val/OR014/0001.png
         ...

`split` is one of {"train", "val", "test"}, matching your `--dataset-path`
CLI convention (cli.py) where the whole train/val/test tree lives under one
root. If your real tree differs (e.g. a flat manifest CSV instead of
subfolders, or scalograms/ vs pulses/ as two parallel roots), change
`_list_samples` only — everything else (episode sampling, collation) is
format-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch


Loader = Callable[[Path], np.ndarray]


def load_image_scalogram(path: Path, image_size: int = 84) -> np.ndarray:
    """Loads a CWT-scalogram PNG/JPG as a (3, H, W) float32 array in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # (3, H, W)


def load_raw_signal_mat(path: Path, target_length: int = 2048, mat_key: str | None = None) -> np.ndarray:
    """Loads a raw 1D pulse from .mat/.npy/.csv into a (1, target_length) array.

    ASSUMPTION: for `.mat` files, the signal is stored under a single
    non-metadata variable, or under `mat_key` if you pass one explicitly
    (e.g. mat_key="pulse" or "data"). Adjust if your files differ.
    """
    suffix = path.suffix.lower()
    if suffix == ".mat":
        from scipy.io import loadmat

        contents = loadmat(str(path))
        if mat_key is not None:
            signal = np.asarray(contents[mat_key]).squeeze()
        else:
            candidates = [v for k, v in contents.items() if not k.startswith("__")]
            if not candidates:
                raise ValueError(f"No data variables found in {path}")
            signal = np.asarray(max(candidates, key=lambda v: np.asarray(v).size)).squeeze()
    elif suffix == ".npy":
        signal = np.load(path).squeeze()
    elif suffix == ".csv":
        signal = np.loadtxt(path, delimiter=",").squeeze()
    else:
        raise ValueError(f"Unsupported raw-signal file type: {path}")

    signal = signal.astype(np.float32).reshape(-1)
    if signal.size >= target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - signal.size))

    # per-sample z-normalization (common for vibration/PD pulses)
    std = signal.std()
    if std > 1e-8:
        signal = (signal - signal.mean()) / std
    return signal[None, :]  # (1, L)


class ClassFolderEpisodicDataset:
    """Samples N-way K-shot(+Q-query) episodes from `<root>/<split>/<class>/*`."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        loader: Loader,
        extensions: tuple[str, ...] = (".mat", ".npy", ".csv", ".png", ".jpg"),
        seed: int | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.loader = loader
        self.class_to_files = self._list_samples(extensions)
        self.class_names = sorted(self.class_to_files)
        if len(self.class_names) < 2:
            raise ValueError(
                f"Found {len(self.class_names)} classes under "
                f"{self.dataset_path / split} — check the folder layout."
            )
        self.rng = np.random.default_rng(seed)

    def _list_samples(self, extensions: tuple[str, ...]) -> dict[str, list[Path]]:
        split_dir = self.dataset_path / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split_dir} does not exist. Expected layout: "
                f"<dataset_path>/<split>/<class_name>/<file>. "
                f"Adjust ClassFolderEpisodicDataset._list_samples if yours differs."
            )
        class_to_files: dict[str, list[Path]] = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in extensions)
            if files:
                class_to_files[class_dir.name] = files
        return class_to_files

    def sample_episode(
        self, way: int, shot: int, query: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        chosen_classes = self.rng.choice(self.class_names, size=way, replace=False)

        support_x, support_y, query_x, query_y = [], [], [], []
        for label, class_name in enumerate(chosen_classes):
            files = self.class_to_files[class_name]
            need = shot + query
            if len(files) < need:
                raise ValueError(
                    f"Class {class_name!r} has only {len(files)} samples, "
                    f"need {need} for {shot}-shot/{query}-query episodes."
                )
            picked = self.rng.choice(len(files), size=need, replace=False)
            for i in picked[:shot]:
                support_x.append(self.loader(files[i]))
                support_y.append(label)
            for i in picked[shot:]:
                query_x.append(self.loader(files[i]))
                query_y.append(label)

        return (
            torch.from_numpy(np.stack(support_x)).float(),
            torch.tensor(support_y, dtype=torch.long),
            torch.from_numpy(np.stack(query_x)).float(),
            torch.tensor(query_y, dtype=torch.long),
            list(chosen_classes),
        )
