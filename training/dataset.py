"""
Dataset for MultiFTNet training.

Loads prepared 80x80 BGR patches from
    <data_root>/<patch_info>/<split>/<class>/*.png
and, for each image, generates the Fourier-spectrum target online (as the
original DatasetFolderFT did — no need to precompute it).

Each __getitem__ returns:
    img   : FloatTensor [3, 80, 80], BGR, pixels in [0, 255]
    ft    : FloatTensor [1, ft_h, ft_w]  (Fourier target, normalised to [0,1])
    label : int  (0 = spoof, 1 = live)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def generate_ft(img_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    FFT-magnitude target, matching the original repo's generate_FT():
    grayscale → fft2 → fftshift → log(|.|+1) → min-max to [0,1] → resize.
    `size` is (ft_h, ft_w).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    fmin, fmax = fimg.min(), fimg.max()
    fimg = (fimg - fmin + 1) / (fmax - fmin + 1)
    # cv2.resize takes (w, h)
    return cv2.resize(fimg, (size[1], size[0])).astype(np.float32)


class SpoofFTDataset(Dataset):
    def __init__(self, data_root: str, patch_info: str, split: str,
                 ft_size: tuple[int, int] = (10, 10), augment: bool = False):
        self.root = Path(data_root) / patch_info / split
        self.ft_size = ft_size
        self.augment = augment

        if not self.root.is_dir():
            raise FileNotFoundError(f"Split dir not found: {self.root}")

        # Discover samples: <root>/<class>/<file>
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)   # folder name IS the label (0/1)
            for f in class_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bgr(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self._load_bgr(path)

        # Mild augmentation only. Aggressive color jitter can hurt anti-spoofing
        # because colour/texture IS the signal (docs/08).
        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])   # horizontal flip

        ft = generate_ft(img, self.ft_size)            # [ft_h, ft_w]

        # BGR HxWxC uint8 → CxHxW float32 in [0,255] (NOT /255) — see docs/01.
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        ft_t = torch.from_numpy(ft).unsqueeze(0)       # [1, ft_h, ft_w]
        return img_t, ft_t, label

    # convenience
    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


def build_loader(cfg, split: str, shuffle: bool, augment: bool) -> DataLoader:
    ds = SpoofFTDataset(
        data_root=cfg.data_root,
        patch_info=cfg.patch_info,
        split=split,
        ft_size=cfg.ft_size,
        augment=augment,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
