"""Manifest-backed BraTS datasets.

Patient membership is resolved before any volume, crop, or patch is generated.
"""

import os
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Cfg
from split_utils import load_split_manifest, resolve_case_paths


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def remap_seg(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def zscore_norm(img: np.ndarray) -> np.ndarray:
    mask = img > 0
    values = img[mask] if mask.any() else img
    std = values.std()
    if std < 1e-8:
        return img
    out = img.copy()
    out[mask] = (img[mask] - values.mean()) / (std + 1e-8)
    return out


class BraTSDataset(Dataset):
    """Load full volumes belonging exclusively to one manifest split."""

    def __init__(self, split: str, eval_mode: bool = False, manifest_path: Optional[str] = None,
                 fold: Optional[int] = None, root: Optional[str] = None, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.split = split
        self.eval_mode = eval_mode
        self.transform = transform
        self.root = root or Cfg.data_root
        manifest_path = manifest_path or Cfg.split_manifest
        fold = Cfg.split_fold if fold is None else fold
        splits = load_split_manifest(manifest_path, fold)
        self.case_ids = list(splits[split])
        self.cases = resolve_case_paths(self.root, self.case_ids)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        case_id = self.case_ids[idx]
        images = []
        for modality in Cfg.modalities:
            image = zscore_norm(load_nifti(os.path.join(case_dir, f"{case_id}_{modality}.nii.gz")))
            images.append(image.astype(np.float32))
        image = torch.from_numpy(np.stack(images, axis=0)).float()
        if self.eval_mode:
            return image, case_id
        segmentation = remap_seg(load_nifti(os.path.join(case_dir, f"{case_id}_seg.nii.gz")))
        target = torch.from_numpy(segmentation).long()
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def get_case_path(self, idx, modality="t1"):
        case_id = self.case_ids[idx]
        return os.path.join(self.cases[idx], f"{case_id}_{modality}.nii.gz")
