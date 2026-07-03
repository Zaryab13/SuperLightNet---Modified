"""Patient-isolated BraTS datasets for training and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Normalize one MRI volume independently; no population statistics are fitted."""
    volume = volume.astype(np.float32, copy=False)
    mask = volume > 0
    values = volume[mask] if mask.any() else volume
    std = float(values.std())
    output = volume.copy()
    if std >= 1e-8:
        output[mask] = (volume[mask] - float(values.mean())) / (std + 1e-8)
    return output


def remap_brats_labels(segmentation: np.ndarray) -> np.ndarray:
    output = np.zeros(segmentation.shape, dtype=np.int64)
    output[segmentation == 1] = 1
    output[segmentation == 2] = 2
    output[segmentation == 4] = 3
    return output


def extract_patch(array: np.ndarray, roi_size: Tuple[int, int, int],
                  center: Sequence[int]) -> np.ndarray:
    """Extract a spatial patch and zero-pad dimensions smaller than the ROI."""
    spatial_shape = array.shape[-3:]
    starts = [max(0, min(size - roi, int(c) - roi // 2))
              for size, roi, c in zip(spatial_shape, roi_size, center)]
    slices = tuple(slice(start, min(start + roi, size))
                   for start, roi, size in zip(starts, roi_size, spatial_shape))
    patch = array[(...,) + slices]
    padding = [(0, 0)] * (patch.ndim - 3)
    padding.extend((0, roi - actual) for roi, actual in zip(roi_size, patch.shape[-3:]))
    if any(right for _, right in padding):
        patch = np.pad(patch, padding, mode="constant")
    return patch


class _PatientDatasetBase(Dataset):
    def __init__(self, dataset_root: Path, case_ids: Iterable[str]):
        self.dataset_root = Path(dataset_root).resolve()
        self.case_ids = tuple(str(case_id) for case_id in case_ids)
        if not self.case_ids:
            raise ValueError("Patient split is empty")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise AssertionError("Duplicate patient IDs supplied to dataset")
        self.case_paths = tuple(self.dataset_root / case_id for case_id in self.case_ids)
        for case_id, case_path in zip(self.case_ids, self.case_paths):
            if not case_path.is_dir() or case_path.name != case_id:
                raise FileNotFoundError(f"Invalid patient directory: {case_path}")
        self.sample_paths = tuple(
            case_path / f"{case_id}_{suffix}.nii.gz"
            for case_id, case_path in zip(self.case_ids, self.case_paths)
            for suffix in (*MODALITIES, "seg")
        )
        missing = [path for path in self.sample_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing BraTS sample file: {missing[0]}")

    def _load_case(self, index: int):
        case_id = self.case_ids[index]
        case_path = self.case_paths[index]
        images = [
            normalize_nonzero(
                nib.load(str(case_path / f"{case_id}_{modality}.nii.gz")).get_fdata(dtype=np.float32)
            )
            for modality in MODALITIES
        ]
        segmentation = nib.load(str(case_path / f"{case_id}_seg.nii.gz")).get_fdata(dtype=np.float32)
        return np.stack(images), remap_brats_labels(segmentation)


class PatientPatchDataset(_PatientDatasetBase):
    """Generate deterministic-per-epoch patches exclusively from training patients."""

    def __init__(self, dataset_root: Path, case_ids: Iterable[str], roi_size=(160, 160, 160),
                 samples_per_patient: int = 1, tumor_sampling_probability: float = 0.7,
                 seed: int = 42, augment: bool = True):
        super().__init__(dataset_root, case_ids)
        if samples_per_patient < 1:
            raise ValueError("samples_per_patient must be positive")
        self.roi_size = tuple(int(value) for value in roi_size)
        self.samples_per_patient = int(samples_per_patient)
        self.tumor_sampling_probability = float(tumor_sampling_probability)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.case_ids) * self.samples_per_patient

    def __getitem__(self, sample_index: int):
        patient_index = sample_index // self.samples_per_patient
        sample_number = sample_index % self.samples_per_patient
        images, target = self._load_case(patient_index)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + patient_index * 101 + sample_number
        )
        tumor_voxels = np.argwhere(target > 0)
        if tumor_voxels.size and rng.random() < self.tumor_sampling_probability:
            center = tumor_voxels[rng.integers(len(tumor_voxels))]
        else:
            center = [rng.integers(size) for size in target.shape]
        image_patch = extract_patch(images, self.roi_size, center)
        target_patch = extract_patch(target, self.roi_size, center)

        if self.augment:
            for spatial_axis in range(3):
                if rng.random() < 0.5:
                    image_patch = np.flip(image_patch, axis=spatial_axis + 1)
                    target_patch = np.flip(target_patch, axis=spatial_axis)
        return (
            torch.from_numpy(np.ascontiguousarray(image_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(target_patch)).long(),
        )


class PatientVolumeDataset(_PatientDatasetBase):
    """Return one complete MRI and label volume per validation patient."""

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index: int):
        images, target = self._load_case(index)
        return (
            torch.from_numpy(np.ascontiguousarray(images)).float(),
            torch.from_numpy(np.ascontiguousarray(target)).long(),
            self.case_ids[index],
        )
