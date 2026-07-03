"""Training and full-volume validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def window_starts(length: int, roi: int, overlap: float = 0.5):
    if length <= roi:
        return [0]
    stride = max(1, int(roi * (1.0 - overlap)))
    starts = list(range(0, length - roi + 1, stride))
    if starts[-1] != length - roi:
        starts.append(length - roi)
    return starts


def region_dice(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    regions = {
        "wt": (prediction > 0, target > 0),
        "tc": ((prediction == 1) | (prediction == 3), (target == 1) | (target == 3)),
        "et": (prediction == 3, target == 3),
    }
    scores = {}
    for name, (pred_mask, target_mask) in regions.items():
        denominator = int(pred_mask.sum()) + int(target_mask.sum())
        scores[name] = (1.0 if denominator == 0 else
                        float(2 * np.logical_and(pred_mask, target_mask).sum() / denominator))
    return scores


@torch.inference_mode()
def validate_volume(model, volume: torch.Tensor, target: torch.Tensor, criterion,
                    device: torch.device, roi_size: Tuple[int, int, int], overlap: float = 0.5):
    """Sliding-window validation returning complete-volume prediction and mean patch loss."""
    spatial_shape = tuple(volume.shape[-3:])
    padded_shape = tuple(max(size, roi) for size, roi in zip(spatial_shape, roi_size))
    padded_volume = torch.zeros((volume.shape[0], *padded_shape), dtype=torch.float32)
    padded_target = torch.zeros(padded_shape, dtype=torch.long)
    source_slices = tuple(slice(0, size) for size in spatial_shape)
    padded_volume[(slice(None),) + source_slices] = volume
    padded_target[source_slices] = target
    scores = np.zeros((4, *padded_shape), dtype=np.float32)
    counts = np.zeros(padded_shape, dtype=np.float32)
    losses = []
    starts = [window_starts(size, roi, overlap) for size, roi in zip(padded_shape, roi_size)]

    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                slices = (slice(x, x + roi_size[0]), slice(y, y + roi_size[1]),
                          slice(z, z + roi_size[2]))
                image_patch = padded_volume[(slice(None),) + slices].unsqueeze(0).to(device)
                target_patch = padded_target[slices].unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(image_patch)
                    loss = criterion(logits, target_patch)
                losses.append(float(loss.item()))
                scores[(slice(None),) + slices] += logits[0].float().cpu().numpy()
                counts[slices] += 1.0
    if np.any(counts == 0):
        raise AssertionError("Validation sliding windows left uncovered voxels")
    prediction = np.argmax(scores / counts[None], axis=0).astype(np.uint8)[source_slices]
    return prediction, float(np.mean(losses))


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def assert_dataset_isolation(train_dataset, val_dataset, splits: Dict[str, Sequence[str]]) -> None:
    train_ids, val_ids, test_ids = (set(splits[name]) for name in ("train", "val", "test"))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Patient split overlap detected")
    if set(train_dataset.case_ids) != train_ids:
        raise AssertionError("Training dataset does not exactly match manifest train IDs")
    if set(val_dataset.case_ids) != val_ids:
        raise AssertionError("Validation dataset does not exactly match manifest val IDs")

    train_case_paths = {path.resolve() for path in train_dataset.case_paths}
    forbidden_paths = {
        (train_dataset.dataset_root / case_id).resolve()
        for case_id in val_ids | test_ids
    }
    if train_case_paths & forbidden_paths:
        raise AssertionError("Validation/test patient path loaded by training dataset")
    for sample_path in train_dataset.sample_paths:
        if sample_path.parent.resolve() in forbidden_paths:
            raise AssertionError(f"Forbidden validation/test sample in training dataset: {sample_path}")
