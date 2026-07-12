#!/usr/bin/env python3
"""Leakage-safe 15-subset evaluation for clean-teacher MACAF KD.

Empty-mask convention:
* prediction and ground truth both empty: Dice=1 and HD95=0 mm;
* exactly one mask empty: Dice=0 and HD95=the physical image diagonal.

The finite diagonal penalty keeps aggregate statistics well-defined while making
complete false-positive/false-negative region failures explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Cfg  # noqa: E402
from main import MultiEncoderRMDUNet  # noqa: E402
from split_utils import load_split_manifest, resolve_case_paths  # noqa: E402

MODEL_MODALITIES = ("t1", "t1ce", "t2", "flair")
REGIONS = ("wt", "tc", "et")
CSV_COLUMNS = (
    "case_id", "modalities", "dice_wt", "dice_tc", "dice_et",
    "hd95_wt", "hd95_tc", "hd95_et", "inference_time_sec",
    "peak_gpu_memory_mb",
)
METRIC_COLUMNS = CSV_COLUMNS[2:]


def parse_modalities(value: str) -> Tuple[str, ...]:
    value = value.strip().lower()
    selected = MODEL_MODALITIES if value == "all" else tuple(part.strip() for part in value.split(","))
    if not selected or any(part not in MODEL_MODALITIES for part in selected):
        raise argparse.ArgumentTypeError(
            "modalities must be 'all' or a comma-separated subset of t1,t1ce,t2,flair"
        )
    if len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("modalities must not contain duplicates")
    return tuple(modality for modality in MODEL_MODALITIES if modality in selected)


def parse_roi_size(value: str) -> Tuple[int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI size must contain three integers") from exc
    if len(result) != 3 or any(size <= 0 or size % 16 for size in result):
        raise argparse.ArgumentTypeError("ROI dimensions must be positive multiples of 16")
    return result


def resolve_dataset_root(manifest_path: Path, dataset_root: str) -> Path:
    root = Path(dataset_root)
    if not root.is_absolute():
        root = manifest_path.resolve().parent.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mask = volume > 0
    values = volume[mask] if mask.any() else volume
    std = float(values.std())
    if std < 1e-8:
        return volume.copy()
    output = volume.copy()
    output[mask] = (volume[mask] - float(values.mean())) / (std + 1e-8)
    return output


def load_case(case_path: Path, selected_modalities: Sequence[str]):
    channels: List[np.ndarray] = []
    reference = None
    shape = None
    for modality in MODEL_MODALITIES:
        path = case_path / f"{case_path.name}_{modality}.nii.gz"
        if modality in selected_modalities:
            image = nib.load(str(path))
            array = zscore_nonzero(image.get_fdata(dtype=np.float32))
            if reference is None:
                reference = image
            shape = array.shape
            channels.append(array)
        else:
            channels.append(None)
    if reference is None or shape is None:
        raise AssertionError("At least one MRI modality must be selected")
    for index, channel in enumerate(channels):
        if channel is None:
            channels[index] = np.zeros(shape, dtype=np.float32)
        elif channel.shape != shape:
            raise ValueError(f"Modality shape mismatch in {case_path.name}")

    gt_image = nib.load(str(case_path / f"{case_path.name}_seg.nii.gz"))
    ground_truth = gt_image.get_fdata(dtype=np.float32).astype(np.uint8)
    if ground_truth.shape != shape:
        raise ValueError(f"Ground-truth shape mismatch in {case_path.name}")
    return np.stack(channels), ground_truth, tuple(float(v) for v in gt_image.header.get_zooms()[:3])


def window_starts(length: int, roi: int, overlap: float) -> List[int]:
    if length <= roi:
        return [0]
    stride = max(1, int(roi * (1.0 - overlap)))
    starts = list(range(0, length - roi + 1, stride))
    if starts[-1] != length - roi:
        starts.append(length - roi)
    return starts


@torch.inference_mode()
def sliding_window_predict(model, volume: np.ndarray, device: torch.device,
                           roi_size: Tuple[int, int, int], overlap: float,
                           avail_mask: torch.Tensor | None = None) -> np.ndarray:
    """Run tiled inference and average overlapping logits on CPU."""
    spatial_shape = volume.shape[1:]
    padded_shape = tuple(max(size, roi) for size, roi in zip(spatial_shape, roi_size))
    padded = np.zeros((volume.shape[0], *padded_shape), dtype=np.float32)
    padded[(slice(None),) + tuple(slice(0, size) for size in spatial_shape)] = volume
    score_sum = np.zeros((Cfg.num_classes, *padded_shape), dtype=np.float32)
    count_sum = np.zeros(padded_shape, dtype=np.float32)
    starts = [window_starts(size, roi, overlap) for size, roi in zip(padded_shape, roi_size)]

    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                slices = (slice(x, x + roi_size[0]), slice(y, y + roi_size[1]), slice(z, z + roi_size[2]))
                patch = torch.from_numpy(padded[(slice(None),) + slices]).unsqueeze(0).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(patch, avail_mask=avail_mask)
                logits = logits[0].float().cpu().numpy()
                score_sum[(slice(None),) + slices] += logits
                count_sum[slices] += 1.0
    if np.any(count_sum == 0):
        raise AssertionError("Sliding-window inference left uncovered voxels")
    prediction = np.argmax(score_sum / count_sum[None], axis=0).astype(np.uint8)
    crop = tuple(slice(0, size) for size in spatial_shape)
    return prediction[crop]


def brats_regions(labels: np.ndarray, prediction: bool) -> Dict[str, np.ndarray]:
    # Model indices: 1=NCR/NET, 2=ED, 3=ET. BraTS ground truth uses ET=4.
    et_label = 3 if prediction else 4
    return {
        "wt": labels > 0,
        "tc": (labels == 1) | (labels == et_label),
        "et": labels == et_label,
    }


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred_count = int(prediction.sum())
    gt_count = int(ground_truth.sum())
    if pred_count + gt_count == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    return float(2.0 * intersection / (pred_count + gt_count))


def hd95(prediction: np.ndarray, ground_truth: np.ndarray,
         spacing: Sequence[float]) -> float:
    pred_empty = not prediction.any()
    gt_empty = not ground_truth.any()
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float(math.sqrt(sum(((size - 1) * step) ** 2
                                   for size, step in zip(prediction.shape, spacing))))

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    gt_surface = ground_truth ^ binary_erosion(ground_truth, structure=structure, border_value=0)
    scale = np.asarray(spacing, dtype=np.float64)
    pred_points = np.argwhere(pred_surface) * scale
    gt_points = np.argwhere(gt_surface) * scale
    pred_to_gt = cKDTree(gt_points).query(pred_points, k=1, workers=-1)[0]
    gt_to_pred = cKDTree(pred_points).query(gt_points, k=1, workers=-1)[0]
    distances = np.concatenate((pred_to_gt, gt_to_pred))
    return float(np.percentile(distances, 95))


def case_metrics(prediction: np.ndarray, ground_truth: np.ndarray,
                 spacing: Sequence[float]) -> Dict[str, float]:
    pred_regions = brats_regions(prediction, prediction=True)
    gt_regions = brats_regions(ground_truth, prediction=False)
    result = {}
    for region in REGIONS:
        result[f"dice_{region}"] = dice_score(pred_regions[region], gt_regions[region])
        result[f"hd95_{region}"] = hd95(pred_regions[region], gt_regions[region], spacing)
    return result


def aggregate(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    output = {}
    for metric in METRIC_COLUMNS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        q25, q75 = np.percentile(values, [25, 75])
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
        }
    return output


def load_model(checkpoint_path: Path, device: torch.device, fusion_reduction: int):
    model = MultiEncoderRMDUNet(
        in_modalities=len(MODEL_MODALITIES), base_ch=16, num_stages=4,
        num_classes=Cfg.num_classes, rmd_enable=False,
        fusion_reduction=fusion_reduction,
    ).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 5_815_280:
        raise RuntimeError(
            f"MACAF parameter mismatch: expected=5815280, actual={parameter_count}"
        )
    model.eval()
    model.rmd_enable = False
    return model


def validate_output_paths(output_csv: Path, output_dir: Path) -> Path:
    safe_root = output_dir.resolve()
    output_csv = output_csv.resolve()
    if safe_root not in output_csv.parents:
        raise ValueError(f"--output_csv must be inside {safe_root}")
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("--output_csv must end in .csv")
    output_json = output_csv.with_suffix(".json")
    if output_csv.exists() or output_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output_csv} / {output_json}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    return output_json


def modality_availability(selected_modalities: Sequence[str], device: torch.device) -> torch.Tensor | None:
    if tuple(selected_modalities) == MODEL_MODALITIES:
        return None
    return torch.tensor(
        [modality in selected_modalities for modality in MODEL_MODALITIES],
        dtype=torch.bool,
        device=device,
    )


def all_modality_subsets() -> List[Tuple[str, ...]]:
    subsets: List[Tuple[str, ...]] = []
    for size in range(1, len(MODEL_MODALITIES) + 1):
        subsets.extend(tuple(combo) for combo in combinations(MODEL_MODALITIES, size))
    return subsets


def output_stem(selected_modalities: Sequence[str]) -> str:
    return "all" if tuple(selected_modalities) == MODEL_MODALITIES else "_".join(selected_modalities)


def evaluate_subset(args, model, device: torch.device, evaluation_ids: Sequence[str],
                    case_paths: Sequence[Path], selected_modalities: Sequence[str],
                    output_csv: Path) -> None:
    output_json = validate_output_paths(output_csv, args.output_dir)
    modality_label = ",".join(selected_modalities)
    avail_mask = modality_availability(selected_modalities, device)
    rows = []
    for index, (case_id, case_path) in enumerate(zip(evaluation_ids, case_paths), start=1):
        volume, ground_truth, spacing = load_case(case_path, selected_modalities)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = sliding_window_predict(
            model, volume, device, args.roi_size, args.overlap, avail_mask=avail_mask,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        else:
            peak_memory = 0.0
        elapsed = time.perf_counter() - started
        metrics = case_metrics(prediction, ground_truth, spacing)
        row = {"case_id": case_id, "modalities": modality_label, **metrics,
               "inference_time_sec": elapsed, "peak_gpu_memory_mb": peak_memory}
        rows.append(row)
        print(f"[{index}/{len(case_paths)}] {case_id} {modality_label}: Dice WT={metrics['dice_wt']:.6f}")

    with output_csv.resolve().open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split, "fold": args.fold, "modalities": list(selected_modalities),
        "model_modality_order": list(MODEL_MODALITIES),
        "avail_mask": None if avail_mask is None else [bool(v) for v in avail_mask.cpu().tolist()],
        "checkpoint": str(args.checkpoint.resolve()), "case_count": len(rows),
        "empty_mask_policy": {
            "both_empty": {"dice": 1.0, "hd95_mm": 0.0},
            "one_empty": {"dice": 0.0, "hd95_mm": "physical image diagonal"},
        },
        "statistics": aggregate(rows),
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved {output_csv.resolve()} and {output_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/swin_kd_clean_B_macaf/best.pth"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/macaf_experiments/kd_clean_swin/sweep"))
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modalities", type=parse_modalities, default=MODEL_MODALITIES)
    parser.add_argument("--sweep", action="store_true", help="Evaluate all 15 non-empty modality subsets")
    parser.add_argument("--fusion_reduction", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0, help="Fold index for k-fold manifests")
    parser.add_argument("--roi_size", type=parse_roi_size, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu explicitly")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.fusion_reduction < 1:
        parser.error("--fusion_reduction must be positive")
    if not args.split_json.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Split manifest or checkpoint does not exist")

    raw_manifest = json.loads(args.split_json.read_text(encoding="utf-8"))
    splits = load_split_manifest(str(args.split_json), args.fold)
    evaluation_ids = splits[args.split]
    if not evaluation_ids:
        raise ValueError(f"The {args.split} split is empty")
    if set(evaluation_ids) & set(splits["train"]):
        raise AssertionError("Evaluation split overlaps training patients")
    dataset_root = resolve_dataset_root(args.split_json, raw_manifest["dataset_root"])
    case_paths = [Path(path) for path in resolve_case_paths(str(dataset_root), evaluation_ids)]
    model = load_model(args.checkpoint.resolve(), device, args.fusion_reduction)

    if args.sweep:
        subsets = all_modality_subsets()
        for selected_modalities in subsets:
            output_csv = (
                args.output_dir /
                f"{args.split}_{output_stem(selected_modalities)}_macaf_clean_swin_kd_sweep.csv"
            )
            evaluate_subset(args, model, device, evaluation_ids, case_paths, selected_modalities, output_csv)
    else:
        output_csv = args.output_csv or (
            args.output_dir /
            f"{args.split}_{output_stem(args.modalities)}_macaf_clean_swin_kd_sweep.csv"
        )
        evaluate_subset(args, model, device, evaluation_ids, case_paths, args.modalities, output_csv)


if __name__ == "__main__":
    main()
