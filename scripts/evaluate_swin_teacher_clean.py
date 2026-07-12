#!/usr/bin/env python3
"""Evaluate the leakage-clean SwinUNETR teacher on all four modalities only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from monai.networks.nets import SwinUNETR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import evaluate_patient_split as student_eval  # noqa: E402
from split_utils import load_split_manifest, resolve_case_paths  # noqa: E402

EXPECTED_PARAM_COUNT = 15_705_621
EXPECTED_MANIFEST_SHA256 = "9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881"
MODEL_MODALITIES = ("t1", "t1ce", "t2", "flair")
REGION_ORDER = ("tc", "wt", "et")
ROI_SIZE = (128, 128, 128)
OVERLAP = 0.5
CSV_COLUMNS = student_eval.CSV_COLUMNS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Clean teacher checkpoint must contain 'model_state_dict'")
    if tuple(checkpoint.get("input_modality_order", ())) != MODEL_MODALITIES:
        raise AssertionError(
            f"Unexpected checkpoint modality order: {checkpoint.get('input_modality_order')}"
        )
    if tuple(checkpoint.get("region_order", ())) != REGION_ORDER:
        raise AssertionError(f"Unexpected checkpoint region order: {checkpoint.get('region_order')}")
    if int(checkpoint.get("feature_size", -1)) != 24:
        raise AssertionError(f"Unexpected feature_size: {checkpoint.get('feature_size')}")

    model = SwinUNETR(
        img_size=ROI_SIZE,
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=False,
    )
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"ASSERT trainable_param_count={param_count} expected={EXPECTED_PARAM_COUNT}", flush=True)
    if param_count != EXPECTED_PARAM_COUNT:
        raise AssertionError(
            f"Trainable parameter count {param_count} != {EXPECTED_PARAM_COUNT}"
        )
    if int(checkpoint.get("param_count", -1)) != EXPECTED_PARAM_COUNT:
        raise AssertionError(
            f"Checkpoint parameter count {checkpoint.get('param_count')} != {EXPECTED_PARAM_COUNT}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, checkpoint, param_count


@torch.inference_mode()
def sliding_window_region_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    roi_size: Tuple[int, int, int] = ROI_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    spatial_shape = volume.shape[1:]
    padded_shape = tuple(max(size, roi) for size, roi in zip(spatial_shape, roi_size))
    padded = np.zeros((4, *padded_shape), dtype=np.float32)
    padded[(slice(None),) + tuple(slice(0, size) for size in spatial_shape)] = volume
    score_sum = np.zeros((3, *padded_shape), dtype=np.float32)
    count_sum = np.zeros(padded_shape, dtype=np.float32)
    starts = [
        student_eval.window_starts(size, roi, overlap)
        for size, roi in zip(padded_shape, roi_size)
    ]
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                slices = (
                    slice(x, x + roi_size[0]),
                    slice(y, y + roi_size[1]),
                    slice(z, z + roi_size[2]),
                )
                patch = torch.from_numpy(padded[(slice(None),) + slices]).unsqueeze(0).to(
                    device=device, dtype=torch.float32,
                )
                logits = model(patch)
                score_sum[(slice(None),) + slices] += logits[0].float().cpu().numpy()
                count_sum[slices] += 1.0
    if np.any(count_sum == 0):
        raise AssertionError("Sliding-window inference left uncovered voxels")
    # sigmoid(logit) > 0.5 is exactly equivalent to logit > 0 and avoids overflow.
    prediction = (score_sum / count_sum[None]) > 0.0
    crop = tuple(slice(0, size) for size in spatial_shape)
    return prediction[(slice(None),) + crop]


def target_regions(raw_labels: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "wt": raw_labels > 0,
        "tc": (raw_labels == 1) | (raw_labels == 4),
        "et": raw_labels == 4,
    }


def region_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Sequence[float],
) -> Dict[str, float]:
    predicted = {name: prediction[index] for index, name in enumerate(REGION_ORDER)}
    target = target_regions(ground_truth)
    metrics: Dict[str, float] = {}
    for region in student_eval.REGIONS:
        metrics[f"dice_{region}"] = student_eval.dice_score(predicted[region], target[region])
        metrics[f"hd95_{region}"] = student_eval.hd95(
            predicted[region], target[region], spacing,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/swin_teacher_clean/best.pth"),
    )
    parser.add_argument(
        "--split_json", type=Path, default=Path("splits/patient_splits.json"),
    )
    parser.add_argument(
        "--output_csv", type=Path,
        default=Path("results/04_kd_clean_swin/swin_teacher_clean_test/test_all_swin_teacher_clean.csv"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    split_json = args.split_json.resolve()
    output_csv = args.output_csv.resolve()
    output_json = output_csv.with_suffix(".json")
    expected_output_dir = (
        PROJECT_ROOT / "results" / "04_kd_clean_swin" / "swin_teacher_clean_test"
    ).resolve()
    if output_csv.parent != expected_output_dir or output_csv.name != "test_all_swin_teacher_clean.csv":
        raise ValueError(f"Output must be {expected_output_dir / 'test_all_swin_teacher_clean.csv'}")
    if expected_output_dir.exists() or output_csv.exists() or output_json.exists():
        raise FileExistsError(f"Refusing to reuse output path: {expected_output_dir}")
    if not checkpoint_path.is_file() or not split_json.is_file():
        raise FileNotFoundError("Checkpoint or split manifest not found")

    manifest_sha256 = file_sha256(split_json)
    print(
        f"ASSERT split_manifest_sha256={manifest_sha256} "
        f"expected={EXPECTED_MANIFEST_SHA256}", flush=True,
    )
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Leakage-safe split manifest SHA-256 mismatch")
    splits = load_split_manifest(str(split_json))
    if tuple(len(splits[name]) for name in ("train", "val", "test")) != (1000, 125, 126):
        raise AssertionError("Expected the 1000/125/126 leakage-safe split")
    if (set(splits["train"]) | set(splits["val"])) & set(splits["test"]):
        raise AssertionError("Patient leakage: train/val intersects test")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, checkpoint, param_count = build_model(checkpoint_path, device)
    if checkpoint.get("split_manifest_sha256") != manifest_sha256:
        raise AssertionError("Checkpoint and evaluation split-manifest hashes differ")
    print(
        "EVAL all_four_only=True input_order=(t1,t1ce,t2,flair) "
        "output_order=(TC,WT,ET) roi=(128,128,128) overlap=0.5 precision=fp32",
        flush=True,
    )

    raw_manifest = json.loads(split_json.read_text(encoding="utf-8"))
    dataset_root = student_eval.resolve_dataset_root(split_json, raw_manifest["dataset_root"])
    case_paths = [
        Path(path) for path in resolve_case_paths(str(dataset_root), splits["test"])
    ]
    rows = []
    for index, (case_id, case_path) in enumerate(zip(splits["test"], case_paths), start=1):
        volume, ground_truth, spacing, avail_mask = student_eval.load_case(
            case_path, MODEL_MODALITIES,
        )
        if not bool(avail_mask.all()):
            raise AssertionError(f"All-four availability failed for {case_id}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = sliding_window_region_predict(model, volume, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        metrics = region_metrics(prediction, ground_truth, spacing)
        row = {
            "case_id": case_id,
            "modalities": ",".join(MODEL_MODALITIES),
            **metrics,
            "inference_time_sec": elapsed,
            "peak_gpu_memory_mb": peak_memory,
        }
        rows.append(row)
        print(
            f"[{index}/126] {case_id}: WT={metrics['dice_wt']:.6f} "
            f"TC={metrics['dice_tc']:.6f} ET={metrics['dice_et']:.6f}",
            flush=True,
        )
    if len(rows) != 126:
        raise AssertionError(f"Expected 126 test cases, evaluated {len(rows)}")

    statistics = student_eval.aggregate(rows)
    expected_output_dir.mkdir(parents=True, exist_ok=False)
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": "test",
        "modalities": list(MODEL_MODALITIES),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "case_count": len(rows),
        "trainable_param_count": param_count,
        "split_manifest_sha256": manifest_sha256,
        "roi_size": list(ROI_SIZE),
        "overlap": OVERLAP,
        "precision": "float32_no_autocast_tf32_disabled",
        "input_modality_order": list(MODEL_MODALITIES),
        "region_order": list(REGION_ORDER),
        "region_threshold": 0.5,
        "empty_mask_policy": {
            "both_empty": {"dice": 1.0, "hd95_mm": 0.0},
            "one_empty": {"dice": 0.0, "hd95_mm": "physical image diagonal"},
        },
        "statistics": statistics,
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        f"ALL4_TEST_MEAN WT={statistics['dice_wt']['mean']:.12f} "
        f"TC={statistics['dice_tc']['mean']:.12f} "
        f"ET={statistics['dice_et']['mean']:.12f}",
        flush=True,
    )
    print(f"Saved {output_csv} and {output_json}", flush=True)
    print("No existing files were modified.", flush=True)


if __name__ == "__main__":
    main()
