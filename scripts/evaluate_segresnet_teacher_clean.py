#!/usr/bin/env python3
"""Evaluate the leakage-clean SegResNet teacher on one BraTS modality subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from monai import __version__ as monai_version
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_patient_split as base  # noqa: E402
from split_utils import load_split_manifest, resolve_case_paths  # noqa: E402

EXPECTED_MANIFEST_SHA256 = "9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881"
EXPECTED_PARAM_COUNT = 4_702_227
EXPECTED_REGION_ORDER = "(TC,WT,ET)"
MODEL_MODALITIES = ("t1", "t1ce", "t2", "flair")
OUTPUT_ROOT = (PROJECT_ROOT / "results" / "segresnet_teacher_clean_sweep").resolve()
REGION_TO_CHANNEL = {"tc": 0, "wt": 1, "et": 2}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def validate_output_paths(output_csv: Path) -> Path:
    output_csv = output_csv.resolve()
    if OUTPUT_ROOT not in output_csv.parents:
        raise ValueError(f"--output_csv must be inside {OUTPUT_ROOT}")
    if output_csv.suffix.lower() != ".csv":
        raise ValueError("--output_csv must end in .csv")
    output_json = output_csv.with_suffix(".json")
    if output_csv.exists() or output_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output_csv} / {output_json}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    return output_json


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "epoch", "val_mean_dice", "init_filters", "param_count",
        "seed", "monai_version", "git_commit_sha", "split_manifest_sha256", "region_order",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(f"Checkpoint is missing provenance keys: {missing}")
    if checkpoint["init_filters"] != 16:
        raise ValueError(f"Unexpected init_filters: {checkpoint['init_filters']}")
    if checkpoint["param_count"] != EXPECTED_PARAM_COUNT:
        raise ValueError(f"Unexpected checkpoint param_count: {checkpoint['param_count']}")
    if checkpoint["split_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Checkpoint split manifest SHA-256 does not match the clean test manifest")
    if checkpoint["region_order"] != EXPECTED_REGION_ORDER:
        raise ValueError(f"Unexpected checkpoint region order: {checkpoint['region_order']!r}")

    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if param_count != EXPECTED_PARAM_COUNT:
        raise RuntimeError(f"SegResNet parameter count {param_count} != {EXPECTED_PARAM_COUNT}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_regions(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    overlap: float,
) -> np.ndarray:
    """Match clean-teacher validation: FP32, constant blending, sigmoid regions."""
    inputs = torch.from_numpy(volume).unsqueeze(0)
    logits = sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        sw_batch_size=1,
        predictor=model,
        overlap=overlap,
        mode="constant",
        sw_device=device,
        device=torch.device("cpu"),
    )
    return (torch.sigmoid(logits[0].float()) > 0.5).numpy()


def case_metrics(
    prediction_regions: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Sequence[float],
) -> Dict[str, float]:
    if prediction_regions.shape[0] != 3:
        raise ValueError(f"Expected three region channels, got {prediction_regions.shape}")
    gt_regions = base.brats_regions(ground_truth, prediction=False)
    result = {}
    for region in base.REGIONS:
        prediction = prediction_regions[REGION_TO_CHANNEL[region]]
        result[f"dice_{region}"] = base.dice_score(prediction, gt_regions[region])
        result[f"hd95_{region}"] = base.hd95(prediction, gt_regions[region], spacing)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/segresnet_teacher_clean/best.pth"),
    )
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modalities", type=base.parse_modalities, default=MODEL_MODALITIES)
    parser.add_argument("--roi_size", type=base.parse_roi_size, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    split_json = args.split_json.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not split_json.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Split manifest or SegResNet checkpoint does not exist")
    manifest_sha256 = file_sha256(split_json)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split manifest SHA-256: {manifest_sha256}")
    output_json = validate_output_paths(args.output_csv)

    raw_manifest = json.loads(split_json.read_text(encoding="utf-8"))
    splits = load_split_manifest(str(split_json))
    if (len(splits["train"]), len(splits["val"]), len(splits["test"])) != (1000, 125, 126):
        raise AssertionError("Expected the clean 1000/125/126 split manifest")
    evaluation_ids = splits["test"]
    if set(evaluation_ids) & (set(splits["train"]) | set(splits["val"])):
        raise AssertionError("Held-out test split overlaps train or validation")
    dataset_root = base.resolve_dataset_root(split_json, raw_manifest["dataset_root"])
    case_paths = [Path(path) for path in resolve_case_paths(str(dataset_root), evaluation_ids)]
    model, checkpoint = load_model(checkpoint_path, device)

    print(f"Evaluator script: {Path(__file__).resolve()}", flush=True)
    print(f"MONAI runtime version: {monai_version}", flush=True)
    print(f"Checkpoint MONAI version: {checkpoint['monai_version']}", flush=True)
    print(f"Checkpoint epoch: {checkpoint['epoch']}", flush=True)
    print(f"Checkpoint SHA-256: {file_sha256(checkpoint_path)}", flush=True)
    print(f"Split manifest SHA-256: {manifest_sha256}", flush=True)
    print(f"Modalities: {args.modalities}", flush=True)
    print("Region order: (TC,WT,ET)", flush=True)

    modality_label = ",".join(args.modalities)
    rows = []
    for index, (case_id, case_path) in enumerate(zip(evaluation_ids, case_paths), start=1):
        volume, ground_truth, spacing, _ = base.load_case(case_path, args.modalities)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = predict_regions(model, volume, device, args.roi_size, args.overlap)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        else:
            peak_memory = 0.0
        elapsed = time.perf_counter() - started
        metrics = case_metrics(prediction, ground_truth, spacing)
        row = {
            "case_id": case_id,
            "modalities": modality_label,
            **metrics,
            "inference_time_sec": elapsed,
            "peak_gpu_memory_mb": peak_memory,
        }
        rows.append(row)
        print(
            f"[{index}/{len(case_paths)}] {case_id}: "
            f"WT={metrics['dice_wt']:.6f} TC={metrics['dice_tc']:.6f} "
            f"ET={metrics['dice_et']:.6f}",
            flush=True,
        )

    output_csv = args.output_csv.resolve()
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": "test",
        "modalities": list(args.modalities),
        "case_count": len(rows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_monai_version": checkpoint["monai_version"],
        "runtime_monai_version": monai_version,
        "checkpoint_git_commit_sha": checkpoint["git_commit_sha"],
        "evaluator_git_commit_sha": git_commit_sha(),
        "split_manifest_sha256": manifest_sha256,
        "region_order": EXPECTED_REGION_ORDER,
        "inference": {
            "roi_size": list(args.roi_size),
            "overlap": args.overlap,
            "blending": "constant",
            "precision": "float32",
            "threshold": 0.5,
        },
        "empty_mask_policy": {
            "both_empty": {"dice": 1.0, "hd95_mm": 0.0},
            "one_empty": {"dice": 0.0, "hd95_mm": "physical image diagonal"},
        },
        "statistics": base.aggregate(rows),
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved {output_csv} and {output_json}", flush=True)


if __name__ == "__main__":
    main()
