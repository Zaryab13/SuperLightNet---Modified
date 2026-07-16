#!/usr/bin/env python3
"""Train a leakage-clean SwinUNETR teacher from random initialization on PC3."""

from __future__ import annotations

import argparse
from concurrent.futures import process
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from monai import __version__ as monai_version
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset, PatientVolumeDataset  # noqa: E402
from superlightnet.training import assert_dataset_isolation, save_checkpoint_atomic  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "swin_teacher_clean"
RESULT_DIR = PROJECT_ROOT / "results" / "swin_teacher_clean"
BEST_PATH = CHECKPOINT_DIR / "best.pth"
LAST_PATH = CHECKPOINT_DIR / "last.pth"
LOG_PATH = RESULT_DIR / "training_log.csv"

EXPECTED_MONAI_VERSION = "1.5.2"
EXPECTED_PARAM_COUNT = 15_705_621
FEATURE_SIZE = 24
MAX_EPOCHS = 150
WARMUP_EPOCHS = 25
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5
ETA_MIN = 0.0
BATCH_SIZE = 1
VAL_ROI = (128, 128, 128)
VAL_OVERLAP = 0.5
VAL_EVERY = 5
SEED = 42
WALL_CLOCK_LIMIT_SEC = 66.0 * 60.0 * 60.0
REGION_ORDER = ("tc", "wt", "et")
INPUT_MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
LOG_COLUMNS = (
    "epoch", "train_loss", "learning_rate", "elapsed_sec",
    "val_dice_tc", "val_dice_wt", "val_dice_et", "val_mean_dice", "is_best",
)
OUR_CHOICE_LINES = (
    "[OUR CHOICE: feature_size=24; official is 48. Compute budget.]",
    "[OUR CHOICE: official is 300 epochs / 50 warmup. Shape preserved, scaled 1/2.]",
    "seed=42.  [OUR CHOICE]",
    "Validate every 5 epochs.  [OUR CHOICE; official is 10]",
)
FORBIDDEN_PRETRAINED_DIRS = (
    (PROJECT_ROOT / "pretrained_teachers").resolve(),
    (PROJECT_ROOT / "checkpoints" / "swin_kd_v1").resolve(),
)


def parse_roi_size(value: str) -> Tuple[int, int, int]:
    try:
        roi = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain three integers") from exc
    if len(roi) != 3 or any(size not in (96, 128) for size in roi):
        raise argparse.ArgumentTypeError("Training ROI must be 128,128,128 or the OOM fallback 96,96,96")
    if len(set(roi)) != 1:
        raise argparse.ArgumentTypeError("Training ROI must be cubic")
    return roi  # type: ignore[return-value]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def resolve_dataset_root(split_json: Path) -> Path:
    manifest = json.loads(split_json.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    if not root.is_absolute():
        root = split_json.resolve().parent.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def assert_not_forbidden_checkpoint(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in FORBIDDEN_PRETRAINED_DIRS:
        if resolved == forbidden or forbidden in resolved.parents:
            raise RuntimeError(f"Refusing to load contaminated pretrained checkpoint: {resolved}")


def check_gpu_idle(gpu_index: int) -> None:
    """Print nvidia-smi memory and abort if a compute process uses more than 1 GiB."""
    summary = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu_index}",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    print(f"NVIDIA-SMI GPU MEMORY (index,name,used_MiB,total_MiB): {summary.stdout.strip()}", flush=True)

    try:
        from pynvml import (  # type: ignore[import-not-found]
            NVMLError,
            nvmlDeviceGetComputeRunningProcesses,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
            nvmlShutdown,
        )
    except ImportError as exc:
        raise RuntimeError("nvidia-ml-py is required to verify that the GPU is free") from exc

    busy = []
    nvmlInit()
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        total_memory = int(nvmlDeviceGetMemoryInfo(handle).total)
        try:
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
        except NVMLError as exc:
            raise RuntimeError("Unable to query GPU compute-process memory safely") from exc

        for process in processes:
            used = process.usedGpuMemory

            if used is None:
                continue

            used = int(used)

            if used <= 0 or used > total_memory:
                raise RuntimeError(f"Unable to verify GPU memory for compute PID {process.pid}")

            if process.pid != os.getpid() and used > 1024**3:
                busy.append((int(process.pid), used / 1024**2))

    finally:
        nvmlShutdown()

    if busy:
        print(f"GPU BUSY - not starting: {busy}", flush=True)
        raise SystemExit(2)

def labels_to_regions(labels: torch.Tensor) -> torch.Tensor:
    """Convert contiguous BraTS labels to output order (TC, WT, ET)."""
    # TC = labels {1,3}; WT = labels > 0; ET = label 3. Do not reorder.
    tc = (labels == 1) | (labels == 3)
    wt = labels > 0
    et = labels == 3
    return torch.stack((tc, wt, et), dim=1).float()


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / denominator)


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in REGION_ORDER}
    for index, (volume, target, case_ids) in enumerate(loader, start=1):
        if volume.shape[0] != 1 or target.shape[0] != 1:
            raise AssertionError("Validation requires batch_size=1")
        logits = sliding_window_inference(
            inputs=volume,
            roi_size=VAL_ROI,
            sw_batch_size=1,
            predictor=model,
            overlap=VAL_OVERLAP,
            mode="constant",
            sw_device=device,
            device=torch.device("cpu"),
        )
        prediction = (torch.sigmoid(logits[0].float()) > 0.5).numpy()
        target_regions = labels_to_regions(target.long())[0].bool().numpy()
        for region_index, name in enumerate(REGION_ORDER):
            totals[name] += binary_dice(prediction[region_index], target_regions[region_index])
        print(f"Validation {index}/{len(loader)} {case_ids[0]}", flush=True)
    return {name: total / len(loader) for name, total in totals.items()}


def lr_factor(epoch_index: int) -> float:
    """Linear warmup for 25 epochs, then cosine decay to zero at epoch 150."""
    epoch_number = epoch_index + 1
    if epoch_number <= WARMUP_EPOCHS:
        return epoch_number / WARMUP_EPOCHS
    progress = (epoch_number - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)
    return ETA_MIN / BASE_LR + (1.0 - ETA_MIN / BASE_LR) * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_payload(
    model: torch.nn.Module,
    epoch: int,
    scores: Dict[str, float],
    val_mean: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    manifest_sha256: str,
    commit_sha: str,
    best_val_mean: float,
    training_wall_time_sec: float,
    train_roi: Tuple[int, int, int],
    validated: bool,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_dice_tc": scores["tc"],
        "val_dice_wt": scores["wt"],
        "val_dice_et": scores["et"],
        "val_mean_dice": val_mean,
        "feature_size": FEATURE_SIZE,
        "param_count": EXPECTED_PARAM_COUNT,
        "seed": SEED,
        "git_commit_sha": commit_sha,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_mean_dice": best_val_mean,
        "training_wall_time_sec": training_wall_time_sec,
        "split_manifest_sha256": manifest_sha256,
        "train_roi_size": train_roi,
        "val_roi_size": VAL_ROI,
        "validated": validated,
        "random_initialization": True,
        "region_order": REGION_ORDER,
        "input_modality_order": INPUT_MODALITY_ORDER,
        "monai_version": monai_version,
    }


def last_logged_epoch(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["epoch"]) if rows else 0


def validate_output_state(resume: bool) -> None:
    if resume:
        assert_not_forbidden_checkpoint(LAST_PATH)
        missing = [path for path in (LAST_PATH, LOG_PATH) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot resume; missing {missing[0]}")
    else:
        existing = [path for path in (CHECKPOINT_DIR, RESULT_DIR) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing clean-teacher output: {existing[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=0)
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--train_roi_size", type=parse_roi_size, default=(128, 128, 128))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 0 or args.val_workers < 0 or args.samples_per_patient < 1:
        parser.error("worker counts must be non-negative and samples_per_patient must be positive")
    if monai_version != EXPECTED_MONAI_VERSION:
        raise RuntimeError(
            f"MONAI {EXPECTED_MONAI_VERSION} is required; found {monai_version}. "
            "Do not pass img_size to the MONAI 1.5.2 SwinUNETR."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for clean SwinUNETR teacher training")

    check_gpu_idle(args.gpu_index)
    validate_output_state(args.resume)

    split_json = args.split_json.resolve()
    if not split_json.is_file():
        raise FileNotFoundError(split_json)
    manifest_sha256 = file_sha256(split_json)
    splits = load_split_manifest(str(split_json))
    if (len(splits["train"]), len(splits["val"]), len(splits["test"])) != (1000, 125, 126):
        raise AssertionError("Expected the 1000/125/126 leakage-safe manifest")
    train_ids, val_ids, test_ids = (set(splits[name]) for name in ("train", "val", "test"))
    if (train_ids | val_ids) & test_ids:
        raise AssertionError("Patient leakage: train/val intersects the held-out test IDs")

    dataset_root = resolve_dataset_root(split_json)
    train_dataset = PatientPatchDataset(
        dataset_root,
        splits["train"],
        roi_size=args.train_roi_size,
        samples_per_patient=args.samples_per_patient,
        seed=SEED,
        augment=True,
    )
    val_dataset = PatientVolumeDataset(dataset_root, splits["val"])
    assert_dataset_isolation(train_dataset, val_dataset, splits)
    print("LEAKAGE-CHECK PASSED", flush=True)
    print("Test IDs were checked for disjointness; no test dataset or test image was instantiated.", flush=True)

    print("OUR CHOICES", flush=True)
    for line in OUR_CHOICE_LINES:
        print(line, flush=True)
    print(f"Input channel order: {INPUT_MODALITY_ORDER}", flush=True)
    print("Output channel order: (TC, WT, ET)", flush=True)
    print("TC={1,3}; WT=label>0; ET={3}", flush=True)
    print("No pretrained weights loaded - training from random initialisation.", flush=True)

    set_seed(SEED)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=FEATURE_SIZE,
        use_checkpoint=True,
    ).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameter count: {param_count}", flush=True)
    if param_count != EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"Unexpected SwinUNETR parameter count: {param_count}; expected {EXPECTED_PARAM_COUNT}"
        )

    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    commit_sha = git_commit_sha()

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=True,
        persistent_workers=args.val_workers > 0,
    )

    start_epoch = 1
    best_val_mean = float("-inf")
    accumulated_wall_time = 0.0
    last_scores = {name: float("nan") for name in REGION_ORDER}
    last_val_mean = float("nan")
    if args.resume:
        checkpoint = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
        expected_resume = {
            "feature_size": FEATURE_SIZE,
            "param_count": EXPECTED_PARAM_COUNT,
            "seed": SEED,
            "split_manifest_sha256": manifest_sha256,
            "random_initialization": True,
            "monai_version": monai_version,
        }
        for key, expected in expected_resume.items():
            if checkpoint.get(key) != expected:
                raise ValueError(
                    f"Resume mismatch for {key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                )
        completed_epoch = int(checkpoint["epoch"])
        if last_logged_epoch(LOG_PATH) != completed_epoch:
            raise ValueError("training_log.csv and last.pth disagree on completed epoch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = completed_epoch + 1
        best_val_mean = float(checkpoint["best_val_mean_dice"])
        accumulated_wall_time = float(checkpoint.get("training_wall_time_sec", 0.0))
        last_scores = {name: float(checkpoint[f"val_dice_{name}"]) for name in REGION_ORDER}
        last_val_mean = float(checkpoint["val_mean_dice"])
        print(
            f"Resuming clean teacher at epoch {start_epoch}; "
            f"accumulated training time={accumulated_wall_time / 3600.0:.2f}h; "
            f"best val mean={best_val_mean:.6f}",
            flush=True,
        )
    else:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
        RESULT_DIR.mkdir(parents=True, exist_ok=False)
        initial_payload = checkpoint_payload(
            model, 0, last_scores, float("-inf"), optimizer, scheduler,
            manifest_sha256, commit_sha, best_val_mean, 0.0, args.train_roi_size, False,
        )
        save_checkpoint_atomic(BEST_PATH, initial_payload)
        save_checkpoint_atomic(LAST_PATH, initial_payload)

    if start_epoch > MAX_EPOCHS:
        print(f"Training already completed through epoch {start_epoch - 1}.", flush=True)
        return
    if accumulated_wall_time >= WALL_CLOCK_LIMIT_SEC:
        print(
            f"HARD STOP: epoch reached={start_epoch - 1}; best val mean Dice={best_val_mean:.6f}",
            flush=True,
        )
        return

    log_mode = "a" if args.resume else "x"
    session_started = time.perf_counter()
    with LOG_PATH.open(log_mode, newline="", encoding="utf-8") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=LOG_COLUMNS)
        if not args.resume:
            writer.writeheader()
            log_handle.flush()

        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            epoch_started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            random.seed(SEED + epoch)
            loader_generator.manual_seed(SEED + epoch)
            model.train()
            loss_total = 0.0
            learning_rate = float(optimizer.param_groups[0]["lr"])

            for step, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                targets = labels_to_regions(labels)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite FP32 loss at epoch {epoch}, step {step}")
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item())
                if step % 25 == 0 or step == len(train_loader):
                    print(
                        f"Epoch {epoch}/{MAX_EPOCHS} train {step}/{len(train_loader)} "
                        f"loss={loss.item():.6f} lr={learning_rate:.8g}",
                        flush=True,
                    )

            train_loss = loss_total / len(train_loader)
            scores = {name: float("nan") for name in REGION_ORDER}
            val_mean = float("nan")
            is_best = False
            if epoch % VAL_EVERY == 0:
                scores = validate(model, val_loader, device)
                val_mean = float(np.mean([scores[name] for name in REGION_ORDER]))
                last_scores = scores
                last_val_mean = val_mean
                if val_mean > best_val_mean:
                    best_val_mean = val_mean
                    is_best = True

            scheduler.step()
            epoch_elapsed = time.perf_counter() - epoch_started
            total_wall_time = accumulated_wall_time + (time.perf_counter() - session_started)
            payload = checkpoint_payload(
                model, epoch, scores if epoch % VAL_EVERY == 0 else last_scores,
                val_mean if epoch % VAL_EVERY == 0 else last_val_mean,
                optimizer, scheduler, manifest_sha256, commit_sha, best_val_mean,
                total_wall_time, args.train_roi_size, epoch % VAL_EVERY == 0,
            )
            save_checkpoint_atomic(LAST_PATH, payload)
            if is_best:
                save_checkpoint_atomic(BEST_PATH, payload)
                print(f"Saved new best checkpoint: val mean Dice={best_val_mean:.6f}", flush=True)

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "elapsed_sec": epoch_elapsed,
                "val_dice_tc": "" if epoch % VAL_EVERY else scores["tc"],
                "val_dice_wt": "" if epoch % VAL_EVERY else scores["wt"],
                "val_dice_et": "" if epoch % VAL_EVERY else scores["et"],
                "val_mean_dice": "" if epoch % VAL_EVERY else val_mean,
                "is_best": is_best,
            })
            log_handle.flush()
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f} "
                f"val_mean={val_mean if epoch % VAL_EVERY == 0 else 'NA'} "
                f"elapsed={epoch_elapsed:.1f}s total_hours={total_wall_time / 3600.0:.2f}",
                flush=True,
            )

            if total_wall_time >= WALL_CLOCK_LIMIT_SEC:
                print(
                    f"HARD STOP: epoch reached={epoch}; best val mean Dice={best_val_mean:.6f}",
                    flush=True,
                )
                break

    print("No existing files were modified. Test set was never accessed.", flush=True)


if __name__ == "__main__":
    main()
