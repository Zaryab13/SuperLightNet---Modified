#!/usr/bin/env python3
"""Train SuperLightNet from scratch on leakage-safe patient-level splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss, MultiEncoderRMDUNet  # noqa: E402
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset, PatientVolumeDataset  # noqa: E402
from superlightnet.training import (  # noqa: E402
    assert_dataset_isolation,
    region_dice,
    save_checkpoint_atomic,
    validate_volume,
)

LOG_COLUMNS = (
    "epoch", "train_loss", "val_loss", "val_dice_wt", "val_dice_tc",
    "val_dice_et", "learning_rate",
)
BATCH_LOG_COLUMNS = ("epoch", "step", "global_step", "batch_loss", "learning_rate")


def parse_roi_size(value: str):
    try:
        roi = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain three integers") from exc
    if len(roi) != 3 or any(size <= 0 or size % 16 for size in roi):
        raise argparse.ArgumentTypeError("ROI dimensions must be positive multiples of 16")
    return roi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dataset_root(split_json: Path) -> Path:
    manifest = json.loads(split_json.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    if not root.is_absolute():
        root = split_json.resolve().parent.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root


def last_logged_epoch(log_path: Path) -> int:
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training log has no completed epochs: {log_path}")
    return int(rows[-1]["epoch"])


def validate_output_locations(output_dir: Path, resume: Path | None):
    expected = (PROJECT_ROOT / "checkpoints" / "leakage_safe").resolve()
    output_dir = output_dir.resolve()
    if output_dir != expected:
        raise ValueError(f"--output_dir must resolve to {expected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_patient_split.pth"
    last_path = output_dir / "last_patient_split.pth"
    log_path = PROJECT_ROOT / "results" / "leakage_safe" / "training_log.csv"
    batch_log_path = PROJECT_ROOT / "results" / "leakage_safe" / "training_batch_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if resume is None:
        existing = [path for path in (best_path, last_path, log_path, batch_log_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing training artifact: {existing[0]}")
    else:
        if resume.resolve() != last_path.resolve():
            raise ValueError(f"Resume checkpoint must be the run's last checkpoint: {last_path}")
        missing = [path for path in (last_path, log_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot resume; required artifact is missing: {missing[0]}")
    return best_path, last_path, log_path, batch_log_path


def load_resume_state(path: Path, model, optimizer, scaler, manifest_sha256: str,
                      args, log_path: Path):
    try:
        checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=args.device)
    expected = {
        "split_manifest_sha256": manifest_sha256,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "roi_size": tuple(args.roi_size),
        "seed": args.seed,
    }
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if key == "roi_size" and actual is not None:
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"Resume checkpoint mismatch for {key}: expected {value!r}, got {actual!r}")
    completed_epoch = int(checkpoint["epoch"])
    if last_logged_epoch(log_path) != completed_epoch:
        raise ValueError("training_log.csv and last checkpoint disagree on the completed epoch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["opt"])
    scaler.load_state_dict(checkpoint["scaler"])
    return completed_epoch + 1, float(checkpoint["best_val_dice"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--train_split", choices=("train",), default="train")
    parser.add_argument("--val_split", choices=("val",), default="val")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/leakage_safe"))
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roi_size", type=parse_roi_size, default=(160, 160, 160))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_workers", type=int, default=0)
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Resume from last_patient_split.pth")
    args = parser.parse_args()

    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or
            args.num_workers < 0 or args.val_workers < 0):
        parser.error("epochs, batch_size, and learning rate must be positive; workers cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    split_json = args.split_json.resolve()
    if not split_json.is_file():
        raise FileNotFoundError(split_json)
    splits = load_split_manifest(str(split_json))
    dataset_root = resolve_dataset_root(split_json)
    resume_path = args.resume.resolve() if args.resume else None
    best_path, last_path, log_path, batch_log_path = validate_output_locations(
        args.output_dir, resume_path,
    )

    train_dataset = PatientPatchDataset(
        dataset_root, splits[args.train_split], roi_size=args.roi_size,
        samples_per_patient=args.samples_per_patient, seed=args.seed, augment=True,
    )
    val_dataset = PatientVolumeDataset(dataset_root, splits[args.val_split])
    assert_dataset_isolation(train_dataset, val_dataset, splits)
    print(
        f"LEAKAGE CHECK PASSED: train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}; test patients are not instantiated."
    )

    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, generator=loader_generator,
    )
    val_loader_kwargs = {
        "batch_size": 1, "shuffle": False, "num_workers": args.val_workers,
        "pin_memory": device.type == "cuda", "persistent_workers": args.val_workers > 0,
    }
    if args.val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    model = MultiEncoderRMDUNet(
        in_modalities=4, base_ch=16, num_stages=4, num_classes=4, rmd_enable=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Cfg.weight_decay)
    criterion = DiceCELoss(num_classes=4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    manifest_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    best_validation_dice = float("-inf")
    start_epoch = 1
    if resume_path is not None:
        start_epoch, best_validation_dice = load_resume_state(
            resume_path, model, optimizer, scaler, manifest_sha256, args, log_path,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint completed epoch {start_epoch - 1}; --epochs must be at least {start_epoch}"
            )
        print(
            f"RESUME CHECK PASSED: starting epoch {start_epoch}; "
            f"best validation Dice={best_validation_dice:.6f}; "
            f"restored learning rate={optimizer.param_groups[0]['lr']}."
        )

    log_mode = "a" if resume_path else "x"
    batch_log_mode = "a" if batch_log_path.exists() else "x"
    with log_path.open(log_mode, newline="", encoding="utf-8") as log_handle, \
            batch_log_path.open(batch_log_mode, newline="", encoding="utf-8") as batch_log_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=LOG_COLUMNS)
        batch_log_writer = csv.DictWriter(batch_log_handle, fieldnames=BATCH_LOG_COLUMNS)
        if not resume_path:
            log_writer.writeheader()
        if batch_log_mode == "x":
            batch_log_writer.writeheader()
        log_handle.flush()
        batch_log_handle.flush()
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            random.seed(args.seed + epoch)
            loader_generator.manual_seed(args.seed + epoch)
            model.train()
            train_loss_total = 0.0
            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_loss = float(loss.item())
                train_loss_total += batch_loss
                batch_log_writer.writerow({
                    "epoch": epoch, "step": step,
                    "global_step": (epoch - 1) * len(train_loader) + step,
                    "batch_loss": batch_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                })
                if step % 25 == 0 or step == len(train_loader):
                    print(f"Epoch {epoch}/{args.epochs} train {step}/{len(train_loader)} loss={loss.item():.6f}")
                    batch_log_handle.flush()
            train_loss = train_loss_total / len(train_loader)

            model.eval()
            validation_loss = 0.0
            dice_totals = {"wt": 0.0, "tc": 0.0, "et": 0.0}
            for index, (volume, target, case_ids) in enumerate(val_loader):
                volume = volume.squeeze(0)
                target = target.squeeze(0)
                case_id = case_ids[0]
                prediction, case_loss = validate_volume(
                    model, volume, target, criterion, device, args.roi_size, overlap=0.5,
                )
                scores = region_dice(prediction, target.numpy())
                validation_loss += case_loss
                for name in dice_totals:
                    dice_totals[name] += scores[name]
                print(f"Epoch {epoch}/{args.epochs} val {index + 1}/{len(val_dataset)} {case_id}")
            val_loss = validation_loss / len(val_dataset)
            val_dice = {name: total / len(val_dataset) for name, total in dice_totals.items()}
            selection_dice = float(np.mean(list(val_dice.values())))
            learning_rate = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_dice_wt": val_dice["wt"], "val_dice_tc": val_dice["tc"],
                "val_dice_et": val_dice["et"], "learning_rate": learning_rate,
            }
            log_writer.writerow(row)
            log_handle.flush()
            checkpoint = {
                "epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_val_dice": max(best_validation_dice, selection_dice),
                "val_dice_mean": selection_dice, "split_manifest_sha256": manifest_sha256,
                "train_split": args.train_split, "val_split": args.val_split,
                "roi_size": args.roi_size, "seed": args.seed,
                "batch_size": args.batch_size,
                "samples_per_patient": args.samples_per_patient,
            }
            save_checkpoint_atomic(last_path, checkpoint)
            if selection_dice > best_validation_dice:
                best_validation_dice = selection_dice
                checkpoint["best_val_dice"] = best_validation_dice
                save_checkpoint_atomic(best_path, checkpoint)
                print(f"Saved new best checkpoint (mean region Dice={best_validation_dice:.6f})")
            elapsed = time.perf_counter() - started
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"WT={val_dice['wt']:.6f} TC={val_dice['tc']:.6f} "
                f"ET={val_dice['et']:.6f} time={elapsed:.1f}s"
            )


if __name__ == "__main__":
    main()
