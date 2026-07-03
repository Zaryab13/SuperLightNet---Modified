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


def validate_output_locations(output_dir: Path):
    expected = (PROJECT_ROOT / "checkpoints" / "leakage_safe").resolve()
    output_dir = output_dir.resolve()
    if output_dir != expected:
        raise ValueError(f"--output_dir must resolve to {expected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_patient_split.pth"
    last_path = output_dir / "last_patient_split.pth"
    log_path = PROJECT_ROOT / "results" / "leakage_safe" / "training_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (best_path, last_path, log_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing training artifact: {existing[0]}")
    return best_path, last_path, log_path


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
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch_size, and learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    split_json = args.split_json.resolve()
    if not split_json.is_file():
        raise FileNotFoundError(split_json)
    splits = load_split_manifest(str(split_json))
    dataset_root = resolve_dataset_root(split_json)
    best_path, last_path, log_path = validate_output_locations(args.output_dir)

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
    model = MultiEncoderRMDUNet(
        in_modalities=4, base_ch=16, num_stages=4, num_classes=4, rmd_enable=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Cfg.weight_decay)
    criterion = DiceCELoss(num_classes=4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    manifest_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    best_validation_dice = float("-inf")

    with log_path.open("x", newline="", encoding="utf-8") as log_handle:
        log_writer = csv.DictWriter(log_handle, fieldnames=LOG_COLUMNS)
        log_writer.writeheader()
        log_handle.flush()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            random.seed(args.seed + epoch)
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
                train_loss_total += float(loss.item())
                if step % 25 == 0 or step == len(train_loader):
                    print(f"Epoch {epoch}/{args.epochs} train {step}/{len(train_loader)} loss={loss.item():.6f}")
            train_loss = train_loss_total / len(train_loader)

            model.eval()
            validation_loss = 0.0
            dice_totals = {"wt": 0.0, "tc": 0.0, "et": 0.0}
            for index in range(len(val_dataset)):
                volume, target, case_id = val_dataset[index]
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
