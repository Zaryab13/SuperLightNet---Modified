#!/usr/bin/env python3
"""Post-train SuperLightNet by self-distilling full-modality predictions into missing-modality inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss, MultiEncoderRMDUNet  # noqa: E402
from scripts.evaluate_patient_split import (  # noqa: E402
    MODEL_MODALITIES,
    case_metrics,
    load_case,
    sliding_window_predict,
)
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset  # noqa: E402
from superlightnet.training import save_checkpoint_atomic  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "self_distill"
RESULT_DIR = PROJECT_ROOT / "results" / "03_self_kd" / "self_distill"
DROP_T1CE_MODALITIES = ("t1", "t2", "flair")
ALL_MODALITIES = tuple(MODEL_MODALITIES)
EPOCH_LOG_COLUMNS = (
    "epoch", "train_loss", "seg_loss", "kd_loss", "feature_loss",
    "val_dice_wt", "val_dice_tc", "val_dice_et", "selection_tc_et",
    "learning_rate", "elapsed_sec",
)
BATCH_LOG_COLUMNS = (
    "epoch", "step", "global_step", "availability_mask", "total_loss",
    "seg_loss", "kd_loss", "feature_loss", "learning_rate",
)


def parse_roi_size(value: str) -> Tuple[int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must contain three integers") from exc
    if len(result) != 3 or any(size <= 0 or size % 16 for size in result):
        raise argparse.ArgumentTypeError("ROI dimensions must be positive multiples of 16")
    return result


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


def load_checkpoint_weights(path: Path) -> Tuple[Dict[str, torch.Tensor], dict]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    return state_dict, checkpoint


def build_model(state_dict: Dict[str, torch.Tensor], device: torch.device) -> MultiEncoderRMDUNet:
    model = MultiEncoderRMDUNet(
        in_modalities=4, base_ch=16, num_stages=4, num_classes=4, rmd_enable=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def state_dicts_identical(left: torch.nn.Module, right: torch.nn.Module) -> Tuple[bool, float]:
    maximum = 0.0
    identical = True
    right_state = right.state_dict()
    for name, tensor in left.state_dict().items():
        other = right_state[name]
        if not torch.equal(tensor, other):
            identical = False
        if tensor.is_floating_point():
            maximum = max(maximum, float((tensor - other).abs().max().item()))
    return identical, maximum


def forward_with_features(model: MultiEncoderRMDUNet, inputs: torch.Tensor,
                          avail_mask: torch.Tensor):
    captured = {}

    def capture_bottleneck_input(_module, args):
        captured["fused_bottleneck"] = args[0]

    def capture_decoder_inputs(_module, args):
        captured["aggregated_skips"] = tuple(args[1])

    handles = (
        model.bottleneck.register_forward_pre_hook(capture_bottleneck_input),
        model.decoder.register_forward_pre_hook(capture_decoder_inputs),
    )
    try:
        logits = model(inputs, avail_mask=avail_mask)
    finally:
        for handle in handles:
            handle.remove()
    if "fused_bottleneck" not in captured or "aggregated_skips" not in captured:
        raise RuntimeError("Feature hooks did not capture fused encoder features")
    return logits, captured


def sample_availability_mask(rng: random.Random, device: torch.device) -> torch.Tensor:
    if rng.random() < 0.6:
        candidates = (0, 2, 3)
        keep_count = rng.randint(1, len(candidates))
        kept = rng.sample(candidates, keep_count)
    else:
        keep_count = rng.randint(1, len(MODEL_MODALITIES) - 1)
        kept = rng.sample(range(len(MODEL_MODALITIES)), keep_count)
    mask = torch.zeros(len(MODEL_MODALITIES), dtype=torch.bool, device=device)
    mask[kept] = True
    if bool(mask.all()) or not bool(mask.any()):
        raise AssertionError("Availability sampling must retain 1-3 modalities")
    return mask


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    per_voxel = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
    return per_voxel.mean() * (temperature ** 2)


def feature_loss(student_features: dict, teacher_features: dict) -> torch.Tensor:
    losses = [F.mse_loss(
        student_features["fused_bottleneck"], teacher_features["fused_bottleneck"],
    )]
    losses.extend(
        F.mse_loss(student, teacher)
        for student, teacher in zip(
            student_features["aggregated_skips"], teacher_features["aggregated_skips"],
        )
    )
    return torch.stack(losses).mean()


def validate_subset(model: MultiEncoderRMDUNet, case_ids: Sequence[str], dataset_root: Path,
                    modalities: Sequence[str], device: torch.device,
                    roi_size: Tuple[int, int, int], overlap: float,
                    label: str) -> Dict[str, float]:
    model.eval()
    totals = {"wt": 0.0, "tc": 0.0, "et": 0.0}
    for index, case_id in enumerate(case_ids, start=1):
        volume, ground_truth, spacing, avail_mask = load_case(
            dataset_root / case_id, modalities,
        )
        prediction = sliding_window_predict(
            model, volume, avail_mask, device, roi_size, overlap,
        )
        metrics = case_metrics(prediction, ground_truth, spacing)
        for region in totals:
            totals[region] += metrics[f"dice_{region}"]
        print(f"{label} {index}/{len(case_ids)} {case_id}", flush=True)
    return {region: total / len(case_ids) for region, total in totals.items()}


def stored_validation_row(log_path: Path, epoch: int) -> Dict[str, float]:
    with log_path.open(newline="", encoding="utf-8") as handle:
        row = next((row for row in csv.DictReader(handle) if int(row["epoch"]) == epoch), None)
    if row is None:
        raise ValueError(f"No stored validation row for checkpoint epoch {epoch}")
    return {
        "wt": float(row["val_dice_wt"]),
        "tc": float(row["val_dice_tc"]),
        "et": float(row["val_dice_et"]),
    }


def checkpoint_payload(epoch: int, student, optimizer, scaler, args,
                       split_sha256: str, best_score: float, best_epoch: int,
                       validation: Dict[str, float] | None) -> dict:
    return {
        "epoch": epoch,
        "model": student.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_drop_t1ce_tc_et": best_score,
        "best_epoch": best_epoch,
        "validation": validation,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "split_manifest_sha256": split_sha256,
        "roi_size": args.roi_size,
        "seed": args.seed,
        "weights": {
            "segmentation": args.w_seg,
            "kd": args.w_kd,
            "feature": args.w_feat,
            "temperature": args.temperature,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/leakage_safe/best_patient_split.pth"))
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--roi_size", type=parse_roi_size, default=(160, 160, 160))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_kd", type=float, default=1.0)
    parser.add_argument("--w_feat", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    if (args.epochs < 1 or args.lr <= 0 or args.batch_size < 1 or args.val_every < 1 or
            args.early_stop_patience < 1 or args.samples_per_patient < 1 or
            args.num_workers < 0 or not 0.0 <= args.overlap < 1.0 or
            min(args.w_seg, args.w_kd, args.w_feat) < 0 or args.temperature <= 0):
        parser.error("Invalid training, validation, or loss parameter")
    if CHECKPOINT_DIR.exists() or RESULT_DIR.exists():
        raise FileExistsError(
            f"Refusing to reuse self-distillation outputs: {CHECKPOINT_DIR} / {RESULT_DIR}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This post-training run requires an explicitly available CUDA device")
    checkpoint_path = args.checkpoint.resolve()
    split_json = args.split_json.resolve()
    if not checkpoint_path.is_file() or not split_json.is_file():
        raise FileNotFoundError("Checkpoint or split manifest is missing")

    set_seed(args.seed)
    splits = load_split_manifest(str(split_json))
    dataset_root = resolve_dataset_root(split_json)
    state_dict, source_checkpoint = load_checkpoint_weights(checkpoint_path)
    args.source_checkpoint_epoch = int(source_checkpoint["epoch"])
    teacher = build_model(state_dict, device)
    student = build_model(state_dict, device)
    teacher.eval()
    teacher.rmd_enable = False
    teacher.requires_grad_(False)
    student.rmd_enable = False

    identical, maximum_difference = state_dicts_identical(teacher, student)
    print(f"SANITY student_teacher_identical={identical} max_parameter_difference={maximum_difference}")
    if not identical or maximum_difference != 0.0:
        raise RuntimeError("Student and teacher are not identical at step zero")

    stored = stored_validation_row(
        PROJECT_ROOT / "results" / "01_base_model" / "leakage_safe" / "leakage_safe_training_log.csv",
        args.source_checkpoint_epoch,
    )
    teacher_validation = validate_subset(
        teacher, splits["val"], dataset_root, ALL_MODALITIES, device,
        args.roi_size, args.overlap, "SANITY teacher_full",
    )
    print(f"SANITY teacher_full_stored={stored} reproduced={teacher_validation}")
    for region in stored:
        if abs(teacher_validation[region] - stored[region]) > 0.001:
            raise RuntimeError(
                f"Teacher full-modality validation mismatch for {region}: "
                f"stored={stored[region]}, reproduced={teacher_validation[region]}"
            )

    epoch_zero = validate_subset(
        student, splits["val"], dataset_root, DROP_T1CE_MODALITIES, device,
        args.roi_size, args.overlap, "SANITY student_epoch0_drop_t1ce",
    )
    best_score = (epoch_zero["tc"] + epoch_zero["et"]) / 2.0
    best_epoch = 0
    print(
        f"SANITY epoch0_drop_t1ce WT={epoch_zero['wt']:.12f} "
        f"TC={epoch_zero['tc']:.12f} ET={epoch_zero['et']:.12f} "
        f"selection={best_score:.12f}"
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    epoch_log_path = RESULT_DIR / "self_distill_training_log.csv"
    batch_log_path = RESULT_DIR / "self_distill_training_batch_log.csv"
    sanity_path = RESULT_DIR / "sanity.json"
    sanity_path.write_text(json.dumps({
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "student_teacher_identical": identical,
        "max_parameter_difference": maximum_difference,
        "teacher_full_stored": stored,
        "teacher_full_reproduced": teacher_validation,
        "student_epoch0_drop_t1ce": epoch_zero,
        "epoch0_selection_tc_et": best_score,
    }, indent=2) + "\n", encoding="utf-8")

    train_dataset = PatientPatchDataset(
        dataset_root, splits["train"], roi_size=args.roi_size,
        samples_per_patient=args.samples_per_patient, seed=args.seed, augment=True,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=Cfg.weight_decay)
    criterion = DiceCELoss(num_classes=4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    split_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    mask_rng = random.Random(args.seed)
    all_available = torch.ones(4, dtype=torch.bool, device=device)

    initial_payload = checkpoint_payload(
        0, student, optimizer, scaler, args, split_sha256,
        best_score, best_epoch, epoch_zero,
    )
    save_checkpoint_atomic(CHECKPOINT_DIR / "best_self_distill.pth", initial_payload)

    with epoch_log_path.open("x", newline="", encoding="utf-8") as epoch_handle, \
            batch_log_path.open("x", newline="", encoding="utf-8") as batch_handle:
        epoch_writer = csv.DictWriter(epoch_handle, fieldnames=EPOCH_LOG_COLUMNS)
        batch_writer = csv.DictWriter(batch_handle, fieldnames=BATCH_LOG_COLUMNS)
        epoch_writer.writeheader()
        batch_writer.writeheader()
        epoch_handle.flush()
        batch_handle.flush()

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            loader_generator.manual_seed(args.seed + epoch)
            student.train()
            teacher.eval()
            sums = {"total": 0.0, "seg": 0.0, "kd": 0.0, "feature": 0.0}

            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                availability = sample_availability_mask(mask_rng, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True,
                ):
                    teacher_logits, teacher_features = forward_with_features(
                        teacher, images, all_available,
                    )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    student_logits, student_features = forward_with_features(
                        student, images, availability,
                    )
                    segmentation = criterion(student_logits, targets)
                    distillation = kd_loss(student_logits, teacher_logits, args.temperature)
                    features = feature_loss(student_features, teacher_features)
                    loss = (
                        args.w_seg * segmentation + args.w_kd * distillation +
                        args.w_feat * features
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                values = {
                    "total": float(loss.item()),
                    "seg": float(segmentation.item()),
                    "kd": float(distillation.item()),
                    "feature": float(features.item()),
                }
                for name in sums:
                    sums[name] += values[name]
                batch_writer.writerow({
                    "epoch": epoch,
                    "step": step,
                    "global_step": (epoch - 1) * len(train_loader) + step,
                    "availability_mask": "".join("1" if value else "0" for value in availability.tolist()),
                    "total_loss": values["total"],
                    "seg_loss": values["seg"],
                    "kd_loss": values["kd"],
                    "feature_loss": values["feature"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })
                if step % 25 == 0 or step == len(train_loader):
                    print(
                        f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                        f"loss={values['total']:.6f} seg={values['seg']:.6f} "
                        f"kd={values['kd']:.6f} feat={values['feature']:.6f}",
                        flush=True,
                    )
                    batch_handle.flush()

            averages = {name: total / len(train_loader) for name, total in sums.items()}
            validation = None
            selection = None
            if epoch % args.val_every == 0 or epoch == args.epochs:
                validation = validate_subset(
                    student, splits["val"], dataset_root, DROP_T1CE_MODALITIES,
                    device, args.roi_size, args.overlap, f"Epoch {epoch} val_drop_t1ce",
                )
                selection = (validation["tc"] + validation["et"]) / 2.0
                if selection > best_score:
                    best_score = selection
                    best_epoch = epoch
                    save_checkpoint_atomic(
                        CHECKPOINT_DIR / "best_self_distill.pth",
                        checkpoint_payload(
                            epoch, student, optimizer, scaler, args, split_sha256,
                            best_score, best_epoch, validation,
                        ),
                    )
                    print(
                        f"NEW BEST epoch={epoch} TC={validation['tc']:.12f} "
                        f"ET={validation['et']:.12f} selection={selection:.12f}",
                        flush=True,
                    )

            elapsed = time.perf_counter() - started
            epoch_writer.writerow({
                "epoch": epoch,
                "train_loss": averages["total"],
                "seg_loss": averages["seg"],
                "kd_loss": averages["kd"],
                "feature_loss": averages["feature"],
                "val_dice_wt": "" if validation is None else validation["wt"],
                "val_dice_tc": "" if validation is None else validation["tc"],
                "val_dice_et": "" if validation is None else validation["et"],
                "selection_tc_et": "" if selection is None else selection,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
            })
            epoch_handle.flush()
            save_checkpoint_atomic(
                CHECKPOINT_DIR / "last_self_distill.pth",
                checkpoint_payload(
                    epoch, student, optimizer, scaler, args, split_sha256,
                    best_score, best_epoch, validation,
                ),
            )
            val_tc_text = "" if validation is None else f"{validation['tc']:.6f}"
            val_et_text = "" if validation is None else f"{validation['et']:.6f}"
            print(
                f"Epoch {epoch}: loss={averages['total']:.6f} "
                f"seg={averages['seg']:.6f} kd={averages['kd']:.6f} "
                f"feat={averages['feature']:.6f} "
                f"val_TC={val_tc_text} val_ET={val_et_text} "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            if epoch - best_epoch >= args.early_stop_patience:
                print(
                    f"EARLY STOP epoch={epoch}: no validation improvement for "
                    f"{args.early_stop_patience} epochs",
                    flush=True,
                )
                break

    print(
        f"COMPLETE epoch0_TC={epoch_zero['tc']:.12f} epoch0_ET={epoch_zero['et']:.12f} "
        f"best_epoch={best_epoch} best_selection={best_score:.12f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
