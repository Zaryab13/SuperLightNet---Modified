#!/usr/bin/env python3
"""Self-distillation v2 with full-modality anchors, frozen BN statistics, and constrained selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss  # noqa: E402
from scripts import swin_kd_clean_B_common as common  # noqa: E402
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset  # noqa: E402
from superlightnet.training import save_checkpoint_atomic  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "self_distill_v2"
RESULT_DIR = PROJECT_ROOT / "results" / "self_distill_v2"
ALL_MODALITIES = ("t1", "t1ce", "t2", "flair")
DROP_T1CE_MODALITIES = ("t1", "t2", "flair")

EPOCH_COLUMNS = (
    "epoch", "train_loss", "seg_loss", "kd_loss", "feature_loss",
    "all_dice_wt", "all_dice_tc", "all_dice_et", "all_mean",
    "drop_t1ce_dice_wt", "drop_t1ce_dice_tc", "drop_t1ce_dice_et",
    "drop_t1ce_tc_et", "full_constraint_met", "learning_rate", "elapsed_sec",
)
BATCH_COLUMNS = (
    "epoch", "step", "global_step", "mask_category", "availability_mask",
    "total_loss", "seg_loss", "kd_loss", "feature_loss", "learning_rate",
)
TRADEOFF_COLUMNS = (
    "epoch", "all_dice_wt", "all_dice_tc", "all_dice_et", "all_mean",
    "drop_t1ce_dice_wt", "drop_t1ce_dice_tc", "drop_t1ce_dice_et",
    "drop_t1ce_tc_et", "constraint_threshold", "constraint_met", "selected_best",
)


def freeze_batchnorm_running_stats(model: nn.Module) -> Tuple[int, bool, bool]:
    model.train()
    batchnorms = [module for module in model.modules() if isinstance(module, nn.BatchNorm3d)]
    for module in batchnorms:
        module.eval()
    all_bn_eval = all(not module.training for module in batchnorms)
    non_bn_training = all(
        module.training for module in model.modules()
        if module is not model and not isinstance(module, nn.BatchNorm3d)
    )
    return len(batchnorms), all_bn_eval, non_bn_training


def sample_v2_mask(rng: random.Random, device: torch.device):
    draw = rng.random()
    if draw < 0.45:
        category = "full_anchor"
        kept = (0, 1, 2, 3)
    elif draw < 0.80:
        category = "drop_t1ce"
        kept = (0, 2, 3)
    else:
        category = "random_subset"
        keep_count = rng.randint(1, 3)
        kept = tuple(rng.sample(range(4), keep_count))
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    mask[list(kept)] = True
    if not bool(mask.any()):
        raise AssertionError("At least one modality must be retained")
    return mask, category


def mean_regions(scores: Dict[str, float]) -> float:
    return (scores["wt"] + scores["tc"] + scores["et"]) / 3.0


def mean_tc_et(scores: Dict[str, float]) -> float:
    return (scores["tc"] + scores["et"]) / 2.0


def checkpoint_payload(epoch, student, optimizer, scaler, args, split_sha256,
                       teacher_baseline, constraint_threshold, all_scores,
                       drop_scores, best_epoch, best_score):
    return {
        "epoch": epoch,
        "model": student.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_epoch": best_epoch,
        "best_drop_t1ce_tc_et": best_score,
        "all_validation": all_scores,
        "drop_t1ce_validation": drop_scores,
        "teacher_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
        "full_constraint_met": (
            None if all_scores is None else mean_regions(all_scores) >= constraint_threshold
        ),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "split_manifest_sha256": split_sha256,
        "roi_size": args.roi_size,
        "seed": args.seed,
        "mask_probabilities": {
            "full_anchor": 0.45,
            "drop_t1ce": 0.35,
            "random_subset": 0.20,
        },
        "batchnorm_running_stats_frozen": True,
        "weights": {
            "segmentation": args.w_seg,
            "kd": args.w_kd,
            "feature": args.w_feat,
            "temperature": args.temperature,
        },
    }


def validate_both(student, splits, dataset_root, device, args, label):
    all_scores = common.validate_subset(
        student, splits["val"], dataset_root, ALL_MODALITIES,
        device, args.roi_size, args.overlap, f"{label} all",
    )
    drop_scores = common.validate_subset(
        student, splits["val"], dataset_root, DROP_T1CE_MODALITIES,
        device, args.roi_size, args.overlap, f"{label} drop_t1ce",
    )
    return all_scores, drop_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/leakage_safe/best_patient_split.pth"))
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--roi_size", type=common.parse_roi_size, default=(160, 160, 160))
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
            f"Refusing to reuse v2 outputs: {CHECKPOINT_DIR} / {RESULT_DIR}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V2 post-training requires a free CUDA device")
    checkpoint_path = args.checkpoint.resolve()
    split_json = args.split_json.resolve()
    if not checkpoint_path.is_file() or not split_json.is_file():
        raise FileNotFoundError("Original checkpoint or split manifest is missing")
    expected_source = (PROJECT_ROOT / "checkpoints" / "leakage_safe" /
                       "best_patient_split.pth").resolve()
    if checkpoint_path != expected_source:
        raise ValueError(f"V2 must initialize from the original teacher: {expected_source}")

    common.set_seed(args.seed)
    splits = load_split_manifest(str(split_json))
    dataset_root = common.resolve_dataset_root(split_json)
    state_dict, source_checkpoint = common.load_checkpoint_weights(checkpoint_path)
    args.source_checkpoint_epoch = int(source_checkpoint["epoch"])
    teacher = common.build_model(state_dict, device)
    student = common.build_model(state_dict, device)
    teacher.eval()
    teacher.rmd_enable = False
    teacher.requires_grad_(False)
    student.rmd_enable = False

    identical, maximum_difference = common.state_dicts_identical(teacher, student)
    print(
        f"SANITY student_teacher_identical={identical} "
        f"max_parameter_difference={maximum_difference}", flush=True,
    )
    if not identical or maximum_difference != 0.0:
        raise RuntimeError("Student and teacher are not identical at epoch zero")

    stored = common.stored_validation_row(
        PROJECT_ROOT / "results" / "leakage_safe" / "training_log.csv",
        args.source_checkpoint_epoch,
    )
    epoch_zero_all, epoch_zero_drop = validate_both(
        student, splits, dataset_root, device, args, "SANITY epoch0",
    )
    print(f"SANITY epoch0_all stored={stored} reproduced={epoch_zero_all}", flush=True)
    for region in stored:
        if abs(epoch_zero_all[region] - stored[region]) > 0.005:
            raise RuntimeError(
                f"Epoch-zero all-modality mismatch for {region}: "
                f"stored={stored[region]}, reproduced={epoch_zero_all[region]}"
            )

    teacher_baseline = mean_regions(epoch_zero_all)
    constraint_threshold = 0.97 * teacher_baseline
    print(
        f"SANITY epoch0_all WT={epoch_zero_all['wt']:.12f} "
        f"TC={epoch_zero_all['tc']:.12f} ET={epoch_zero_all['et']:.12f} "
        f"mean={teacher_baseline:.12f}", flush=True,
    )
    print(
        f"SANITY epoch0_drop_t1ce WT={epoch_zero_drop['wt']:.12f} "
        f"TC={epoch_zero_drop['tc']:.12f} ET={epoch_zero_drop['et']:.12f} "
        f"selection={mean_tc_et(epoch_zero_drop):.12f}", flush=True,
    )
    print(
        f"SANITY full_constraint_threshold={constraint_threshold:.12f}", flush=True,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    split_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    (RESULT_DIR / "sanity.json").write_text(json.dumps({
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "student_teacher_identical": identical,
        "max_parameter_difference": maximum_difference,
        "teacher_full_stored": stored,
        "epoch0_all": epoch_zero_all,
        "epoch0_drop_t1ce": epoch_zero_drop,
        "teacher_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
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
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=Cfg.weight_decay,
    )
    criterion = DiceCELoss(num_classes=4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    all_available = torch.ones(4, dtype=torch.bool, device=device)
    mask_rng = random.Random(args.seed)
    best_epoch = None
    best_score = float("-inf")
    validation_history = []

    epoch_path = RESULT_DIR / "training_log.csv"
    batch_path = RESULT_DIR / "training_batch_log.csv"
    tradeoff_path = RESULT_DIR / "validation_tradeoff.csv"
    with epoch_path.open("x", newline="", encoding="utf-8") as epoch_handle, \
            batch_path.open("x", newline="", encoding="utf-8") as batch_handle, \
            tradeoff_path.open("x", newline="", encoding="utf-8") as tradeoff_handle:
        epoch_writer = csv.DictWriter(epoch_handle, fieldnames=EPOCH_COLUMNS)
        batch_writer = csv.DictWriter(batch_handle, fieldnames=BATCH_COLUMNS)
        tradeoff_writer = csv.DictWriter(tradeoff_handle, fieldnames=TRADEOFF_COLUMNS)
        epoch_writer.writeheader()
        batch_writer.writeheader()
        tradeoff_writer.writeheader()
        tradeoff_writer.writerow({
            "epoch": 0,
            "all_dice_wt": epoch_zero_all["wt"],
            "all_dice_tc": epoch_zero_all["tc"],
            "all_dice_et": epoch_zero_all["et"],
            "all_mean": teacher_baseline,
            "drop_t1ce_dice_wt": epoch_zero_drop["wt"],
            "drop_t1ce_dice_tc": epoch_zero_drop["tc"],
            "drop_t1ce_dice_et": epoch_zero_drop["et"],
            "drop_t1ce_tc_et": mean_tc_et(epoch_zero_drop),
            "constraint_threshold": constraint_threshold,
            "constraint_met": True,
            "selected_best": False,
        })
        epoch_handle.flush()
        batch_handle.flush()
        tradeoff_handle.flush()

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            loader_generator.manual_seed(args.seed + epoch)
            bn_count, all_bn_eval, non_bn_training = freeze_batchnorm_running_stats(student)
            teacher.eval()
            print(
                f"BN_FREEZE epoch={epoch} modules={bn_count} all_bn_eval={all_bn_eval} "
                f"non_bn_training={non_bn_training} student_training={student.training}",
                flush=True,
            )
            if not all_bn_eval or not non_bn_training or not student.training:
                raise RuntimeError("BatchNorm freeze state is incorrect")

            sums = {"total": 0.0, "seg": 0.0, "kd": 0.0, "feature": 0.0}
            category_counts = {"full_anchor": 0, "drop_t1ce": 0, "random_subset": 0}
            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                availability, category = sample_v2_mask(mask_rng, device)
                category_counts[category] += 1
                optimizer.zero_grad(set_to_none=True)

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True,
                ):
                    teacher_logits, teacher_features = common.forward_with_features(
                        teacher, images, all_available,
                    )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    student_logits, student_features = common.forward_with_features(
                        student, images, availability,
                    )
                    segmentation = criterion(student_logits, targets)
                    distillation = common.kd_loss(
                        student_logits, teacher_logits, args.temperature,
                    )
                    features = common.feature_loss(student_features, teacher_features)
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
                    "mask_category": category,
                    "availability_mask": "".join(
                        "1" if value else "0" for value in availability.tolist()
                    ),
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
                        f"kd={values['kd']:.6f} feat={values['feature']:.6f} "
                        f"mask={category}", flush=True,
                    )
                    batch_handle.flush()

            averages = {name: value / len(train_loader) for name, value in sums.items()}
            all_scores = None
            drop_scores = None
            all_mean = None
            drop_selection = None
            constraint_met = None
            selected_best = False
            if epoch % args.val_every == 0 or epoch == args.epochs:
                all_scores, drop_scores = validate_both(
                    student, splits, dataset_root, device, args, f"Epoch {epoch} val",
                )
                all_mean = mean_regions(all_scores)
                drop_selection = mean_tc_et(drop_scores)
                constraint_met = all_mean >= constraint_threshold
                if constraint_met and drop_selection > best_score:
                    best_score = drop_selection
                    best_epoch = epoch
                    selected_best = True
                    save_checkpoint_atomic(
                        CHECKPOINT_DIR / "best_self_distill_v2.pth",
                        checkpoint_payload(
                            epoch, student, optimizer, scaler, args, split_sha256,
                            teacher_baseline, constraint_threshold, all_scores,
                            drop_scores, best_epoch, best_score,
                        ),
                    )
                tradeoff = {
                    "epoch": epoch,
                    "all_dice_wt": all_scores["wt"],
                    "all_dice_tc": all_scores["tc"],
                    "all_dice_et": all_scores["et"],
                    "all_mean": all_mean,
                    "drop_t1ce_dice_wt": drop_scores["wt"],
                    "drop_t1ce_dice_tc": drop_scores["tc"],
                    "drop_t1ce_dice_et": drop_scores["et"],
                    "drop_t1ce_tc_et": drop_selection,
                    "constraint_threshold": constraint_threshold,
                    "constraint_met": constraint_met,
                    "selected_best": selected_best,
                }
                validation_history.append(tradeoff)
                tradeoff_writer.writerow(tradeoff)
                tradeoff_handle.flush()
                print(
                    f"VALIDATION epoch={epoch} all_WT={all_scores['wt']:.12f} "
                    f"all_TC={all_scores['tc']:.12f} all_ET={all_scores['et']:.12f} "
                    f"all_mean={all_mean:.12f} drop_WT={drop_scores['wt']:.12f} "
                    f"drop_TC={drop_scores['tc']:.12f} drop_ET={drop_scores['et']:.12f} "
                    f"drop_selection={drop_selection:.12f} "
                    f"constraint_met={constraint_met} selected_best={selected_best}",
                    flush=True,
                )

            elapsed = time.perf_counter() - started
            epoch_writer.writerow({
                "epoch": epoch,
                "train_loss": averages["total"],
                "seg_loss": averages["seg"],
                "kd_loss": averages["kd"],
                "feature_loss": averages["feature"],
                "all_dice_wt": "" if all_scores is None else all_scores["wt"],
                "all_dice_tc": "" if all_scores is None else all_scores["tc"],
                "all_dice_et": "" if all_scores is None else all_scores["et"],
                "all_mean": "" if all_mean is None else all_mean,
                "drop_t1ce_dice_wt": "" if drop_scores is None else drop_scores["wt"],
                "drop_t1ce_dice_tc": "" if drop_scores is None else drop_scores["tc"],
                "drop_t1ce_dice_et": "" if drop_scores is None else drop_scores["et"],
                "drop_t1ce_tc_et": "" if drop_selection is None else drop_selection,
                "full_constraint_met": "" if constraint_met is None else constraint_met,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
            })
            epoch_handle.flush()
            save_checkpoint_atomic(
                CHECKPOINT_DIR / "last_self_distill_v2.pth",
                checkpoint_payload(
                    epoch, student, optimizer, scaler, args, split_sha256,
                    teacher_baseline, constraint_threshold, all_scores,
                    drop_scores, best_epoch, best_score,
                ),
            )
            print(
                f"Epoch {epoch}: loss={averages['total']:.6f} "
                f"seg={averages['seg']:.6f} kd={averages['kd']:.6f} "
                f"feat={averages['feature']:.6f} masks={category_counts} "
                f"time={elapsed:.1f}s", flush=True,
            )
            if best_epoch is not None and epoch - best_epoch >= args.early_stop_patience:
                print(
                    f"EARLY STOP epoch={epoch}: no eligible selection improvement for "
                    f"{args.early_stop_patience} epochs", flush=True,
                )
                break

    selection_status = {
        "teacher_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
        "best_epoch": best_epoch,
        "best_drop_t1ce_tc_et": None if best_epoch is None else best_score,
        "constraint_ever_met_after_training": any(
            row["constraint_met"] for row in validation_history
        ),
        "validation_tradeoff": validation_history,
    }
    (RESULT_DIR / "selection_status.json").write_text(
        json.dumps(selection_status, indent=2) + "\n", encoding="utf-8",
    )
    if best_epoch is None:
        print("NO ELIGIBLE CHECKPOINT: full-modality constraint was never met", flush=True)
        for row in validation_history:
            print(f"TRADEOFF {row}", flush=True)
    else:
        selected = next(row for row in validation_history if row["epoch"] == best_epoch)
        print(
            f"COMPLETE selected_epoch={best_epoch} "
            f"all_WT={selected['all_dice_wt']:.12f} "
            f"all_TC={selected['all_dice_tc']:.12f} "
            f"all_ET={selected['all_dice_et']:.12f} "
            f"all_mean={selected['all_mean']:.12f} "
            f"drop_WT={selected['drop_t1ce_dice_wt']:.12f} "
            f"drop_TC={selected['drop_t1ce_dice_tc']:.12f} "
            f"drop_ET={selected['drop_t1ce_dice_et']:.12f} "
            f"drop_selection={selected['drop_t1ce_tc_et']:.12f}", flush=True,
        )


if __name__ == "__main__":
    main()
