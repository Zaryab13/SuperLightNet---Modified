#!/usr/bin/env python3
"""V4 self-distillation with local enhancing-tumor-biased patch sampling."""

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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss  # noqa: E402
from scripts import self_distill_posttrain as common  # noqa: E402
from scripts import self_distill_posttrain_v2 as v2  # noqa: E402
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset, extract_patch  # noqa: E402
from superlightnet.training import save_checkpoint_atomic  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "self_distill_v5"
RESULT_DIR = PROJECT_ROOT / "results" / "03_self_kd" / "self_distill_v5"
ALL_MODALITIES = ("t1", "t1ce", "t2", "flair")
DROP_T1CE_MODALITIES = ("t1", "t2", "flair")

EPOCH_COLUMNS = v2.EPOCH_COLUMNS + ("successful_steps", "skipped_nonfinite_steps")
BATCH_COLUMNS = (
    "epoch", "step", "global_step", "mask_category", "availability_mask",
    "total_loss", "seg_loss", "kd_loss", "feature_loss", "grad_norm",
    "finite", "skipped", "learning_rate",
)
TRADEOFF_COLUMNS = v2.TRADEOFF_COLUMNS

SAMPLING_ET = 0
SAMPLING_TUMOR = 1
SAMPLING_RANDOM = 2
SAMPLING_ET_FALLBACK = 3


class EnhancingBiasedPatientPatchDataset(PatientPatchDataset):
    """V5-local sampler; target index 3 is enhancing tumor after BraTS remapping."""

    def __init__(self, *args, enhancing_sampling_probability=0.30, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhancing_sampling_probability = float(enhancing_sampling_probability)
        if (self.enhancing_sampling_probability < 0.0 or
                self.enhancing_sampling_probability + self.tumor_sampling_probability > 1.0):
            raise ValueError("ET and any-tumor sampling probabilities must sum to at most 1")

    def __getitem__(self, sample_index: int):
        patient_index = sample_index // self.samples_per_patient
        sample_number = sample_index % self.samples_per_patient
        images, target = self._load_case(patient_index)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + patient_index * 101 + sample_number
        )
        enhancing_voxels = np.argwhere(target == 3)
        tumor_voxels = np.argwhere(target > 0)
        draw = rng.random()
        if draw < self.enhancing_sampling_probability:
            if enhancing_voxels.size:
                center = enhancing_voxels[rng.integers(len(enhancing_voxels))]
                sampling_category = SAMPLING_ET
            elif tumor_voxels.size:
                center = tumor_voxels[rng.integers(len(tumor_voxels))]
                sampling_category = SAMPLING_ET_FALLBACK
            else:
                center = [rng.integers(size) for size in target.shape]
                sampling_category = SAMPLING_RANDOM
        elif draw < (self.enhancing_sampling_probability +
                     self.tumor_sampling_probability):
            if tumor_voxels.size:
                center = tumor_voxels[rng.integers(len(tumor_voxels))]
                sampling_category = SAMPLING_TUMOR
            else:
                center = [rng.integers(size) for size in target.shape]
                sampling_category = SAMPLING_RANDOM
        else:
            center = [rng.integers(size) for size in target.shape]
            sampling_category = SAMPLING_RANDOM

        center_label = int(target[tuple(center)])
        image_patch = extract_patch(images, self.roi_size, center)
        target_patch = extract_patch(target, self.roi_size, center)
        if self.augment:
            for spatial_axis in range(3):
                if rng.random() < 0.5:
                    image_patch = np.flip(image_patch, axis=spatial_axis + 1)
                    target_patch = np.flip(target_patch, axis=spatial_axis)
        sampling_metadata = torch.tensor(
            [center_label, sampling_category], dtype=torch.int64,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(image_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(target_patch)).long(),
            sampling_metadata,
        )


def freeze_batchnorm_fully(model: nn.Module):
    model.train()
    batchnorms = [module for module in model.modules() if isinstance(module, nn.BatchNorm3d)]
    frozen_affine_parameters = 0
    for module in batchnorms:
        module.eval()
        if module.weight is not None:
            module.weight.requires_grad_(False)
            frozen_affine_parameters += module.weight.numel()
        if module.bias is not None:
            module.bias.requires_grad_(False)
            frozen_affine_parameters += module.bias.numel()
    all_bn_eval = all(not module.training for module in batchnorms)
    all_bn_affine_frozen = all(
        (module.weight is None or not module.weight.requires_grad) and
        (module.bias is None or not module.bias.requires_grad)
        for module in batchnorms
    )
    return len(batchnorms), frozen_affine_parameters, all_bn_eval, all_bn_affine_frozen


def kd_loss_float32(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    temperature: float) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.float() / temperature, dim=1)
    return F.kl_div(
        student_log_prob, teacher_prob, reduction="batchmean",
    ) * (temperature * temperature)


def feature_loss_float32(student_features: dict, teacher_features: dict) -> torch.Tensor:
    matched = [(
        student_features["fused_bottleneck"],
        teacher_features["fused_bottleneck"],
    )]
    matched.extend(zip(
        student_features["aggregated_skips"],
        teacher_features["aggregated_skips"],
    ))
    losses = [
        F.smooth_l1_loss(
            student.float(), teacher.float(), beta=1.0, reduction="mean",
        )
        for student, teacher in matched
    ]
    return torch.stack(losses).mean()


def finite_value(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def sample_v4_mask(rng: random.Random, device: torch.device):
    """Sample 30% full, 50% drop-T1ce, and 20% random non-empty masks."""
    draw = rng.random()
    if draw < 0.30:
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


def payload(epoch, student, optimizer, args, split_sha256, teacher_baseline,
            constraint_threshold, all_scores, drop_scores, best_epoch, best_score):
    return {
        "epoch": epoch,
        "model": student.state_dict(),
        "opt": optimizer.state_dict(),
        "best_epoch": best_epoch,
        "best_drop_t1ce_tc_et": best_score,
        "all_validation": all_scores,
        "drop_t1ce_validation": drop_scores,
        "teacher_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
        "full_constraint_met": (
            None if all_scores is None else v2.mean_regions(all_scores) >= constraint_threshold
        ),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "split_manifest_sha256": split_sha256,
        "roi_size": args.roi_size,
        "seed": args.seed,
        "autocast_dtype": "bfloat16",
        "grad_scaler": False,
        "max_grad_norm": 1.0,
        "batchnorm_running_stats_frozen": True,
        "batchnorm_affine_frozen": True,
        "mask_probabilities": {
            "full_anchor": 0.30,
            "drop_t1ce": 0.50,
            "random_subset": 0.20,
        },
        "lr_schedule": {
            "name": "CosineAnnealingLR",
            "initial_lr": args.lr,
            "minimum_lr": args.min_lr,
            "t_max_epochs": args.epochs,
        },
        "weights": {
            "segmentation": args.w_seg,
            "kd": args.w_kd,
            "feature": args.w_feat,
            "temperature": args.temperature,
        },
        "patch_sampling": {
            "enhancing_target_index": 3,
            "enhancing_sampling_probability": args.enhancing_sampling_probability,
            "tumor_sampling_probability": args.tumor_sampling_probability,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/leakage_safe/best_patient_split.pth"))
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--roi_size", type=common.parse_roi_size, default=(160, 160, 160))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_kd", type=float, default=1.0)
    parser.add_argument("--w_feat", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--enhancing_sampling_probability", type=float, default=0.30)
    parser.add_argument("--tumor_sampling_probability", type=float, default=0.50)
    args = parser.parse_args()

    if (args.epochs < 1 or args.lr <= 0 or args.min_lr < 0 or args.min_lr >= args.lr or
            args.batch_size < 1 or args.val_every < 1 or
            args.early_stop_patience < 1 or args.samples_per_patient < 1 or
            args.num_workers < 0 or not 0.0 <= args.overlap < 1.0 or
            args.enhancing_sampling_probability < 0.0 or
            args.tumor_sampling_probability < 0.0 or
            args.enhancing_sampling_probability + args.tumor_sampling_probability > 1.0 or
            min(args.w_seg, args.w_kd, args.w_feat) < 0 or args.temperature <= 0):
        parser.error("Invalid training, validation, or loss parameter")
    if CHECKPOINT_DIR.exists() or RESULT_DIR.exists():
        raise FileExistsError(
            f"Refusing to reuse v5 outputs: {CHECKPOINT_DIR} / {RESULT_DIR}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V5 post-training requires a free CUDA device")
    checkpoint_path = args.checkpoint.resolve()
    split_json = args.split_json.resolve()
    expected_source = (PROJECT_ROOT / "checkpoints" / "leakage_safe" /
                       "best_patient_split.pth").resolve()
    if checkpoint_path != expected_source or not checkpoint_path.is_file():
        raise ValueError(f"V5 must initialize from the original teacher: {expected_source}")
    if not split_json.is_file():
        raise FileNotFoundError(split_json)

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
        PROJECT_ROOT / "results" / "01_base_model" / "leakage_safe" / "leakage_safe_training_log.csv",
        args.source_checkpoint_epoch,
    )
    epoch_zero_all, epoch_zero_drop = v2.validate_both(
        student, splits, dataset_root, device, args, "SANITY epoch0",
    )
    print(f"SANITY epoch0_all stored={stored} reproduced={epoch_zero_all}", flush=True)
    for region in stored:
        if abs(epoch_zero_all[region] - stored[region]) > 0.005:
            raise RuntimeError(
                f"Epoch-zero all-modality mismatch for {region}: "
                f"stored={stored[region]}, reproduced={epoch_zero_all[region]}"
            )
    teacher_baseline = v2.mean_regions(epoch_zero_all)
    constraint_threshold = 0.97 * teacher_baseline
    print(
        f"SANITY epoch0_all WT={epoch_zero_all['wt']:.12f} "
        f"TC={epoch_zero_all['tc']:.12f} ET={epoch_zero_all['et']:.12f} "
        f"mean={teacher_baseline:.12f}", flush=True,
    )
    print(
        f"SANITY epoch0_drop_t1ce WT={epoch_zero_drop['wt']:.12f} "
        f"TC={epoch_zero_drop['tc']:.12f} ET={epoch_zero_drop['et']:.12f} "
        f"selection={v2.mean_tc_et(epoch_zero_drop):.12f}", flush=True,
    )
    print(f"SANITY full_constraint_threshold={constraint_threshold:.12f}", flush=True)

    bn_count, frozen_bn_parameters, all_bn_eval, all_bn_affine_frozen = \
        freeze_batchnorm_fully(student)
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=Cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )
    optimizer_parameter_count = sum(
        parameter.numel() for group in optimizer.param_groups for parameter in group["params"]
    )
    print(
        f"BN_FREEZE modules={bn_count} frozen_affine_parameters={frozen_bn_parameters} "
        f"all_bn_eval={all_bn_eval} all_bn_affine_frozen={all_bn_affine_frozen} "
        f"trainable_parameters={trainable_count} "
        f"optimizer_parameters={optimizer_parameter_count}", flush=True,
    )
    if (not all_bn_eval or not all_bn_affine_frozen or not trainable_parameters or
            trainable_count != optimizer_parameter_count):
        raise RuntimeError("BatchNorm or optimizer parameter freeze check failed")

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
        "batchnorm_modules_frozen": bn_count,
        "batchnorm_affine_parameters_frozen": frozen_bn_parameters,
        "trainable_optimizer_parameters": optimizer_parameter_count,
    }, indent=2) + "\n", encoding="utf-8")

    train_dataset = EnhancingBiasedPatientPatchDataset(
        dataset_root, splits["train"], roi_size=args.roi_size,
        samples_per_patient=args.samples_per_patient,
        tumor_sampling_probability=args.tumor_sampling_probability,
        enhancing_sampling_probability=args.enhancing_sampling_probability,
        seed=args.seed, augment=True,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, generator=loader_generator,
    )
    criterion = DiceCELoss(num_classes=4)
    all_available = torch.ones(4, dtype=torch.bool, device=device)
    mask_rng = random.Random(args.seed)
    best_epoch = None
    best_score = float("-inf")
    validation_history = []
    nonfinite_steps = []
    successful_global_steps = 0
    sampling_sanity = []

    epoch_path = RESULT_DIR / "self_distill_v5_training_log.csv"
    batch_path = RESULT_DIR / "self_distill_v5_training_batch_log.csv"
    tradeoff_path = RESULT_DIR / "self_distill_v5_validation_tradeoff.csv"
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
            "drop_t1ce_tc_et": v2.mean_tc_et(epoch_zero_drop),
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
            bn_count_epoch, _, all_bn_eval, all_bn_affine_frozen = freeze_batchnorm_fully(student)
            teacher.eval()
            if not all_bn_eval or not all_bn_affine_frozen or bn_count_epoch != bn_count:
                raise RuntimeError("BatchNorm freeze state changed")

            sums = {"total": 0.0, "seg": 0.0, "kd": 0.0, "feature": 0.0}
            successful_epoch_steps = 0
            skipped_epoch_steps = 0
            category_counts = {"full_anchor": 0, "drop_t1ce": 0, "random_subset": 0}
            for step, (images, targets, sampling_metadata) in enumerate(train_loader, start=1):
                global_step = (epoch - 1) * len(train_loader) + step
                if len(sampling_sanity) < 50:
                    needed = 50 - len(sampling_sanity)
                    sampling_sanity.extend(sampling_metadata[:needed].tolist())
                    if len(sampling_sanity) == 50:
                        actual_et_count = sum(row[0] == 3 for row in sampling_sanity)
                        forced_et_count = sum(row[1] == SAMPLING_ET for row in sampling_sanity)
                        fallback_count = sum(
                            row[1] == SAMPLING_ET_FALLBACK for row in sampling_sanity
                        )
                        sampling_report = {
                            "patches_checked": 50,
                            "actual_et_center_count": actual_et_count,
                            "actual_et_center_fraction": actual_et_count / 50.0,
                            "forced_et_branch_count": forced_et_count,
                            "forced_et_branch_fraction": forced_et_count / 50.0,
                            "empty_et_fallback_count": fallback_count,
                        }
                        (RESULT_DIR / "sampling_sanity_50.json").write_text(
                            json.dumps(sampling_report, indent=2) + "\n", encoding="utf-8",
                        )
                        print(
                            "SAMPLING_SANITY first_50 "
                            f"actual_et_centers={actual_et_count}/50 "
                            f"actual_et_fraction={actual_et_count / 50.0:.6f} "
                            f"forced_et_branch={forced_et_count}/50 "
                            f"forced_et_fraction={forced_et_count / 50.0:.6f} "
                            f"empty_et_fallbacks={fallback_count}", flush=True,
                        )
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                availability, category = sample_v4_mask(mask_rng, device)
                category_counts[category] += 1
                optimizer.zero_grad(set_to_none=True)

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True,
                ):
                    teacher_logits, teacher_features = common.forward_with_features(
                        teacher, images, all_available,
                    )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    student_logits, student_features = common.forward_with_features(
                        student, images, availability,
                    )
                    segmentation = criterion(student_logits, targets)
                distillation = kd_loss_float32(
                    student_logits, teacher_logits, args.temperature,
                )
                features = feature_loss_float32(student_features, teacher_features)
                loss = (
                    args.w_seg * segmentation.float() + args.w_kd * distillation +
                    args.w_feat * features
                )

                term_values = {
                    "total": float(loss.detach().item()),
                    "seg": float(segmentation.detach().float().item()),
                    "kd": float(distillation.detach().item()),
                    "feature": float(features.detach().item()),
                }
                terms_finite = {
                    "total": finite_value(loss),
                    "seg": finite_value(segmentation),
                    "kd": finite_value(distillation),
                    "feature": finite_value(features),
                }
                if not all(terms_finite.values()):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_epoch_steps += 1
                    nonfinite_steps.append({
                        "global_step": global_step,
                        "values": term_values,
                        "finite": terms_finite,
                    })
                    print(
                        f"SKIP non-finite step {global_step} values={term_values} "
                        f"finite={terms_finite}", flush=True,
                    )
                    batch_writer.writerow({
                        "epoch": epoch, "step": step, "global_step": global_step,
                        "mask_category": category,
                        "availability_mask": "".join(
                            "1" if value else "0" for value in availability.tolist()
                        ),
                        "total_loss": term_values["total"],
                        "seg_loss": term_values["seg"],
                        "kd_loss": term_values["kd"],
                        "feature_loss": term_values["feature"],
                        "grad_norm": "", "finite": False, "skipped": True,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })
                    batch_handle.flush()
                    if global_step <= 200:
                        raise RuntimeError(
                            f"First-200 stability failure at step {global_step}: "
                            f"values={term_values}, finite={terms_finite}"
                        )
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                grad_norm_finite = finite_value(grad_norm)
                if not grad_norm_finite:
                    optimizer.zero_grad(set_to_none=True)
                    skipped_epoch_steps += 1
                    nonfinite_steps.append({
                        "global_step": global_step,
                        "values": term_values,
                        "finite": terms_finite,
                        "grad_norm": float(grad_norm.detach().item()),
                    })
                    print(
                        f"SKIP non-finite step {global_step} finite_losses={term_values} "
                        f"grad_norm={float(grad_norm.detach().item())}", flush=True,
                    )
                    batch_writer.writerow({
                        "epoch": epoch, "step": step, "global_step": global_step,
                        "mask_category": category,
                        "availability_mask": "".join(
                            "1" if value else "0" for value in availability.tolist()
                        ),
                        "total_loss": term_values["total"],
                        "seg_loss": term_values["seg"],
                        "kd_loss": term_values["kd"],
                        "feature_loss": term_values["feature"],
                        "grad_norm": float(grad_norm.detach().item()),
                        "finite": False, "skipped": True,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    })
                    batch_handle.flush()
                    if global_step <= 200:
                        raise RuntimeError(
                            f"First-200 non-finite gradient at step {global_step}: "
                            f"losses={term_values}"
                        )
                    continue

                optimizer.step()
                successful_global_steps += 1
                successful_epoch_steps += 1
                for name in sums:
                    sums[name] += term_values[name]
                batch_writer.writerow({
                    "epoch": epoch, "step": step, "global_step": global_step,
                    "mask_category": category,
                    "availability_mask": "".join(
                        "1" if value else "0" for value in availability.tolist()
                    ),
                    "total_loss": term_values["total"],
                    "seg_loss": term_values["seg"],
                    "kd_loss": term_values["kd"],
                    "feature_loss": term_values["feature"],
                    "grad_norm": float(grad_norm.detach().item()),
                    "finite": True, "skipped": False,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })
                if global_step == 200:
                    stability = {
                        "steps_checked": 200,
                        "nonfinite_steps": nonfinite_steps,
                        "passed": len(nonfinite_steps) == 0,
                    }
                    (RESULT_DIR / "stability_200.json").write_text(
                        json.dumps(stability, indent=2) + "\n", encoding="utf-8",
                    )
                    print("STABILITY first_200_steps_passed=True nonfinite_losses=0", flush=True)
                if step % 25 == 0 or step == len(train_loader):
                    print(
                        f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                        f"loss={term_values['total']:.6f} seg={term_values['seg']:.6f} "
                        f"kd={term_values['kd']:.6f} feat={term_values['feature']:.6f} "
                        f"grad_norm={float(grad_norm.detach().item()):.6f} "
                        f"mask={category}", flush=True,
                    )
                    batch_handle.flush()

            if successful_epoch_steps == 0:
                raise RuntimeError(f"Epoch {epoch} had no successful optimizer steps")
            averages = {
                name: value / successful_epoch_steps for name, value in sums.items()
            }
            all_scores = None
            drop_scores = None
            all_mean = None
            drop_selection = None
            constraint_met = None
            selected_best = False
            if epoch % args.val_every == 0 or epoch == args.epochs:
                all_scores, drop_scores = v2.validate_both(
                    student, splits, dataset_root, device, args, f"Epoch {epoch} val",
                )
                all_mean = v2.mean_regions(all_scores)
                drop_selection = v2.mean_tc_et(drop_scores)
                constraint_met = all_mean >= constraint_threshold
                if constraint_met and drop_selection > best_score:
                    best_score = drop_selection
                    best_epoch = epoch
                    selected_best = True
                    save_checkpoint_atomic(
                        CHECKPOINT_DIR / "best_self_distill_v5.pth",
                        payload(
                            epoch, student, optimizer, args, split_sha256,
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
                "successful_steps": successful_epoch_steps,
                "skipped_nonfinite_steps": skipped_epoch_steps,
            })
            epoch_handle.flush()
            save_checkpoint_atomic(
                CHECKPOINT_DIR / "last_self_distill_v5.pth",
                payload(
                    epoch, student, optimizer, args, split_sha256,
                    teacher_baseline, constraint_threshold, all_scores,
                    drop_scores, best_epoch, best_score,
                ),
            )
            print(
                f"Epoch {epoch}: loss={averages['total']:.6f} "
                f"seg={averages['seg']:.6f} kd={averages['kd']:.6f} "
                f"feat={averages['feature']:.6f} masks={category_counts} "
                f"successful={successful_epoch_steps} skipped={skipped_epoch_steps} "
                f"time={elapsed:.1f}s", flush=True,
            )
            scheduler.step()
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
        "nonfinite_steps": nonfinite_steps,
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
            f"drop_selection={selected['drop_t1ce_tc_et']:.12f} "
            f"constraint_met={selected['constraint_met']}", flush=True,
        )


if __name__ == "__main__":
    main()
