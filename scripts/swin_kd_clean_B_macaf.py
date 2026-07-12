#!/usr/bin/env python3
"""Clean-teacher region-space KD into the leakage-safe MACAF student."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss  # noqa: E402
from scripts import swin_kd_clean_B_common as common  # noqa: E402
from scripts import swin_kd_clean_B_common_v2 as v2  # noqa: E402
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset, PatientVolumeDataset  # noqa: E402
from superlightnet.training import save_checkpoint_atomic  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "swin_kd_clean_B_macaf"
RESULT_DIR = (
    PROJECT_ROOT / "results" / "macaf_experiments" / "kd_clean_swin" / "training"
)
ALL_MODALITIES = ("t1", "t1ce", "t2", "flair")
DROP_T1CE_MODALITIES = ("t1", "t2", "flair")
EXPECTED_TEACHER_MODALITIES = ("t1", "t1ce", "t2", "flair")
EXPECTED_TEACHER_REORDER = (0, 1, 2, 3)
REJECTED_TEACHER_REORDERS = ((3, 1, 0, 2), (3, 0, 1, 2))
TEACHER_REGION_ORDER = ("tc", "wt", "et")
EXPECTED_TEACHER_PARAMETERS = 15_705_621
EXPECTED_STUDENT_PARAMETERS = 5_815_280
EXPECTED_TEACHER_SHA256 = "62d166b0cf82e41a21bdd018f6c8ca901833a9f29f732c9468192ff45cbdcb80"
EXPECTED_STUDENT_SHA256_PREFIX = "2c19ab71"
EXPECTED_SPLIT_SHA256 = "9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881"

EPOCH_COLUMNS = tuple(
    column for column in v2.EPOCH_COLUMNS if column != "feature_loss"
) + ("successful_steps", "skipped_nonfinite_steps")
BATCH_COLUMNS = (
    "epoch", "step", "global_step", "mask_category", "availability_mask",
    "total_loss", "seg_loss", "kd_loss", "grad_norm",
    "finite", "skipped", "learning_rate",
)
TRADEOFF_COLUMNS = v2.TRADEOFF_COLUMNS


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


def student_region_probabilities(student_logits: torch.Tensor) -> torch.Tensor:
    """Convert 4-class softmax to teacher order [TC, WT, ET]."""
    probabilities = F.softmax(student_logits.float(), dim=1)
    p_tc = probabilities[:, 1] + probabilities[:, 3]
    p_wt = 1.0 - probabilities[:, 0]
    p_et = probabilities[:, 3]
    return torch.stack((p_tc, p_wt, p_et), dim=1)


def region_kd_loss_float32(student_logits: torch.Tensor,
                           teacher_logits: torch.Tensor) -> torch.Tensor:
    if teacher_logits.ndim != 5 or teacher_logits.shape[1] != 3:
        raise RuntimeError(
            "Teacher must output exactly three sigmoid channels in TC,WT,ET order; "
            f"got shape={tuple(teacher_logits.shape)}"
        )
    student_regions = student_region_probabilities(student_logits)
    teacher_regions = torch.sigmoid(teacher_logits.float()).detach()
    return F.mse_loss(student_regions, teacher_regions, reduction="mean")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_gpu_idle(max_external_mib: int = 1024) -> None:
    """Fail before CUDA context creation when compute processes hold over 1 GiB."""
    command = [
        "nvidia-smi", "--query-compute-apps=used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify external GPU usage with nvidia-smi") from exc
    held_mib = 0
    unavailable = False
    for line in completed.stdout.splitlines():
        value = line.strip()
        if "N/A" in value.upper():
            unavailable = True
        elif value:
            held_mib += int(value)
    source = "compute_apps"
    if unavailable:
        fallback = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        held_mib = sum(int(line.strip()) for line in fallback.stdout.splitlines() if line.strip())
        source = "gpu_total_wddm_fallback"
    print(
        f"GPU_EXTERNAL_USAGE used_memory_mib={held_mib} limit_mib={max_external_mib} "
        f"source={source}", flush=True,
    )
    if held_mib > max_external_mib:
        raise RuntimeError("GPU BUSY - not starting")


def derive_teacher_alignment(checkpoint: dict):
    if checkpoint.get("validated") is not True:
        raise RuntimeError("Clean-teacher checkpoint is not marked validated")
    teacher_modalities = tuple(checkpoint.get("input_modality_order", ()))
    if set(teacher_modalities) != set(ALL_MODALITIES) or len(teacher_modalities) != 4:
        raise RuntimeError(
            "Clean-teacher checkpoint has invalid input_modality_order metadata: "
            f"{teacher_modalities!r}"
        )
    reorder = tuple(ALL_MODALITIES.index(name) for name in teacher_modalities)
    print(f"DERIVED teacher input order={teacher_modalities} reorder={reorder}", flush=True)
    print(
        "REJECTED teacher reorders: old_fold=(3, 1, 0, 2) "
        "macaf_swin_kd_scramble=(3, 0, 1, 2)", flush=True,
    )
    if teacher_modalities != EXPECTED_TEACHER_MODALITIES or reorder != EXPECTED_TEACHER_REORDER:
        raise RuntimeError(
            f"Clean-teacher reorder must be identity {EXPECTED_TEACHER_REORDER}; "
            f"metadata={teacher_modalities}, derived={reorder}"
        )
    if reorder in REJECTED_TEACHER_REORDERS:
        raise RuntimeError(f"Rejected historical teacher reorder selected: {reorder}")
    region_order = tuple(checkpoint.get("region_order", ()))
    if region_order != TEACHER_REGION_ORDER:
        raise RuntimeError(
            f"Teacher output order mismatch: metadata={region_order}, "
            f"required={TEACHER_REGION_ORDER}"
        )
    print(
        f"KD teacher outputs=3 sigmoid channels order={region_order}; "
        "region-space MSE consumes order=('tc', 'wt', 'et')", flush=True,
    )
    return teacher_modalities, reorder


def leakage_and_alignment_gate(splits, split_sha256: str, teacher_checkpoint: dict,
                               teacher_modalities, teacher_reorder):
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    overlaps = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient split overlap detected: {overlaps}")
    teacher_split_sha = str(teacher_checkpoint.get("split_manifest_sha256", "")).lower()
    if teacher_split_sha != split_sha256:
        raise RuntimeError(
            f"Clean-teacher split hash mismatch: teacher={teacher_split_sha!r}, "
            f"current={split_sha256!r}"
        )
    report = {
        "train_cases": len(train_ids),
        "validation_cases": len(val_ids),
        "test_cases": len(test_ids),
        "overlaps": overlaps,
        "split_manifest_sha256": split_sha256,
        "teacher_split_manifest_sha256": teacher_split_sha,
        "teacher_modalities": list(teacher_modalities),
        "student_modalities": list(ALL_MODALITIES),
        "student_to_teacher_reorder": list(teacher_reorder),
        "teacher_region_order": list(TEACHER_REGION_ORDER),
        "teacher_test_access": False,
    }
    print(
        f"LEAKAGE-CHECK PASSED train={len(train_ids)} val={len(val_ids)} "
        f"test={len(test_ids)} zero_overlap=True", flush=True,
    )
    print(
        "CLEAN-TEACHER-NEVER-SAW-TEST: checkpoint split metadata matches; "
        "test IDs are excluded from teacher/student training and validation selection.",
        flush=True,
    )
    return report


def build_swin_teacher(checkpoint_path: Path, device: torch.device):
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required. Install it in gan_workshop before running this script."
        ) from exc
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_size = int(checkpoint.get("feature_size", -1))
    if feature_size != 24:
        raise RuntimeError(
            f"Clean-teacher feature_size metadata mismatch: expected=24, actual={feature_size}"
        )
    kwargs = {
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": feature_size,
        "use_checkpoint": False,
    }
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = (128, 128, 128)
    teacher = SwinUNETR(**kwargs)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model_keys = set(teacher.state_dict())
    obsolete_buffer_suffixes = ("relative_position_index", "attn_mask")
    ignored_obsolete_buffers = sorted(
        key for key in state_dict
        if key not in model_keys and key.endswith(obsolete_buffer_suffixes)
    )
    unexpected = sorted(
        key for key in state_dict
        if key not in model_keys and key not in ignored_obsolete_buffers
    )
    missing = sorted(key for key in model_keys if key not in state_dict)
    if missing or unexpected:
        raise RuntimeError(
            f"Teacher checkpoint incompatibility: missing={missing}, unexpected={unexpected}"
        )
    compatible_state = {
        key: value for key, value in state_dict.items() if key in model_keys
    }
    teacher.load_state_dict(compatible_state, strict=True)
    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Swin teacher freeze failed")
    parameter_count = sum(p.numel() for p in teacher.parameters())
    if parameter_count != EXPECTED_TEACHER_PARAMETERS:
        raise RuntimeError(
            f"Teacher parameter mismatch: expected={EXPECTED_TEACHER_PARAMETERS}, "
            f"actual={parameter_count}"
        )
    if int(checkpoint.get("param_count", -1)) != EXPECTED_TEACHER_PARAMETERS:
        raise RuntimeError(
            "Teacher checkpoint param_count metadata mismatch: "
            f"{checkpoint.get('param_count', 'NOT_FOUND')}"
        )
    print(
        f"TEACHER_LOAD checkpoint={checkpoint_path.resolve()} "
        f"epoch={checkpoint.get('epoch', 'NOT_FOUND')} "
        f"best_val_mean_dice={checkpoint.get('best_val_mean_dice', 'NOT_FOUND')} "
        f"parameters={parameter_count} "
        f"ignored_obsolete_buffers={len(ignored_obsolete_buffers)}", flush=True,
    )
    return teacher, checkpoint


def binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int((prediction & target).sum()) / denominator


@torch.inference_mode()
def teacher_sanity(teacher, case_ids, dataset_root, device, overlap: float,
                   minimum_region_dice: float, teacher_reorder):
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise RuntimeError("MONAI inferers are unavailable") from exc
    selected_ids = tuple(case_ids[:5])
    if len(selected_ids) != 5:
        raise RuntimeError("Teacher sanity requires five training cases")
    dataset = PatientVolumeDataset(dataset_root, selected_ids)
    scores = []
    for index in range(len(dataset)):
        images, target, case_id = dataset[index]
        teacher_input = images.unsqueeze(0).to(device)[:, teacher_reorder]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = sliding_window_inference(
                teacher_input, roi_size=(128, 128, 128), sw_batch_size=1,
                predictor=teacher, overlap=overlap, mode="constant",
            )
        prediction = torch.sigmoid(logits.float()).cpu()[0] >= 0.5
        case_scores = {
            "tc": binary_dice(prediction[0], (target == 1) | (target == 3)),
            "wt": binary_dice(prediction[1], target > 0),
            "et": binary_dice(prediction[2], target == 3),
        }
        scores.append(case_scores)
        print(
            f"TEACHER_SANITY {index + 1}/5 {case_id} "
            f"WT={case_scores['wt']:.12f} TC={case_scores['tc']:.12f} "
            f"ET={case_scores['et']:.12f}", flush=True,
        )
        del teacher_input, logits, prediction
        torch.cuda.empty_cache()
    means = {
        region: sum(row[region] for row in scores) / len(scores)
        for region in ("wt", "tc", "et")
    }
    print(
        f"TEACHER_SANITY_MEAN WT={means['wt']:.12f} "
        f"TC={means['tc']:.12f} ET={means['et']:.12f}", flush=True,
    )
    if min(means.values()) < minimum_region_dice:
        raise RuntimeError(
            f"Teacher alignment gate failed: means={means}, "
            f"minimum_region_dice={minimum_region_dice}. Check channel order, "
            "normalization, output order, and checkpoint compatibility."
        )
    return {"case_ids": list(selected_ids), "per_case": scores, "mean": means}


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
        "student_all_baseline": teacher_baseline,
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
        },
        "kd_space": "regions_tc_wt_et",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": args.teacher_checkpoint_sha256,
        "teacher_modalities": args.teacher_modalities,
        "student_to_teacher_reorder": args.teacher_reorder,
        "teacher_region_order": TEACHER_REGION_ORDER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/macaf/best_patient_split.pth"))
    parser.add_argument(
        "--teacher_checkpoint", type=Path,
        default=Path("checkpoints/swin_teacher_clean/best.pth"),
    )
    parser.add_argument("--teacher_sanity_min_region", type=float, default=0.50)
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
    args = parser.parse_args()

    if (args.epochs < 1 or args.lr <= 0 or args.min_lr < 0 or args.min_lr >= args.lr or
            args.batch_size < 1 or args.val_every < 1 or
            args.early_stop_patience < 1 or args.samples_per_patient < 1 or
            args.num_workers < 0 or not 0.0 <= args.overlap < 1.0 or
            min(args.w_seg, args.w_kd) < 0 or
            not 0.0 <= args.teacher_sanity_min_region <= 1.0):
        parser.error("Invalid training, validation, or loss parameter")
    if RESULT_DIR.exists():
        raise FileExistsError(
            f"Refusing to reuse Swin KD results: {RESULT_DIR}"
        )
    for output_name in ("best.pth", "last.pth"):
        if (CHECKPOINT_DIR / output_name).exists():
            raise FileExistsError(f"Refusing to overwrite {CHECKPOINT_DIR / output_name}")

    require_gpu_idle()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Swin KD requires a free CUDA device")
    checkpoint_path = args.checkpoint.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    split_json = args.split_json.resolve()
    expected_source = (PROJECT_ROOT / "checkpoints" / "macaf" /
                       "best_patient_split.pth").resolve()
    if checkpoint_path != expected_source or not checkpoint_path.is_file():
        raise ValueError(f"Student must initialize from the MACAF v1 checkpoint: {expected_source}")
    for required in (split_json, args.teacher_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    student_sha256 = file_sha256(checkpoint_path)
    if not student_sha256.startswith(EXPECTED_STUDENT_SHA256_PREFIX):
        raise RuntimeError(
            f"MACAF v1 checkpoint SHA mismatch: expected prefix="
            f"{EXPECTED_STUDENT_SHA256_PREFIX}, actual={student_sha256}"
        )
    teacher_sha256 = file_sha256(args.teacher_checkpoint)
    if teacher_sha256 != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(
            f"Clean-teacher SHA mismatch: expected={EXPECTED_TEACHER_SHA256}, "
            f"actual={teacher_sha256}"
        )
    split_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    if split_sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"Split manifest SHA mismatch: expected={EXPECTED_SPLIT_SHA256}, "
            f"actual={split_sha256}"
        )
    print(f"STUDENT_CHECKPOINT_SHA256 {student_sha256}", flush=True)
    print(f"TEACHER_CHECKPOINT_SHA256 {teacher_sha256}", flush=True)
    print(f"SPLIT_MANIFEST_SHA256 {split_sha256}", flush=True)

    common.set_seed(args.seed)
    splits = load_split_manifest(str(split_json))
    dataset_root = common.resolve_dataset_root(split_json)
    state_dict, source_checkpoint = common.load_checkpoint_weights(checkpoint_path)
    args.source_checkpoint_epoch = int(source_checkpoint["epoch"])
    student = common.build_model(state_dict, device)
    student.rmd_enable = False
    student_parameter_count = sum(parameter.numel() for parameter in student.parameters())
    if student_parameter_count != EXPECTED_STUDENT_PARAMETERS:
        raise RuntimeError(
            f"MACAF student parameter mismatch: expected={EXPECTED_STUDENT_PARAMETERS}, "
            f"actual={student_parameter_count}"
        )
    print(
        f"MACAF_STUDENT parameters={student_parameter_count} "
        "delta_vs_mean_fusion=19532", flush=True,
    )
    teacher, teacher_checkpoint = build_swin_teacher(args.teacher_checkpoint, device)
    args.teacher_modalities, args.teacher_reorder = derive_teacher_alignment(teacher_checkpoint)
    leakage_report = leakage_and_alignment_gate(
        splits, split_sha256, teacher_checkpoint,
        args.teacher_modalities, args.teacher_reorder,
    )
    args.teacher_checkpoint_sha256 = teacher_sha256
    teacher_sanity_report = teacher_sanity(
        teacher, splits["train"], dataset_root, device, args.overlap,
        args.teacher_sanity_min_region, args.teacher_reorder,
    )
    print(
        "NORMALIZATION confirmed=per_channel_nonzero_zscore "
        "student_and_teacher_reuse_same_normalized_tensor=True", flush=True,
    )

    stored = common.stored_validation_row(
        PROJECT_ROOT / "results" / "macaf_experiments" / "base" / "v1" /
        "training" / "macaf_v1_training_log.csv",
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    (RESULT_DIR / "macaf_clean_swin_kd_sanity.json").write_text(json.dumps({
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": args.source_checkpoint_epoch,
        "student_full_stored": stored,
        "epoch0_all": epoch_zero_all,
        "epoch0_drop_t1ce": epoch_zero_drop,
        "student_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_checkpoint_sha256": args.teacher_checkpoint_sha256,
        "teacher_sanity": teacher_sanity_report,
        "leakage_gate": leakage_report,
        "normalization": "per-channel z-score over nonzero voxels",
        "teacher_region_order": TEACHER_REGION_ORDER,
        "teacher_modalities": args.teacher_modalities,
        "student_to_teacher_reorder": args.teacher_reorder,
        "batchnorm_modules_frozen": bn_count,
        "batchnorm_affine_parameters_frozen": frozen_bn_parameters,
        "trainable_optimizer_parameters": optimizer_parameter_count,
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
    criterion = DiceCELoss(num_classes=4)
    mask_rng = random.Random(args.seed)
    best_epoch = None
    best_score = float("-inf")
    validation_history = []
    nonfinite_steps = []
    successful_global_steps = 0

    epoch_path = RESULT_DIR / "macaf_clean_swin_kd_training_log.csv"
    batch_path = RESULT_DIR / "macaf_clean_swin_kd_training_batch_log.csv"
    tradeoff_path = RESULT_DIR / "macaf_clean_swin_kd_validation_tradeoff.csv"
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

            sums = {"total": 0.0, "seg": 0.0, "kd": 0.0}
            successful_epoch_steps = 0
            skipped_epoch_steps = 0
            category_counts = {"full_anchor": 0, "drop_t1ce": 0, "random_subset": 0}
            for step, (images, targets) in enumerate(train_loader, start=1):
                global_step = (epoch - 1) * len(train_loader) + step
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                availability, category = sample_v4_mask(mask_rng, device)
                category_counts[category] += 1
                optimizer.zero_grad(set_to_none=True)

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True,
                ):
                    teacher_logits = teacher(images[:, args.teacher_reorder])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    student_logits = student(images, avail_mask=availability)
                    segmentation = criterion(student_logits, targets)
                distillation = region_kd_loss_float32(student_logits, teacher_logits)
                loss = args.w_seg * segmentation.float() + args.w_kd * distillation

                term_values = {
                    "total": float(loss.detach().item()),
                    "seg": float(segmentation.detach().float().item()),
                    "kd": float(distillation.detach().item()),
                }
                terms_finite = {
                    "total": finite_value(loss),
                    "seg": finite_value(segmentation),
                    "kd": finite_value(distillation),
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
                    (RESULT_DIR / "macaf_clean_swin_kd_stability_200.json").write_text(
                        json.dumps(stability, indent=2) + "\n", encoding="utf-8",
                    )
                    print("STABILITY first_200_steps_passed=True nonfinite_losses=0", flush=True)
                if step % 25 == 0 or step == len(train_loader):
                    print(
                        f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                        f"loss={term_values['total']:.6f} seg={term_values['seg']:.6f} "
                        f"kd={term_values['kd']:.6f} "
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
                        CHECKPOINT_DIR / "best.pth",
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
                CHECKPOINT_DIR / "last.pth",
                payload(
                    epoch, student, optimizer, args, split_sha256,
                    teacher_baseline, constraint_threshold, all_scores,
                    drop_scores, best_epoch, best_score,
                ),
            )
            print(
                f"Epoch {epoch}: loss={averages['total']:.6f} "
                f"seg={averages['seg']:.6f} kd={averages['kd']:.6f} "
                f"masks={category_counts} "
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
        "student_all_baseline": teacher_baseline,
        "full_constraint_threshold": constraint_threshold,
        "best_epoch": best_epoch,
        "best_drop_t1ce_tc_et": None if best_epoch is None else best_score,
        "constraint_ever_met_after_training": any(
            row["constraint_met"] for row in validation_history
        ),
        "nonfinite_steps": nonfinite_steps,
        "validation_tradeoff": validation_history,
    }
    (RESULT_DIR / "macaf_clean_swin_kd_selection_status.json").write_text(
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
    print("No existing files were modified. Test set never accessed during training.", flush=True)


if __name__ == "__main__":
    main()
