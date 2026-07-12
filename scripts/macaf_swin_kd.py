#!/usr/bin/env python3
"""Post-train MACAF with region-space KD from a frozen MONAI Swin UNETR teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import Cfg  # noqa: E402
from main import DiceCELoss, MultiEncoderRMDUNet  # noqa: E402
from split_utils import load_split_manifest  # noqa: E402
from superlightnet.patient_data import PatientPatchDataset, PatientVolumeDataset  # noqa: E402
from superlightnet.training import assert_dataset_isolation, save_checkpoint_atomic, window_starts  # noqa: E402

try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SwinUNETR
except ImportError as exc:  # pragma: no cover - exercised by the user's env.
    raise ImportError(
        "MONAI is required for Swin UNETR KD. Install it in the active env with: "
        "pip install monai[all]"
    ) from exc


MODEL_MODALITIES = ("t1", "t1ce", "t2", "flair")
TEACHER_MODALITIES = ("flair", "t1", "t1ce", "t2")
TEACHER_REORDER = tuple(MODEL_MODALITIES.index(name) for name in TEACHER_MODALITIES)
TEACHER_OUTPUT_ORDER = ("tc", "wt", "et")
DROP_T1CE_MASK = (True, False, True, True)
LOG_COLUMNS = (
    "epoch", "train_loss", "seg_loss", "kd_loss", "skipped_steps",
    "all_dice_wt", "all_dice_tc", "all_dice_et", "all_mean_dice",
    "drop_t1ce_dice_wt", "drop_t1ce_dice_tc", "drop_t1ce_dice_et",
    "drop_t1ce_tc_et_mean", "all_baseline", "all_constraint",
    "eligible_for_best", "best_drop_t1ce_tc_et_mean", "learning_rate", "epoch_time_sec",
)
TEACHER_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
    "0.8.1/fold0_f48_ep300_4gpu_dice0_8854.zip"
)


def parse_roi_size(value: str) -> Tuple[int, int, int]:
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


def safe_output_paths(overwrite: bool) -> Tuple[Path, Path, Path]:
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "macaf_swin_kd"
    result_dir = PROJECT_ROOT / "results" / "macaf_experiments" / "kd_old_swin" / "training"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "macaf_old_swin_kd_training_log.csv"
    best_path = ckpt_dir / "best_macaf_swin_kd.pth"
    last_path = ckpt_dir / "last_macaf_swin_kd.pth"
    existing = [path for path in (log_path, best_path, last_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing KD artifact: {existing[0]}. "
            "Use --overwrite only if you intentionally want a fresh KD run."
        )
    return best_path, last_path, log_path


def download_teacher_checkpoint(output_path: Path, url: str = TEACHER_URL) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        print(f"Teacher checkpoint already present: {output_path}", flush=True)
        return output_path

    download_path = output_path.with_suffix(".zip")
    print(f"Downloading Swin UNETR teacher from {url}", flush=True)
    urllib.request.urlretrieve(url, download_path)
    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.lower().endswith((".pt", ".pth", ".ckpt"))
            ]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint file found inside {download_path}")
            candidates.sort(key=lambda name: (len(Path(name).parts), name))
            with archive.open(candidates[0]) as source, output_path.open("wb") as target:
                target.write(source.read())
        download_path.unlink(missing_ok=True)
    else:
        download_path.replace(output_path)
    print(f"Saved teacher checkpoint to {output_path}", flush=True)
    return output_path


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    state_dict = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ("module.", "model.", "network."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        state_dict[new_key] = value
    return state_dict


def load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        print(
            f"weights_only checkpoint load failed for {path.name}: {exc}. "
            "Retrying with standard torch.load.",
            flush=True,
        )
        return torch.load(path, map_location=map_location)


def build_teacher(checkpoint_path: Path, device: torch.device) -> nn.Module:
    signature = inspect.signature(SwinUNETR)
    kwargs = {"in_channels": 4, "out_channels": 3, "feature_size": 48}
    if "img_size" in signature.parameters:
        kwargs["img_size"] = (128, 128, 128)
    if "spatial_dims" in signature.parameters:
        kwargs["spatial_dims"] = 3
    if "use_checkpoint" in signature.parameters:
        kwargs["use_checkpoint"] = False
    teacher = SwinUNETR(**kwargs).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Swin teacher checkpoint keys: {unexpected[:10]}")
    if len(missing) > 0:
        raise RuntimeError(f"Missing Swin teacher checkpoint keys: {missing[:10]}")
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def build_student(checkpoint_path: Path, device: torch.device, rmd_enable: bool) -> nn.Module:
    student = MultiEncoderRMDUNet(
        in_modalities=4, base_ch=16, num_stages=4, num_classes=4,
        rmd_enable=rmd_enable, fusion_reduction=4,
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return student


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            if module.weight is not None:
                module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False


def set_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()


def target_regions(target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1)
    tc = ((target == 1) | (target == 3)).float()
    wt = (target > 0).float()
    et = (target == 3).float()
    return torch.cat((tc, wt, et), dim=1)


def student_region_probs(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=1)
    wt = 1.0 - probs[:, 0:1]
    tc = probs[:, 1:2] + probs[:, 3:4]
    et = probs[:, 3:4]
    return torch.cat((tc, wt, et), dim=1)


def dice_from_masks(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    scores = {}
    for index, name in enumerate(TEACHER_OUTPUT_ORDER):
        pred = prediction[:, index].bool()
        truth = target[:, index].bool()
        denom = int(pred.sum().item()) + int(truth.sum().item())
        if denom == 0:
            score = 1.0
        else:
            score = float(2.0 * torch.logical_and(pred, truth).sum().item() / denom)
        scores[name] = score
    return scores


def average_region_scores(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    return {name: float(np.mean([row[name] for row in rows])) for name in ("wt", "tc", "et")}


@torch.inference_mode()
def teacher_predict_volume(teacher: nn.Module, volume: torch.Tensor, device: torch.device,
                           roi_size: Tuple[int, int, int], overlap: float) -> torch.Tensor:
    image = volume.unsqueeze(0).to(device)
    image = image[:, TEACHER_REORDER]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = sliding_window_inference(
            image, roi_size=roi_size, sw_batch_size=1, predictor=teacher, overlap=overlap,
        )
    return torch.sigmoid(logits.float()).cpu()


@torch.inference_mode()
def teacher_sanity(teacher: nn.Module, dataset: PatientVolumeDataset, device: torch.device,
                   roi_size: Tuple[int, int, int], overlap: float, count: int) -> Dict[str, float]:
    rows = []
    for index in range(min(count, len(dataset))):
        volume, target, case_id = dataset[index]
        probs = teacher_predict_volume(teacher, volume, device, roi_size, overlap)
        pred = probs > 0.5
        truth = target_regions(target.unsqueeze(0))
        scores = dice_from_masks(pred, truth)
        rows.append(scores)
        print(
            f"Teacher sanity {case_id}: WT={scores['wt']:.6f} "
            f"TC={scores['tc']:.6f} ET={scores['et']:.6f}",
            flush=True,
        )
    return average_region_scores(rows)


@torch.inference_mode()
def sliding_window_student_logits(model: nn.Module, volume: torch.Tensor, device: torch.device,
                                  roi_size: Tuple[int, int, int], overlap: float,
                                  avail_mask: torch.Tensor | None) -> torch.Tensor:
    spatial_shape = tuple(volume.shape[-3:])
    padded_shape = tuple(max(size, roi) for size, roi in zip(spatial_shape, roi_size))
    padded = torch.zeros((volume.shape[0], *padded_shape), dtype=torch.float32)
    padded[(slice(None),) + tuple(slice(0, size) for size in spatial_shape)] = volume
    scores = torch.zeros((4, *padded_shape), dtype=torch.float32)
    counts = torch.zeros(padded_shape, dtype=torch.float32)
    starts = [window_starts(size, roi, overlap) for size, roi in zip(padded_shape, roi_size)]
    mask = avail_mask.to(device) if avail_mask is not None else None

    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                slices = (slice(x, x + roi_size[0]), slice(y, y + roi_size[1]), slice(z, z + roi_size[2]))
                patch = padded[(slice(None),) + slices].unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(patch, avail_mask=mask)
                scores[(slice(None),) + slices] += logits[0].float().cpu()
                counts[slices] += 1.0
    if torch.any(counts == 0):
        raise AssertionError("Sliding-window inference left uncovered voxels")
    crop = tuple(slice(0, size) for size in spatial_shape)
    return (scores / counts.unsqueeze(0))[(slice(None),) + crop]


@torch.inference_mode()
def evaluate_student(model: nn.Module, dataset: PatientVolumeDataset, device: torch.device,
                     roi_size: Tuple[int, int, int], overlap: float,
                     avail_mask: torch.Tensor | None, label: str,
                     max_cases: int | None = None) -> Dict[str, float]:
    model.eval()
    rows = []
    case_count = len(dataset) if max_cases is None else min(max_cases, len(dataset))
    for index in range(case_count):
        volume, target, case_id = dataset[index]
        volume = volume.clone()
        if avail_mask is not None:
            mask_cpu = avail_mask.cpu().bool()
            for channel, keep in enumerate(mask_cpu):
                if not bool(keep):
                    volume[channel].zero_()
        logits = sliding_window_student_logits(model, volume, device, roi_size, overlap, avail_mask)
        pred_label = torch.argmax(logits, dim=0).unsqueeze(0)
        pred_regions = target_regions(pred_label)
        truth_regions = target_regions(target.unsqueeze(0))
        scores = dice_from_masks(pred_regions.bool(), truth_regions.bool())
        rows.append(scores)
        print(
            f"{label} {index + 1}/{case_count} {case_id}: "
            f"WT={scores['wt']:.6f} TC={scores['tc']:.6f} ET={scores['et']:.6f}",
            flush=True,
        )
    return average_region_scores(rows)


def random_student_mask(device: torch.device) -> torch.Tensor:
    draw = random.random()
    if draw < 0.30:
        mask = [True, True, True, True]
    elif draw < 0.80:
        mask = list(DROP_T1CE_MASK)
    else:
        mask = [bool(random.getrandbits(1)) for _ in MODEL_MODALITIES]
        if not any(mask):
            mask[random.randrange(len(mask))] = True
    return torch.tensor(mask, dtype=torch.bool, device=device)


def require_cuda_sane(device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 autocast was requested, but this GPU does not report bf16 support")


def print_alignment() -> None:
    print("ALIGNMENT CHECKS", flush=True)
    print(
        "Input modality order: repo uses "
        f"{MODEL_MODALITIES}; MONAI Swin teacher uses {TEACHER_MODALITIES}; "
        f"teacher reorder indices={TEACHER_REORDER}.",
        flush=True,
    )
    print(
        "Normalization: repo normalize_nonzero is per-channel nonzero z-score; "
        "MONAI BRATS21 transforms use NormalizeIntensityd(nonzero=True, channel_wise=True).",
        flush=True,
    )
    print(
        "Teacher output order: TC, WT, ET. Student regions are mapped as "
        "TC=p_necrotic+p_enhancing, WT=1-p_bg, ET=p_enhancing.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_json", type=Path, default=Path("splits/patient_splits.json"))
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--student_checkpoint", type=Path, default=Path("checkpoints/macaf/best_patient_split.pth"))
    parser.add_argument(
        "--teacher_checkpoint", type=Path,
        default=Path("pretrained_teachers/swin_unetr_brats21_fold0.pth"),
    )
    parser.add_argument("--teacher_url", default=TEACHER_URL)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_kd", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--samples_per_patient", type=int, default=1)
    parser.add_argument("--roi_size", type=parse_roi_size, default=(128, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sanity_cases", type=int, default=5)
    parser.add_argument("--teacher_min_wt", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error("This KD recipe expects batch_size=1 so each step has one modality mask")
    if args.epochs < 1 or args.lr <= 0 or args.w_seg < 0 or args.w_kd < 0 or args.patience < 1:
        parser.error("epochs/lr/weights/patience must be positive")
    device = torch.device(args.device)
    require_cuda_sane(device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    split_json = args.split_json.resolve()
    if not split_json.is_file():
        raise FileNotFoundError(split_json)
    student_checkpoint = args.student_checkpoint.resolve()
    if not student_checkpoint.is_file():
        raise FileNotFoundError(student_checkpoint)
    dataset_root = resolve_dataset_root(split_json)
    splits = load_split_manifest(str(split_json))
    best_path, last_path, log_path = safe_output_paths(args.overwrite)

    print_alignment()
    teacher_checkpoint = download_teacher_checkpoint(args.teacher_checkpoint.resolve(), args.teacher_url)
    teacher = build_teacher(teacher_checkpoint, device)
    teacher_sha = hashlib.sha256(teacher_checkpoint.read_bytes()).hexdigest()
    print(f"Teacher checkpoint sha256={teacher_sha}", flush=True)

    train_dataset = PatientPatchDataset(
        dataset_root, splits[args.train_split], roi_size=args.roi_size,
        samples_per_patient=args.samples_per_patient, seed=args.seed, augment=True,
    )
    val_dataset = PatientVolumeDataset(dataset_root, splits[args.val_split])
    train_volume_dataset = PatientVolumeDataset(dataset_root, splits[args.train_split])
    assert_dataset_isolation(train_dataset, val_dataset, splits)
    print(
        f"LEAKAGE CHECK PASSED: train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}; test patients are not instantiated.",
        flush=True,
    )

    teacher_scores = teacher_sanity(
        teacher, train_volume_dataset, device, args.roi_size, args.overlap, args.sanity_cases,
    )
    print(
        "Teacher sanity mean: "
        f"WT={teacher_scores['wt']:.6f} TC={teacher_scores['tc']:.6f} ET={teacher_scores['et']:.6f}",
        flush=True,
    )
    if teacher_scores["wt"] < args.teacher_min_wt:
        raise RuntimeError(
            f"Teacher sanity WT={teacher_scores['wt']:.6f} is below --teacher_min_wt={args.teacher_min_wt}; "
            "alignment may be wrong, stopping before KD."
        )

    eval_student = build_student(student_checkpoint, device, rmd_enable=False)
    student_train = build_student(student_checkpoint, device, rmd_enable=True)
    student_sanity = evaluate_student(
        eval_student, train_volume_dataset, device, args.roi_size, args.overlap,
        avail_mask=None, label="Student full-4 sanity", max_cases=args.sanity_cases,
    )
    print(
        "Student sanity mean: "
        f"WT={student_sanity['wt']:.6f} TC={student_sanity['tc']:.6f} ET={student_sanity['et']:.6f}",
        flush=True,
    )

    all_baseline_scores = evaluate_student(
        eval_student, val_dataset, device, args.roi_size, args.overlap,
        avail_mask=None, label="Student val all-4 baseline",
    )
    all_baseline = float(np.mean([all_baseline_scores["wt"], all_baseline_scores["tc"], all_baseline_scores["et"]]))
    all_constraint = 0.97 * all_baseline
    drop_mask = torch.tensor(DROP_T1CE_MASK, dtype=torch.bool)
    epoch0_drop = evaluate_student(
        eval_student, val_dataset, device, args.roi_size, args.overlap,
        avail_mask=drop_mask, label="Student epoch0 drop-T1ce",
    )
    print(
        "SANITY SUMMARY: "
        f"teacher WT/TC/ET={teacher_scores['wt']:.6f}/{teacher_scores['tc']:.6f}/{teacher_scores['et']:.6f}; "
        f"student train full-4 WT/TC/ET={student_sanity['wt']:.6f}/{student_sanity['tc']:.6f}/{student_sanity['et']:.6f}; "
        f"student val all_baseline={all_baseline:.6f}; constraint={all_constraint:.6f}; "
        f"epoch0 drop-T1ce TC={epoch0_drop['tc']:.6f} ET={epoch0_drop['et']:.6f}",
        flush=True,
    )

    del eval_student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    freeze_batch_norm(student_train)
    trainable_params = sum(p.numel() for p in student_train.parameters() if p.requires_grad)
    print(f"Trainable student params after BN freeze: {trainable_params}", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in student_train.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DiceCELoss(num_classes=4)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        generator=loader_generator,
    )

    best_primary = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            random.seed(args.seed + epoch)
            loader_generator.manual_seed(args.seed + epoch)
            student_train.train()
            set_batch_norm_eval(student_train)
            teacher.eval()
            train_loss_total = 0.0
            seg_loss_total = 0.0
            kd_loss_total = 0.0
            skipped_steps = 0

            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                mask = random_student_mask(device)
                student_images = images.clone()
                for channel, keep in enumerate(mask):
                    if not bool(keep):
                        student_images[:, channel].zero_()

                with torch.no_grad():
                    teacher_images = images[:, TEACHER_REORDER]
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        teacher_logits = teacher(teacher_images)
                    teacher_probs = torch.sigmoid(teacher_logits.float()).detach()

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    student_logits = student_train(student_images, avail_mask=mask)

                logits_f32 = student_logits.float()
                seg_loss = criterion(logits_f32, targets)
                regions = student_region_probs(logits_f32)
                kd_loss = F.binary_cross_entropy(
                    regions.clamp(1e-6, 1.0 - 1e-6), teacher_probs.float(),
                )
                total_loss = args.w_seg * seg_loss + args.w_kd * kd_loss
                if not torch.isfinite(total_loss):
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    print(f"Epoch {epoch} step {step}: non-finite loss skipped", flush=True)
                    continue
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student_train.parameters() if p.requires_grad], max_norm=1.0,
                )
                optimizer.step()
                train_loss_total += float(total_loss.item())
                seg_loss_total += float(seg_loss.item())
                kd_loss_total += float(kd_loss.item())
                print(
                    f"Epoch {epoch}/{args.epochs} train {step}/{len(train_loader)} "
                    f"loss={float(total_loss.item()):.6f} seg={float(seg_loss.item()):.6f} "
                    f"kd={float(kd_loss.item()):.6f}",
                    flush=True,
                )

            valid_steps = max(1, len(train_loader) - skipped_steps)
            eval_student_epoch = student_train
            all_scores = evaluate_student(
                eval_student_epoch, val_dataset, device, args.roi_size, args.overlap,
                avail_mask=None, label=f"Epoch {epoch} val all-4",
            )
            drop_scores = evaluate_student(
                eval_student_epoch, val_dataset, device, args.roi_size, args.overlap,
                avail_mask=drop_mask, label=f"Epoch {epoch} val drop-T1ce",
            )
            all_mean = float(np.mean([all_scores["wt"], all_scores["tc"], all_scores["et"]]))
            primary = float(np.mean([drop_scores["tc"], drop_scores["et"]]))
            eligible = all_mean >= all_constraint
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            improved = eligible and primary > best_primary
            if improved:
                best_primary = primary
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint_atomic(best_path, {
                    "epoch": epoch,
                    "model": student_train.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_drop_t1ce_tc_et_mean": best_primary,
                    "all_baseline": all_baseline,
                    "all_constraint": all_constraint,
                    "all_scores": all_scores,
                    "drop_t1ce_scores": drop_scores,
                    "teacher_checkpoint": str(teacher_checkpoint),
                    "teacher_sha256": teacher_sha,
                    "teacher_output_order": TEACHER_OUTPUT_ORDER,
                    "teacher_modalities": TEACHER_MODALITIES,
                    "model_modalities": MODEL_MODALITIES,
                    "teacher_reorder": TEACHER_REORDER,
                    "args": vars(args),
                })
            else:
                stale_epochs += 1
            save_checkpoint_atomic(last_path, {
                "epoch": epoch,
                "model": student_train.state_dict(),
                "opt": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_drop_t1ce_tc_et_mean": best_primary,
                "best_epoch": best_epoch,
                "all_baseline": all_baseline,
                "all_constraint": all_constraint,
                "args": vars(args),
            })
            row = {
                "epoch": epoch,
                "train_loss": train_loss_total / valid_steps,
                "seg_loss": seg_loss_total / valid_steps,
                "kd_loss": kd_loss_total / valid_steps,
                "skipped_steps": skipped_steps,
                "all_dice_wt": all_scores["wt"],
                "all_dice_tc": all_scores["tc"],
                "all_dice_et": all_scores["et"],
                "all_mean_dice": all_mean,
                "drop_t1ce_dice_wt": drop_scores["wt"],
                "drop_t1ce_dice_tc": drop_scores["tc"],
                "drop_t1ce_dice_et": drop_scores["et"],
                "drop_t1ce_tc_et_mean": primary,
                "all_baseline": all_baseline,
                "all_constraint": all_constraint,
                "eligible_for_best": int(eligible),
                "best_drop_t1ce_tc_et_mean": best_primary,
                "learning_rate": current_lr,
                "epoch_time_sec": time.perf_counter() - started,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch}: train_loss={row['train_loss']:.6f} "
                f"all_mean={all_mean:.6f} drop_TC={drop_scores['tc']:.6f} "
                f"drop_ET={drop_scores['et']:.6f} primary={primary:.6f} "
                f"eligible={eligible} best_epoch={best_epoch} lr={current_lr:.9g}",
                flush=True,
            )
            if stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without eligible improvement.", flush=True)
                break

    print(f"KD training complete. Best checkpoint: {best_path}", flush=True)
    print(f"Training log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
