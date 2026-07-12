#!/usr/bin/env python3
"""Run the fixed 15-subset test sweep for the clean-teacher Swin-KD student."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import evaluate_patient_split as evaluation  # noqa: E402
from scripts.swin_kd_clean_A import (  # noqa: E402
    EXPECTED_SPLIT_SHA256,
    EXPECTED_TEACHER_SHA256,
    ensure_gpu_idle,
)

SUBSETS = (
    ("t1", "t1"),
    ("t1ce", "t1ce"),
    ("t2", "t2"),
    ("flair", "flair"),
    ("t1_t1ce", "t1,t1ce"),
    ("t1_t2", "t1,t2"),
    ("t1_flair", "t1,flair"),
    ("t1ce_t2", "t1ce,t2"),
    ("t1ce_flair", "t1ce,flair"),
    ("t2_flair", "t2,flair"),
    ("t1_t1ce_t2", "t1,t1ce,t2"),
    ("t1_t1ce_flair", "t1,t1ce,flair"),
    ("t1_t2_flair", "t1,t2,flair"),
    ("t1ce_t2_flair", "t1ce,t2,flair"),
    ("t1_t1ce_t2_flair", "t1,t1ce,t2,flair"),
)


def csv_means(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    means = {
        region: sum(float(row[f"dice_{region}"]) for row in rows) / len(rows)
        for region in ("wt", "tc", "et")
    }
    return len(rows), means


def main() -> None:
    split_json = PROJECT_ROOT / "splits" / "patient_splits.json"
    checkpoint = PROJECT_ROOT / "checkpoints" / "swin_kd_clean_A" / "best.pth"
    output_dir = PROJECT_ROOT / "results" / "04_kd_clean_swin" / "swin_kd_clean_A_sweep"
    manifest_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    if manifest_sha256 != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(
            f"Split manifest SHA mismatch: expected={EXPECTED_SPLIT_SHA256} "
            f"found={manifest_sha256}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if metadata.get("best_epoch") is None or metadata.get("best_drop_t1ce_tc_et") is None:
        raise RuntimeError("Selected checkpoint lacks the required selection metadata")
    if metadata.get("split_manifest_sha256") != manifest_sha256:
        raise RuntimeError("Training and sweep split manifest hashes differ")
    if metadata.get("teacher_checkpoint_sha256") != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("Selected student does not record the clean teacher SHA")
    if tuple(metadata.get("student_to_teacher_reorder", ())) != (0, 1, 2, 3):
        raise RuntimeError("Selected student records a scrambled clean-teacher reorder")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse sweep directory: {output_dir}")
    ensure_gpu_idle()
    output_dir.mkdir(parents=True, exist_ok=False)

    def validate_clean_output(output_csv: Path) -> Path:
        resolved = output_csv.resolve()
        if output_dir.resolve() not in resolved.parents or resolved.suffix.lower() != ".csv":
            raise ValueError(f"Output must be a CSV inside {output_dir}")
        output_json = resolved.with_suffix(".json")
        if resolved.exists() or output_json.exists():
            raise FileExistsError(f"Refusing to overwrite {resolved} / {output_json}")
        return output_json

    evaluation.validate_output_paths = validate_clean_output
    print(
        f"SWEEP checkpoint={checkpoint} best_epoch={metadata['best_epoch']} "
        f"best_drop_t1ce_tc_et={metadata['best_drop_t1ce_tc_et']} "
        f"manifest_sha256={manifest_sha256} teacher_sha256={EXPECTED_TEACHER_SHA256}",
        flush=True,
    )
    original_argv = sys.argv[:]
    try:
        for subset, modalities in SUBSETS:
            output_csv = output_dir / f"test_{subset}_swin_kd_cleanA.csv"
            sys.argv = [
                "evaluate_patient_split.py",
                "--split_json", str(split_json),
                "--split", "test",
                "--checkpoint", str(checkpoint),
                "--output_csv", str(output_csv),
                "--device", "cuda",
                "--modalities", modalities,
                "--roi_size", "160,160,160",
                "--overlap", "0.5",
            ]
            evaluation.main()
    finally:
        sys.argv = original_argv

    summary = {}
    for subset, _ in SUBSETS:
        path = output_dir / f"test_{subset}_swin_kd_cleanA.csv"
        count, means = csv_means(path)
        if count != 126:
            raise RuntimeError(f"Expected 126 cases for {subset}, found {count}")
        summary[subset] = {"case_count": count, **means}
    print("SWEEP_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    for subset in ("t1_t1ce_t2_flair", "t1_t2_flair"):
        values = summary[subset]
        print(
            f"TEST_RESULT subset={subset} n={values['case_count']} "
            f"WT={values['wt']:.12f} TC={values['tc']:.12f} ET={values['et']:.12f}",
            flush=True,
        )
    print("No existing files were modified.", flush=True)


if __name__ == "__main__":
    main()
