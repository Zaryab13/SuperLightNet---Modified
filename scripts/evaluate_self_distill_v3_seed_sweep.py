#!/usr/bin/env python3
"""Run the fixed evaluation path over all 15 subsets for one v3 seed replicate."""

from __future__ import annotations

import argparse
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
from scripts.self_distill_posttrain_v3_seed import ensure_gpu_idle  # noqa: E402

EXPECTED_MANIFEST_SHA256 = "9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881"
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


def csv_means(path: Path) -> tuple[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    means = {
        region: sum(float(row[f"dice_{region}"]) for row in rows) / len(rows)
        for region in ("wt", "tc", "et")
    }
    return len(rows), means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(43, 44))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split_json = PROJECT_ROOT / "splits" / "patient_splits.json"
    checkpoint = (
        PROJECT_ROOT / "checkpoints" / f"self_distill_v3_seed{args.seed}" /
        f"best_self_distill_v3_seed{args.seed}.pth"
    )
    output_dir = PROJECT_ROOT / "results" / "03_self_kd" / f"self_distill_v3_seed{args.seed}_sweep"
    manifest_sha256 = hashlib.sha256(split_json.read_bytes()).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Split manifest SHA mismatch: expected {EXPECTED_MANIFEST_SHA256}, "
            f"found {manifest_sha256}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        metadata = torch.load(checkpoint, map_location="cpu")
    if int(metadata.get("seed", -1)) != args.seed:
        raise RuntimeError(f"Checkpoint seed mismatch: {metadata.get('seed')} != {args.seed}")
    if metadata.get("best_epoch") is None or metadata.get("best_drop_t1ce_tc_et") is None:
        raise RuntimeError("Checkpoint does not record the required selection metric")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse sweep directory: {output_dir}")
    ensure_gpu_idle()
    output_dir.mkdir(parents=True, exist_ok=False)

    def validate_seed_output(output_csv: Path) -> Path:
        resolved = output_csv.resolve()
        if output_dir.resolve() not in resolved.parents or resolved.suffix.lower() != ".csv":
            raise ValueError(f"Output must be a CSV inside {output_dir}")
        output_json = resolved.with_suffix(".json")
        if resolved.exists() or output_json.exists():
            raise FileExistsError(f"Refusing to overwrite {resolved} / {output_json}")
        return output_json

    evaluation.validate_output_paths = validate_seed_output
    print(
        f"SWEEP seed={args.seed} checkpoint={checkpoint} best_epoch={metadata['best_epoch']} "
        f"best_drop_t1ce_tc_et={metadata['best_drop_t1ce_tc_et']} "
        f"manifest_sha256={manifest_sha256}",
        flush=True,
    )
    original_argv = sys.argv[:]
    try:
        for subset, modalities in SUBSETS:
            output_csv = output_dir / f"test_{subset}_v3_seed{args.seed}.csv"
            sys.argv = [
                "evaluate_patient_split.py",
                "--split_json", str(split_json),
                "--split", "test",
                "--checkpoint", str(checkpoint),
                "--output_csv", str(output_csv),
                "--device", args.device,
                "--modalities", modalities,
                "--roi_size", "160,160,160",
                "--overlap", "0.5",
            ]
            evaluation.main()
    finally:
        sys.argv = original_argv

    summary = {}
    for subset, _ in SUBSETS:
        path = output_dir / f"test_{subset}_v3_seed{args.seed}.csv"
        count, means = csv_means(path)
        if count != 126:
            raise RuntimeError(f"Expected 126 cases for {subset}, found {count}")
        summary[subset] = {"case_count": count, **means}
    print("SWEEP_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    for subset in ("t1_t1ce_t2_flair", "t1_t2_flair"):
        values = summary[subset]
        print(
            f"SEED_RESULT seed={args.seed} subset={subset} n={values['case_count']} "
            f"WT={values['wt']:.12f} TC={values['tc']:.12f} ET={values['et']:.12f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
