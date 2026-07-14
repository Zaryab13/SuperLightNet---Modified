#!/usr/bin/env python3
"""Run the clean SegResNet teacher over all 15 non-empty modality subsets."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_segresnet_teacher_clean import check_gpu_idle  # noqa: E402

EXPECTED_MANIFEST_SHA256 = "9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segresnet_teacher_clean" / "best.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "segresnet_teacher_clean_sweep"
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_segresnet_teacher_clean.py"
SPLIT_JSON = PROJECT_ROOT / "splits" / "patient_splits.json"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (CHECKPOINT, EVALUATOR, SPLIT_JSON):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_sha256 = sha256(SPLIT_JSON)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split manifest SHA-256: {manifest_sha256}")

    targets = []
    for subset, _ in SUBSETS:
        csv_path = OUTPUT_DIR / f"test_{subset}_segresnet_teacher_clean.csv"
        json_path = csv_path.with_suffix(".json")
        if csv_path.exists() != json_path.exists():
            raise FileExistsError(f"Partial output pair exists: {csv_path} / {json_path}")
        targets.append((subset, csv_path, json_path))
    if all(csv_path.exists() for _, csv_path, _ in targets):
        print("All 15 SegResNet teacher subset outputs already exist; inference is not repeated.")
    else:
        check_gpu_idle(0)
        for index, (subset, modalities) in enumerate(SUBSETS, start=1):
            csv_path = OUTPUT_DIR / f"test_{subset}_segresnet_teacher_clean.csv"
            json_path = csv_path.with_suffix(".json")
            if csv_path.exists() and json_path.exists():
                print(f"SKIP complete subset {index}/15: {subset}", flush=True)
                continue
            print(f"START subset {index}/15: {subset} ({modalities})", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(EVALUATOR),
                    "--split_json", str(SPLIT_JSON),
                    "--split", "test",
                    "--checkpoint", str(CHECKPOINT),
                    "--output_csv", str(csv_path),
                    "--device", "cuda",
                    "--modalities", modalities,
                    "--roi_size", "128,128,128",
                    "--overlap", "0.5",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            print(f"DONE subset {index}/15: {subset}", flush=True)

    summary_rows = []
    for subset, modalities in SUBSETS:
        json_path = OUTPUT_DIR / f"test_{subset}_segresnet_teacher_clean.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload["case_count"] != 126:
            raise AssertionError(f"{subset} has {payload['case_count']} cases instead of 126")
        if payload["split_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise AssertionError(f"Manifest provenance mismatch for {subset}")
        statistics = payload["statistics"]
        summary_rows.append({
            "subset": subset,
            "modalities": modalities,
            "case_count": payload["case_count"],
            "dice_wt": statistics["dice_wt"]["mean"],
            "dice_tc": statistics["dice_tc"]["mean"],
            "dice_et": statistics["dice_et"]["mean"],
            "hd95_wt": statistics["hd95_wt"]["mean"],
            "hd95_tc": statistics["hd95_tc"]["mean"],
            "hd95_et": statistics["hd95_et"]["mean"],
        })

    summary_csv = OUTPUT_DIR / "segresnet_teacher_clean_15_subset_summary.csv"
    summary_json = summary_csv.with_suffix(".json")
    if not summary_csv.exists() and not summary_json.exists():
        with summary_csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        with summary_json.open("x", encoding="utf-8") as handle:
            json.dump({
                "checkpoint": str(CHECKPOINT.resolve()),
                "checkpoint_sha256": sha256(CHECKPOINT),
                "split_manifest_sha256": manifest_sha256,
                "subsets": summary_rows,
            }, handle, indent=2)
            handle.write("\n")
    elif summary_csv.exists() != summary_json.exists():
        raise FileExistsError("Partial sweep summary pair exists")

    print("subset,case_count,dice_wt,dice_tc,dice_et", flush=True)
    for row in summary_rows:
        print(
            f"{row['subset']},{row['case_count']},"
            f"{row['dice_wt']:.6f},{row['dice_tc']:.6f},{row['dice_et']:.6f}",
            flush=True,
        )
    print(f"Saved structured sweep to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
