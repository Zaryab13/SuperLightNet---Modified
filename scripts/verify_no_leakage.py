#!/usr/bin/env python3
"""Verify split-level and sample-path-level patient isolation."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_utils import assert_disjoint_splits, case_id_from_path, load_split_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Defaults to the manifest path with a .csv suffix")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = load_split_manifest(str(args.manifest), args.fold)
    assert_disjoint_splits(splits)
    owner = {case_id: split for split, ids in splits.items() for case_id in ids}
    if len(owner) != sum(map(len, splits.values())):
        raise AssertionError("A case has more than one split owner")

    if args.dataset_root:
        root = args.dataset_root.resolve()
    else:
        stored_root = Path(raw["dataset_root"])
        root = (stored_root if stored_root.is_absolute()
                else args.manifest.resolve().parent.parent / stored_root).resolve()
    expected_modalities = ("t1", "t1ce", "t2", "flair", "seg")
    checked_paths = 0
    for split, case_ids in splits.items():
        for case_id in case_ids:
            case_dir = root / case_id
            if not case_dir.is_dir() or case_id_from_path(str(case_dir)) != case_id:
                raise AssertionError(f"Invalid {split} case directory: {case_dir}")
            for modality in expected_modalities:
                sample = case_dir / f"{case_id}_{modality}.nii.gz"
                if not sample.is_file():
                    raise FileNotFoundError(sample)
                resolved_id = case_id_from_path(str(sample))
                if resolved_id != case_id or owner.get(resolved_id) != split:
                    raise AssertionError(f"Sample assigned to incorrect split: {sample}")
                checked_paths += 1

    csv_path = args.csv or args.manifest.with_suffix(".csv")
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = [row for row in rows if raw.get("mode") != "kfold" or int(row["fold"]) == args.fold]
        for row in selected:
            if owner.get(row["case_id"]) != row["split"]:
                raise AssertionError(f"CSV/JSON split mismatch: {row}")

    train = set(splits["train"])
    forbidden = set(splits["val"]) | set(splits["test"])
    if train & forbidden:
        raise AssertionError("Validation/test case used during training")
    print(f"PASS: {len(owner)} unique cases and {checked_paths} sample paths; zero patient leakage.")


if __name__ == "__main__":
    main()
