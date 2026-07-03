#!/usr/bin/env python3
"""Create deterministic, leakage-free BraTS patient split manifests."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_utils import assert_disjoint_splits  # noqa: E402


def scan_case_ids(dataset_root: Path):
    case_ids = []
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("BraTS2021_"):
            continue
        required = [path / f"{path.name}_{mod}.nii.gz" for mod in ("t1", "t1ce", "t2", "flair", "seg")]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete case {path.name}; missing: {missing}")
        case_ids.append(path.name)
    if not case_ids:
        raise ValueError(f"No BraTS2021_* case directories found under {dataset_root}")
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("Duplicate patient/case IDs found")
    return case_ids


def holdout_split(case_ids, seed):
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }
    assert_disjoint_splits(splits)
    return splits


def kfold_splits(case_ids, kfolds, seed):
    if kfolds < 3:
        raise ValueError("--kfolds must be at least 3 to provide train, val, and test sets")
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    buckets = [shuffled[i::kfolds] for i in range(kfolds)]
    folds = []
    for fold in range(kfolds):
        test = set(buckets[fold])
        val = set(buckets[(fold + 1) % kfolds])
        train = set(shuffled) - test - val
        splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
        assert_disjoint_splits(splits)
        folds.append({"fold": fold, "splits": splits})
    return folds


def write_csv(path, manifest, dataset_root):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "split", "case_id", "case_path"])
        writer.writeheader()
        if manifest.get("mode") == "kfold":
            entries = manifest["folds"]
        else:
            entries = [{"fold": "", "splits": manifest["splits"]}]
        for entry in entries:
            for split, case_ids in entry["splits"].items():
                for case_id in case_ids:
                    writer.writerow({
                        "fold": entry["fold"], "split": split, "case_id": case_id,
                        "case_path": f"{case_id}",
                    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kfolds", type=int)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    case_ids = scan_case_ids(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    try:
        stored_root = root.relative_to(project_root).as_posix()
    except ValueError:
        stored_root = str(root)
    manifest = {"version": 1, "seed": args.seed, "dataset_root": stored_root, "case_count": len(case_ids)}
    if args.kfolds:
        manifest.update({"mode": "kfold", "kfolds": args.kfolds,
                         "folds": kfold_splits(case_ids, args.kfolds, args.seed)})
    else:
        manifest.update({"mode": "holdout", "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                         "splits": holdout_split(case_ids, args.seed)})

    stem = "patient_splits_5fold" if args.kfolds == 5 else "patient_splits"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, manifest, root)
    print(f"Wrote {json_path} and {csv_path} for {len(case_ids)} unique cases (seed={args.seed}).")


if __name__ == "__main__":
    main()
