"""Patient-level split manifest helpers."""

import json
import os
from typing import Dict, List, Optional


def case_id_from_path(path: str) -> str:
    """Return the BraTS case ID represented by a case directory or file path."""
    name = os.path.basename(os.path.normpath(path))
    for suffix in ("_t1ce.nii.gz", "_flair.nii.gz", "_t1.nii.gz", "_t2.nii.gz", "_seg.nii.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_split_manifest(path: str, fold: Optional[int] = None) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("mode") == "kfold":
        if fold is None:
            fold = 0
        folds = manifest.get("folds", [])
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"Fold {fold} is outside [0, {len(folds) - 1}]")
        splits = folds[fold]["splits"]
    else:
        splits = manifest["splits"]

    required = {"train", "val", "test"}
    if set(splits) != required:
        raise ValueError(f"Manifest splits must be exactly {sorted(required)}")
    normalized = {name: [str(case_id) for case_id in ids] for name, ids in splits.items()}
    assert_disjoint_splits(normalized)
    return normalized


def assert_disjoint_splits(splits: Dict[str, List[str]]) -> None:
    sets = {name: set(ids) for name, ids in splits.items()}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise AssertionError(f"Duplicate case ID inside {name} split")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise AssertionError(f"Patient leakage between {left} and {right}: {sorted(overlap)}")


def resolve_case_paths(dataset_root: str, case_ids: List[str]) -> List[str]:
    paths = []
    for case_id in case_ids:
        path = os.path.join(dataset_root, case_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Case directory from manifest not found: {path}")
        if case_id_from_path(path) != case_id:
            raise AssertionError(f"Case path {path} does not resolve to {case_id}")
        paths.append(path)
    return paths
