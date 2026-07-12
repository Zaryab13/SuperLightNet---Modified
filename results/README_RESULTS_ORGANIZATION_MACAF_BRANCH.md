# MACAF results organization

All MACAF-family artifacts live under `results/macaf_experiments/`.

- `base/v1/`: original MACAF baseline, with separate `training/` and `sweep/` folders.
- `base/v2/`: cosine-decay MACAF v2 baseline and sweep.
- `base/v3/`: warmup/cosine/clipping MACAF v3 baseline and sweep.
- `kd_old_swin/`: MACAF distilled from the older Swin teacher.
- `self_kd/`: reserved for MACAF self-distillation; no artifacts are present on this branch.
- `kd_clean_swin/`: MACAF v1 distilled from the clean, test-safe Swin teacher.
- `teacher_sanity/`: clean Swin teacher sanity evaluation.

Canonical experiment folders use descriptive filenames containing the MACAF version,
training/sweep role, and KD type. Historical duplicate exports remain under nested
`archive_upload/` and `archive_legacy_sweep/` folders for auditability.

Mean-fusion results remain outside this hierarchy, for example `results/leakage_safe/`.
