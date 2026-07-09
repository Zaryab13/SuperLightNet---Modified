# Results organization — main branch

This branch is for the mean-fusion MultiEncoderRMDUNet family and its KD/ablation results.

## Keep on `main`

- `results/leakage_safe_fixed/`
  - Base mean-fusion model with RMD enabled, evaluated with corrected availability-mask handling.

- `results/ablation_no_rmd/`
  - Mean-fusion ablation with RMD disabled.
  - Clean upload bundle: `results/ablation_no_rmd/_upload_ablation_no_rmd_for_claude/`.

- `results/swin_kd_v1/`
  - Cross-architecture KD training logs and selection metadata for mean-fusion student distilled from Swin UNETR.
  - Clean upload bundle: `results/swin_kd_v1/_upload_swin_kd_v1_for_claude/`.

- `results/swin_kd_v1_sweep/`
  - 15-subset test sweep for the mean-fusion Swin-KD student.

- `results/swin_teacher_sanity/`
  - Swin teacher all-modality test sanity evaluation. Use for provenance only; do not report teacher as leakage-clean baseline due to fold overlap.

- `results/paper_tables/` and `figures/`
  - Paper-ready tables and figures for mean-fusion/KD work.

## Keep on `macaf`, not here

- `results/macaf/` = MACAF v1 / base MACAF / no-KD MACAF.
- `results/macaf_v2/` = rejected/degraded MACAF v2.
- `results/macaf_swin_kd/` = MACAF-Swin-KD.

This split prevents MACAF and mean-fusion results from being mixed during manuscript drafting.
