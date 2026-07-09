# Results organization — MACAF branch

This branch is for MACAF-family experiments and results.

## Keep on `macaf` branch

- `results/macaf/`
  - MACAF v1 / base MACAF / no-KD MACAF.
  - Original tracked files keep historical names such as `test_all_macaf.csv`.
  - Clean upload bundle: `results/macaf/_upload_macaf_v1_no_kd_for_claude/`.

- `results/macaf_v2/`
  - MACAF v2 cosine/aggressive LR variant.
  - Rejected/degraded run; keep for audit/comparison, not as the main method.

- `results/macaf_swin_kd/`
  - MACAF-Swin-KD run.
  - Original tracked files may include historical names.
  - Clear renamed files use `_macaf_swin_kd` suffix.
  - Clean upload bundle: `results/macaf_swin_kd/_upload_macaf_swin_kd_for_claude/`.

- `figures/macaf_v1_training_curves.*` and `figures/macaf_v2_training_curves.*`
  - MACAF training visualizations.

## Keep on `main` branch, not here

- Mean-fusion base/RMD-on corrected eval results.
- Mean-fusion RMD-off ablation: `results/ablation_no_rmd/`.
- Mean-fusion Swin-KD: `results/swin_kd_v1/`, `results/swin_kd_v1_sweep/`.
- Swin teacher sanity: `results/swin_teacher_sanity/`.

This split prevents Claude/manuscript drafting from mixing MACAF, MACAF-Swin-KD, and mean-fusion KD results.
