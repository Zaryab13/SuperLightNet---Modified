# Mean-fusion RMD-off ablation upload bundle

This folder contains the RMD-off mean-fusion ablation artifacts.

Use these files when referring to:

**Mean-fusion model with Random Multi-View Drop disabled (RMD-off)**

Do NOT confuse with:

- `results/01_base_model/leakage_safe_fixed/` = base mean-fusion RMD-on model under corrected evaluation.
- `results/02_kd_old_swin/swin_kd_v1_sweep/` = mean-fusion student distilled from the older Swin teacher.
- MACAF results live only on the `macaf` branch.

Checkpoint metadata confirmed `rmd_enable=False` for `checkpoints/ablation_no_rmd/best_patient_split.pth`.
