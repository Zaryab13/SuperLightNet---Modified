# MACAF-Swin-KD upload bundle

This folder contains the clearly named MACAF-Swin-KD files for Claude/manuscript drafting.

Use these files when referring to:

**MACAF-Swin-KD** = MACAF architecture trained/distilled with the Swin UNETR teacher.

Do NOT confuse with:

- `results/macaf/` = MACAF v1 / base MACAF / no-KD MACAF.
- `results/macaf_v2/` = MACAF v2 cosine/aggressive LR variant, rejected/degraded.
- `results/ablation_no_rmd/` = mean-fusion model with RMD disabled, not MACAF.
- `results/swin_kd_v1_sweep/` = mean-fusion student distilled from Swin, not MACAF.

The original sweep files were initially named `test_*_macaf.*`, which was confusing. The renamed files here use `test_*_macaf_swin_kd.*`.
