# MACAF v1 / Base MACAF / No-KD upload bundle

This folder contains renamed copies of the MACAF v1 no-KD training logs and 15-subset missing-modality sweep results.

Why this folder exists:

- The original MACAF v1 files in `results/macaf/` were named like `test_all_macaf.csv`.
- The MACAF-Swin-KD files were also initially named with `_macaf`, which made Claude Pro confuse base MACAF v1 with MACAF-Swin-KD.
- To avoid collision, the MACAF v1 no-KD files were duplicated/renamed with explicit `_macaf_v1` names and gathered here.

Use these files when referring to:

**MACAF v1 / base MACAF / no-KD MACAF**

Do NOT confuse with:

- `results/macaf_v2/` = MACAF v2 cosine/aggressive LR variant, rejected/degraded.
- `results/macaf_swin_kd/` = MACAF with Swin-KD, distinct method.

Contents:

- `macaf_v1_training_log.csv` = epoch-level MACAF v1 training/validation log.
- `macaf_v1_batch_log.csv` = batch-level MACAF v1 training log.
- `macaf_v1_results_report_for_claude.txt` = existing report text for Claude.
- `test_*_macaf_v1.csv/json` = 15-subset missing-modality test sweep for MACAF v1, n=126.

Original files were not deleted from the project history; this is just a clean upload bundle.
