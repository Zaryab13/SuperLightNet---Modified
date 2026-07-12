# Mean-fusion Swin-KD upload bundle

This folder contains the cross-architecture Swin-KD artifacts for the mean-fusion MultiEncoderRMDUNet student.

Use these files when referring to:

**Mean-fusion student distilled from pretrained MONAI Swin UNETR teacher**

Do NOT confuse with:

- MACAF-Swin-KD, which lives on the `macaf` branch under `results/macaf_swin_kd/`.
- Self-distillation v3/v4/v5, which are earlier KD experiments.
- RMD-off ablation, under `results/01_base_model/ablation_no_rmd/`.

Contents:

- Root files: training/validation logs, sanity checks, selection status.
- `test_sweep_15_subsets/`: 15-subset per-case CSV/JSON test sweep for the Swin-KD student.
- `teacher_sanity/`: Swin teacher 126-case all-modality sanity evaluation. Important: the teacher fold overlaps our test IDs and should not be reported as a clean baseline.
