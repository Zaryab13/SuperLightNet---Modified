# SuperLightNet Modified: Project History

Last updated: 2026-07-06 (Asia/Karachi)

## Purpose and scope

This document records the development and experimental history of this repository from the
initial leakage-safe import through the current Random Multi-View Drop (RMD) ablation work.
It separates committed implementation milestones from generated experiment artifacts and
work that is still in progress.

The project performs multimodal 3D brain-tumor segmentation on BraTS 2021 MRI volumes. The
four input modalities are T1, T1ce, T2, and FLAIR. BraTS labels `{0, 1, 2, 4}` are remapped
internally to `{0, 1, 2, 3}`, where internal class `3` is enhancing tumor (ET).

## Original project baseline

The repository began as a lightweight PyTorch framework for patch-based 3D segmentation,
with a Gradio visualization application and a pretrained `checkpoints/best.pth`. The main
model evolved into `MultiEncoderRMDUNet`, which contains:

- one encoder per MRI modality;
- mean fusion of retained modality features at every encoder level and bottleneck;
- Random Multi-View Drop during training;
- a decoder with Learnable Residual Skip (LRS) fusion, `decoder + alpha * skip`;
- `BatchNorm3d` normalization, retained throughout the later experiments;
- a combined unweighted cross-entropy and uniformly averaged per-class Dice loss.

RMD is enabled by default in the main model. For a selected training batch, it shuffles all
modality indices uniformly and retains a random number of modalities between two and four.
No modality, including T1ce, receives special protection.

## Dataset and split design

The maintained split manifest is `splits/patient_splits.json`:

| Property | Value |
|---|---:|
| Manifest version | 1 |
| Seed | 42 |
| Dataset | BraTS2021 training data |
| Total patients | 1,251 |
| Training patients | 1,000 |
| Validation patients | 125 |
| Test patients | 126 |

The repository also contains deterministic five-fold manifests. The maintained holdout
pipeline enforces patient-level isolation. Before training, it verifies that train,
validation, and test IDs are disjoint, that instantiated datasets exactly match their
manifest partitions, and that no validation or test patient path appears in training data.

The test split is reserved for held-out reporting. However, missing-modality test results
were inspected during development; therefore, the test set is no longer pristine for
iterative model decisions. Subsequent model selection should use validation data only.

## Git development timeline

### 2026-07-04 01:10 — Initial leakage-safe project import

Commit `876f523` — `Initialize leakage-safe SuperLightNet project`

This commit established the repository baseline, including the original model and training
code, pretrained checkpoint, application, evaluation tools, patient split creation and
verification scripts, holdout and five-fold manifests, example results, and snapshots.

### 2026-07-04 01:50 — CUDA evaluation optimization

Commit `ab0e1e6` — `Optimize held-out evaluation for CUDA`

The held-out evaluation path was optimized for CUDA execution. Evaluation supports complete
patient volumes, selected modality subsets, per-patient CSV output, aggregate JSON output,
Dice, HD95, inference time, and peak GPU memory.

### 2026-07-04 01:50 — Leakage-safe training pipeline

Commit `07e21bd` — `Add leakage-safe patient split training pipeline`

This milestone added the maintained patient-isolated training stack:

- `src/superlightnet/patient_data.py` for manifest-selected datasets and deterministic patch
  generation;
- `src/superlightnet/training.py` for leakage assertions, sliding-window validation, regional
  Dice, and atomic checkpoint writes;
- `scripts/train_patient_split.py` as the command-line trainer;
- protected output locations under `checkpoints/leakage_safe/` and
  `results/01_base_model/leakage_safe/`.

Training uses ROI `(160, 160, 160)`, seed `42`, AdamW with learning rate `0.001`, and model
selection by the unweighted mean of validation WT, TC, and ET Dice. Validation uses complete
volumes with overlapping sliding windows.

### 2026-07-04 11:22 — Safe resume and batch logging

Commit `2621352` — `Add safe training resume and batch loss logs`

Resume support was strengthened by checking split-manifest SHA-256, train/validation split
names, ROI, seed, checkpoint epoch, and training-log epoch. Per-batch loss logging was added
alongside the per-epoch training log. Existing artifacts are protected from accidental
overwrite, and checkpoints are written atomically.

### 2026-07-04 14:00 — Missing-modality sweep tooling

Commit `9155149` — `Add held-out missing modality test sweep`

A batch script was added to evaluate all 15 non-empty subsets of the four MRI modalities.
Missing inputs are represented by zero-filled channels so that model architecture remains
unchanged.

### 2026-07-04 14:33 — Partial sweep forensic report

Commit `01aa7e3` — `Document partial missing modality sweep results`

The first sweep produced only the four single-modality evaluations. It was formally marked
invalid as a complete comparative ablation because the checkpoint changed during execution:
T1 used an older checkpoint, while T1ce, T2, and FLAIR used the epoch-68 best checkpoint.
The epoch-68 checkpoint had validation mean regional Dice `0.83187537770075` and SHA-256
`C24CB1CE804ABFBA205FC660BD762B520B949180D2E8D4E111BC2EE576FDD3C4`.

The observed single-modality results were retained as descriptive evidence only:

| Modality | WT Dice | TC Dice | ET Dice | Mean Dice |
|---|---:|---:|---:|---:|
| T1 | 0.001790 | 0.001674 | 0.023827 | 0.009097 |
| T1ce | 0.225594 | 0.231448 | 0.340580 | 0.265874 |
| T2 | 0.069787 | 0.021135 | 0.031757 | 0.040893 |
| FLAIR | 0.641425 | 0.028648 | 0.024247 | 0.231440 |

The detailed provenance warning and HD95 values are recorded in
`docs/missing_modality_sweep_partial_report.md`.

### 2026-07-04 17:40 — Training visualization notebook

Commit `386da82` — `Add inline training visualization notebook`

An inline notebook was added for inspecting training progress and metrics. A later local
modification to this notebook remains uncommitted at the time of this document update.

### 2026-07-05 01:38 — Configurable RMD ablation trainer

Commit `dbe2c4c` — `Add configurable RMD ablation training`

`scripts/train_ablation.py` was added as an isolated copy of the patient-split trainer. It
does not change `main.py` or the validated leakage-safe trainer. It adds:

- `--rmd_enable {true,false}`, defaulting to `true`;
- `--val_every`, defaulting to `1`;
- blank validation fields on epochs where validation is intentionally skipped;
- best-checkpoint selection using the same mean WT/TC/ET validation Dice as the original;
- early stopping after 15 validated epochs without improvement;
- an early failure rule when the first validation on or after epoch 15 has mean Dice below
  `0.40`;
- separate checkpoint and result subdirectories that reject the protected
  `leakage_safe` location;
- persisted RMD, validation-frequency, and early-stopping state for safe resume.

The `samples_per_patient` default was matched to the validated main-run checkpoint value of
`1`. Model architecture and `BatchNorm3d` were not changed.

## Main leakage-safe training outcome

The maintained main run completed 100 epochs. Its final logged epoch was:

| Epoch | Train loss | Validation loss | WT Dice | TC Dice | ET Dice |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.185162 | 0.306776 | 0.772274 | 0.782711 | 0.772294 |

The best checkpoint was selected on validation mean regional Dice rather than final-epoch
loss. Generated checkpoints and result files are intentionally excluded from normal Git
tracking.

## Held-out evaluation records

The repository currently contains generated held-out reports from more than one checkpoint
lineage. These must not be conflated.

### Legacy `checkpoints/best.pth`, all modalities

Across 126 held-out test patients:

| Metric | Mean |
|---|---:|
| WT Dice | 0.923560 |
| TC Dice | 0.873490 |
| ET Dice | 0.870356 |
| WT HD95 | 3.888 mm |
| TC HD95 | 9.866 mm |
| ET HD95 | 8.317 mm |
| Inference time | 0.526 s/patient |
| Peak allocated GPU memory | 1708.262 MB |

The tracked legacy checkpoint is currently absent from the working tree and appears as a
local deletion. This history document does not restore or modify it.

### Leakage-safe patient-split checkpoint, all modalities

Across the same 126-patient test split:

| Metric | Mean |
|---|---:|
| WT Dice | 0.898351 |
| TC Dice | 0.827448 |
| ET Dice | 0.839230 |
| WT HD95 | 5.840 mm |
| TC HD95 | 11.265 mm |
| ET HD95 | 6.545 mm |
| Inference time | 0.522 s/patient |
| Peak allocated GPU memory | 1708.262 MB |

The evaluation policy assigns Dice `1` and HD95 `0 mm` when prediction and target regions
are both empty. If exactly one is empty, Dice is `0` and HD95 is the physical image diagonal.

### Legacy best checkpoint with T1ce and FLAIR only

The recorded means were WT `0.772813`, TC `0.784421`, and ET `0.790259`. These results belong
to the legacy `checkpoints/best.pth` lineage, not necessarily the final leakage-safe model.

## RMD-off ablation status

An RMD-off run was launched separately with:

```bat
python -u scripts\train_ablation.py --split_json splits\patient_splits.json --train_split train --val_split val --output_dir checkpoints\ablation_no_rmd_manual --epochs 40 --val_every 5 --batch_size 1 --lr 0.001 --device cuda --roi_size 160,160,160 --num_workers 4 --val_workers 2 --seed 42 --samples_per_patient 1 --rmd_enable false
```

It writes checkpoints to `checkpoints/ablation_no_rmd_manual/` and logs to
`results/01_base_model/ablation_no_rmd_manual/`, leaving the validated pipeline untouched. Validation is
performed every five epochs, so validation columns are intentionally blank at other epochs.

At the captured snapshot, training had logged through epoch 13. The validated epochs were:

| Epoch | Train loss | Validation loss | WT Dice | TC Dice | ET Dice | Mean Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.253455 | 0.337776 | 0.770240 | 0.758333 | 0.740355 | 0.756310 |
| 10 | 0.218891 | 0.312215 | 0.811224 | 0.781936 | 0.767423 | 0.786861 |

This run was ongoing when the snapshot was taken; the table is not a final result.

An earlier interrupted attempt wrote partial logs under `results/01_base_model/ablation_no_rmd/`. Those
artifacts are not the canonical manual run and remain untracked.

## Class-imbalance diagnostic

All segmentation voxels from the 1,000 training-split patients were counted directly from
their NIfTI label volumes. The aggregate distribution was:

| BraTS label | Meaning | Voxels | Percent of total |
|---:|---|---:|---:|
| 0 | Background | 8,831,847,367 | 98.923021583781% |
| 1 | Necrotic/non-enhancing tumor | 14,177,714 | 0.158800560036% |
| 2 | Edema | 60,653,914 | 0.679367316308% |
| 4 | Enhancing tumor | 21,321,005 | 0.238810539875% |

The raw background:edema:necrotic:enhancing ratio is
`8831847367:60653914:14177714:21321005`. Normalized to enhancing tumor, it is
`414.232226248:2.844796200:0.664964621:1.000000000`.

The current patch sampler selects `target > 0` voxels for tumor-biased centers. It therefore
samples from all tumor labels and does not explicitly favor ET. Because edema contains about
2.845 times as many voxels as ET, uniformly drawing from all foreground voxels makes edema
substantially more likely to provide the patch center.

## Missing-T1ce failure and planned ET-biased experiment

Missing-T1ce evaluation was reported to collapse ET and TC Dice to approximately `0.03` and
`0.04`. The working diagnosis is that the all-foreground patch sampler under-samples ET and
does not sufficiently teach a T1ce-absent fallback.

A separate experiment was planned to add an
`enhancing_sampling_probability` parameter, defaulting to `0.0`, while preserving all
existing defaults. The intended sampling priority is:

1. with probability `0.3`, center on internal ET label `3`;
2. with probability `0.4`, center on any foreground voxel (`target > 0`);
3. otherwise, use a random center;
4. if a case has no ET voxels, fall back safely to any foreground voxel.

The intended run writes only to a new `runs/et_sampling/` location. Per explicit project
coordination, that change and run are being handled elsewhere/on another GPU. They were not
implemented or launched in this working tree as part of the RMD-off work documented here.

## Evaluation-path correction and self-distillation

The missing-modality investigation found that evaluation zero-filled absent input channels
but did not exclude their encoder outputs from mean fusion. Training-time RMD did exclude
dropped encoders. `MultiEncoderRMDUNet.forward()` now accepts an optional per-case
`avail_mask`, and `scripts/evaluate_patient_split.py` passes that mask through every sliding
window. The original training-time RMD branch is unchanged, and an all-true mask preserves
the full-modality path.

The corrected original-checkpoint sweep covers all 15 non-empty modality subsets under
`results/01_base_model/leakage_safe_fixed/`. A self-distillation series then used the original leakage-safe
checkpoint as both the frozen full-modality teacher and initial student:

- v1 recovered missing-T1ce performance but catastrophically forgot the all-modality path;
- v2 added full-modality anchoring and frozen BatchNorm statistics but encountered fp16 NaNs;
- v3 used BF16 forwards, FP32 KD/feature losses, Smooth-L1 feature matching, fully frozen
  BatchNorm, gradient clipping, and non-finite guards. It completed 30 epochs with zero
  skipped/non-finite steps;
- selected v3 epoch 30 preserved full-modality performance and produced held-out missing-T1ce
  Dice WT `0.8874`, TC `0.5997`, and ET `0.4651`;
- the complete post-KD 15-subset sweep is in `results/03_self_kd/self_distill_v3_sweep/`, with the
  three-era comparison in `results/05_analysis/paper_tables/missing_modality_master_table.csv`.

`scripts/self_distill_posttrain_v4.py` is the current specialization experiment. It starts
fresh from the original teacher, uses a 30/50/20 full/drop-T1ce/random mask distribution,
80 epochs, patience 20, cosine LR `5e-5` to `1e-6`, temperature `4.0`, and feature weight
`2.0`. The v4 process was active when this handoff was committed, so its changing runtime
logs and checkpoints were intentionally not committed.

## Current working-tree state at this snapshot

- `checkpoints/best.pth` was removed from the maintained tree; experiment checkpoints remain
  excluded from Git.
- The RMD-off run completed and its five common-subset evaluations are recorded under
  `results/01_base_model/ablation_no_rmd/`.
- RMD-on has complete 15-subset evaluation tables; RMD-off currently has only all-modality
  plus four single-drop evaluations.
- v4 runtime outputs remain local and in progress. Do not clean, overwrite, or treat them as
  final until the process completes.

## Reproducibility and provenance rules established by the project

- Keep patient splits fixed and verify isolation before training.
- Use seed `42` and ROI `(160, 160, 160)` for comparable maintained runs.
- Select checkpoints with validation metrics only; do not tune against test results.
- Freeze and hash a checkpoint before a modality sweep.
- Never compare subset evaluations produced from different checkpoint versions.
- Preserve model architecture and normalization when isolating an RMD ablation.
- Write each experiment to a new checkpoint and results directory.
- Keep generated checkpoints, logs, datasets, and NIfTI volumes out of Git.
- Record empty-region metric policy alongside Dice and HD95 results.
- Treat ongoing-run logs as snapshots, not final conclusions.

## Key files

| Path | Role |
|---|---|
| `main.py` | MultiEncoderRMDUNet, RMD, LRS decoder, and Dice+CE loss |
| `src/superlightnet/patient_data.py` | Patient-isolated loading, normalization, label remapping, and patch sampling |
| `src/superlightnet/training.py` | Sliding-window validation, metrics, checkpoints, and leakage assertions |
| `scripts/train_patient_split.py` | Maintained leakage-safe training entry point |
| `scripts/train_ablation.py` | Isolated configurable RMD ablation trainer |
| `scripts/evaluate_patient_split.py` | Held-out full-volume evaluation |
| `scripts/self_distill_posttrain_v3.py` | Stable completed self-distillation trainer |
| `scripts/self_distill_posttrain_v4.py` | Active drop-T1ce-specialized KD trainer |
| `scripts/run_missing_modality_sweep.bat` | Missing-modality subset sweep |
| `splits/patient_splits.json` | Fixed holdout split manifest |
| `docs/missing_modality_sweep_partial_report.md` | Provenance-aware partial sweep report |
| `notebooks/training_visualizations.ipynb` | Training-log visualization |
