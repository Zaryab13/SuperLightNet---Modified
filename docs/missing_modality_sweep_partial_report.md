# Missing-Modality Test Sweep: Partial Forensic Report

## Status

The requested sweep contains 15 non-empty subsets of `{t1, t1ce, t2, flair}`.
Only `t1`, `t1ce`, `t2`, and `flair` completed. The remaining 11 subset
evaluations were not produced. Each completed CSV contains exactly 126 rows for
126 unique held-out test case IDs.

## Critical checkpoint-provenance limitation

This partial sweep is **not a valid 15-way ablation comparison**. Training was
still running, and `best_patient_split.pth` changed at 2026-07-04 14:14:59. The
T1 evaluation began before that update; the other single-modality evaluations
began afterward. T1 therefore used an older model, while T1CE, T2, and FLAIR
used the epoch-68 best checkpoint.

Epoch-68 checkpoint metadata:

- Mean validation region Dice: `0.83187537770075`
- SHA-256: `C24CB1CE804ABFBA205FC660BD762B520B949180D2E8D4E111BC2EE576FDD3C4`

Do not use this partial sweep for formal subset ranking, statistical claims, or
model selection.

## Observed aggregate results

Means over 126 unique test patients; HD95 is in millimetres. `Mean Dice` is the
unweighted mean of WT, TC, and ET Dice.

| Modalities | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET | Mean Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.0017898976083320908 | 0.0016735507946039167 | 0.023826748153472035 | 68.01015614866004 | 82.94382976833138 | 327.92555446482726 | 0.009096732185469348 |
| T1CE | 0.22559398244547185 | 0.23144843545475896 | 0.34058015306985295 | 74.61937652640137 | 86.38761063503057 | 92.06713705772341 | 0.26587419032336124 |
| T2 | 0.06978735445455399 | 0.021135226428599465 | 0.03175716510133749 | 53.7829704826621 | 86.96309393531294 | 337.6561334926127 | 0.04089324866149698 |
| FLAIR | 0.6414254905903245 | 0.02864768890967954 | 0.024247287304037215 | 58.02514928931779 | 62.29249674547849 | 269.343113845777 | 0.2314401556013471 |

Mean inference time was `0.8590024134887028`–`0.9994733388884924` seconds per
patient. Mean peak allocated GPU memory was `1708.261707124256` MB.

## Descriptive interpretation only

- FLAIR preserved whole-tumour segmentation best (`Dice WT =
  0.6414254905903245`).
- T1CE preserved enhancing tumour best (`Dice ET = 0.34058015306985295`) and
  had the highest observed mean region Dice.
- T1 and T2 alone performed poorly in these runs.
- Large ET HD95 values often reflect the documented empty-mask penalty: when
  exactly one ET mask is empty, HD95 is the physical image diagonal.
- T1 must not be compared directly with the other three because its checkpoint
  differs.

## Required procedure for a valid final sweep

1. Finish training and select the final best checkpoint using validation only.
2. Copy it to an immutable, uniquely named checkpoint.
3. Record its SHA-256 and epoch.
4. Run all 15 subsets against that exact frozen file.
5. Record checkpoint SHA-256 and epoch in every aggregate JSON.
6. Do not alter training or select a model using these test results.

Because test results were inspected during development, the test set is no
longer pristine for iterative decisions. Treat a final sweep as fixed reporting
only and use validation data for further model choices.

## Source files

- `results/leakage_safe/test_<subset>_patient_split.csv`
- `results/leakage_safe/test_<subset>_patient_split.json`
- `results/leakage_safe/missing_modality_sweep.log`
- `scripts/run_missing_modality_sweep.bat`
