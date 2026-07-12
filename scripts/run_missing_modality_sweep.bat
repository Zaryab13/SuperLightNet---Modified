@echo off
setlocal
cd /d "%~dp0.."

rem Held-out test sweep for all 15 non-empty subsets of T1, T1CE, T2, FLAIR.
rem The evaluator expects one comma-separated --modalities argument.
rem Existing CSV or aggregate JSON outputs are skipped, never overwritten.

call :run_subset "t1" "t1"
call :run_subset "t1ce" "t1ce"
call :run_subset "t2" "t2"
call :run_subset "flair" "flair"
call :run_subset "t1_t1ce" "t1,t1ce"
call :run_subset "t1_t2" "t1,t2"
call :run_subset "t1_flair" "t1,flair"
call :run_subset "t1ce_t2" "t1ce,t2"
call :run_subset "t1ce_flair" "t1ce,flair"
call :run_subset "t2_flair" "t2,flair"
call :run_subset "t1_t1ce_t2" "t1,t1ce,t2"
call :run_subset "t1_t1ce_flair" "t1,t1ce,flair"
call :run_subset "t1_t2_flair" "t1,t2,flair"
call :run_subset "t1ce_t2_flair" "t1ce,t2,flair"
call :run_subset "t1_t1ce_t2_flair" "t1,t1ce,t2,flair"
exit /b 0

:run_subset
set "SUBSET=%~1"
set "MODALITIES=%~2"
set "OUTPUT_CSV=results\01_base_model\leakage_safe\test_%SUBSET%_patient_split.csv"
set "OUTPUT_JSON=results\01_base_model\leakage_safe\test_%SUBSET%_patient_split.json"

if exist "%OUTPUT_CSV%" (
    echo SKIP: "%OUTPUT_CSV%" already exists.
    exit /b 0
)
if exist "%OUTPUT_JSON%" (
    echo SKIP: "%OUTPUT_JSON%" already exists.
    exit /b 0
)

python scripts\evaluate_patient_split.py --split_json splits\patient_splits.json --split test --checkpoint checkpoints\leakage_safe\best_patient_split.pth --output_csv "%OUTPUT_CSV%" --device cuda --modalities "%MODALITIES%" --roi_size 160,160,160 --overlap 0.5
if errorlevel 1 exit /b %errorlevel%
exit /b 0
