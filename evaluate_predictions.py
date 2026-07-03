import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ====== CONFIG ======
project_root = os.path.dirname(os.path.abspath(__file__))
pred_dir = os.environ.get("BRATS_OUTPUT_DIR", os.path.join(project_root, "outputs"))
gt_dir = os.environ.get("BRATS_DATA_ROOT", os.path.join(project_root, "archive", "BraTS2021_Training_Data"))

def dice_score(pred, gt):
    intersection = np.sum((pred > 0) & (gt > 0))
    return 2.0 * intersection / (np.sum(pred > 0) + np.sum(gt > 0) + 1e-8)

def sensitivity(pred, gt):
    tp = np.sum((pred > 0) & (gt > 0))
    fn = np.sum((pred == 0) & (gt > 0))
    return tp / (tp + fn + 1e-8)

def specificity(pred, gt):
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred > 0) & (gt == 0))
    return tn / (tn + fp + 1e-8)

results = []

for fname in tqdm(os.listdir(pred_dir)):
    if not fname.endswith("_pred.nii.gz"):
        continue

    case_id = fname.replace("_pred.nii.gz", "")
    pred_path = os.path.join(pred_dir, fname)
    gt_path = os.path.join(gt_dir, case_id, f"{case_id}_seg.nii.gz")

    if not os.path.exists(gt_path):
        print(f"[WARN] GT mask not found for {case_id}")
        continue

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()

    # Remap GT labels (BraTS: 1=necrosis, 2=edema, 4=enhancing tumor)
    gt = np.where(gt == 1, 1, gt)
    gt = np.where(gt == 2, 1, gt)
    gt = np.where(gt == 4, 1, gt)

    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    dice = dice_score(pred, gt)
    sens = sensitivity(pred, gt)
    spec = specificity(pred, gt)

    results.append([case_id, dice, sens, spec])

# Save results
df = pd.DataFrame(results, columns=["CaseID", "Dice", "Sensitivity", "Specificity"])
df.loc["Mean"] = df.mean(numeric_only=True)
df.to_csv("evaluation_results.csv", index=False)

print("✅ Evaluation complete. Saved to evaluation_results.csv")
