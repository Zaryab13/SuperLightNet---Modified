import os
import re
import zipfile
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import gradio as gr

from config import Cfg
from main import MultiEncoderRMDUNet  # ✅ correct import

# ---------------------------
# Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = os.path.join(Cfg.ckpt_dir, "best.pth")
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model = MultiEncoderRMDUNet(
    in_modalities=len(Cfg.modalities),
    base_ch=16,
    num_stages=4,
    num_classes=Cfg.num_classes,
    rmd_enable=Cfg.rmd_enable
).to(device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()
print("✅ Model loaded successfully")

# ---------------------------
# I/O helpers
# ---------------------------
def is_nii(path: str) -> bool:
    p = path.lower()
    return p.endswith(".nii") or p.endswith(".nii.gz")

def load_nifti(path: str):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine, nii.header

def norm_z(vol: np.ndarray) -> np.ndarray:
    # normalize only non-zero voxels (avoid background)
    mask = vol > 0
    if not mask.any():
        v = (vol - vol.mean()) / (vol.std() + 1e-8)
        return v.astype(np.float32)
    m = vol[mask].mean()
    s = vol[mask].std()
    v = vol.copy()
    v[mask] = (vol[mask] - m) / (s + 1e-8)
    return v.astype(np.float32)

def extract_zip_to_tmp(zip_path: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="patient_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir

def find_modalities_in_folder(folder: str):
    """
    Return dict with keys among {'t1','t1ce','t2','flair','seg'} -> filepath
    Case-insensitive, tolerant to names like '*_t1.nii.gz' or 'T1.nii'
    """
    found = {}
    if not os.path.isdir(folder):
        return found

    # collect all candidate nii files (top-level and one level deep)
    cands = []
    for root, _, files in os.walk(folder):
        for f in files:
            fp = os.path.join(root, f)
            if is_nii(fp):
                cands.append(fp)

    # priority match (avoid 't1' catching 't1ce')
    patterns = {
        "t1ce": re.compile(r"(?:^|[_\-])t1ce(?:[_\-\.]|$)", re.I),
        "flair": re.compile(r"(?:^|[_\-])flair(?:[_\-\.]|$)", re.I),
        "t2": re.compile(r"(?:^|[_\-])t2(?:[_\-\.]|$)", re.I),
        "t1": re.compile(r"(?:^|[_\-])t1(?:[_\-\.]|$)", re.I),
        "seg": re.compile(r"(?:^|[_\-])seg(?:[_\-\.]|$)|segmentation", re.I),
    }

    def match_first(key):
        if key in found:
            return
        pat = patterns[key]
        for fp in cands:
            if pat.search(os.path.basename(fp).lower()):
                found[key] = fp
                break

    for key in ["t1ce", "flair", "t2", "t1", "seg"]:
        match_first(key)
    return found

# ---------------------------
# Inference helpers
# ---------------------------
def prepare_input(mri_3d: np.ndarray, want_channels: int = 4) -> np.ndarray:
    """Repeat a single 3D volume to the expected channel count."""
    vol = np.expand_dims(mri_3d, axis=0)  # [1, D, H, W]
    vol = np.repeat(vol, want_channels, axis=0)  # [C, D, H, W]
    return vol

def prepare_multi_input(modals: dict) -> (np.ndarray, np.ndarray):
    """
    Load and stack present modalities in Cfg.modalities order.
    Returns stacked [C,D,H,W] and the background volume for display.
    """
    chosen_bg = None
    arrays = []
    bg_pref = ["flair", "t1ce", "t2", "t1"]

    # load preferred background if available
    for k in bg_pref:
        if k in modals:
            bg, _, _ = load_nifti(modals[k])
            chosen_bg = bg
            break

    # if none present, fall back to any present
    if chosen_bg is None:
        any_key = next((k for k in ["t1ce", "flair", "t2", "t1"] if k in modals), None)
        if any_key is not None:
            chosen_bg, _, _ = load_nifti(modals[any_key])

    # stack channels in configured order
    for m in Cfg.modalities:  # ["t1", "t1ce", "t2", "flair"]
        if m in modals:
            v, _, _ = load_nifti(modals[m])
            arrays.append(norm_z(v))
        else:
            # if missing, duplicate background (or zeros if none)
            if chosen_bg is not None:
                arrays.append(norm_z(chosen_bg))
            else:
                arrays.append(np.zeros_like(arrays[0]) if arrays else np.zeros((1, 1, 1), dtype=np.float32))

    vol = np.stack(arrays, axis=0)  # [C,D,H,W]
    if chosen_bg is None:
        chosen_bg = vol[0]
    return vol, chosen_bg

def run_model(volume_chw_dhw: np.ndarray) -> np.ndarray:
    """
    volume_chw_dhw: [C, D, H, W]
    returns label volume: [D, H, W] with classes in {0,1,2,3}
    """
    x = torch.from_numpy(volume_chw_dhw).unsqueeze(0).float().to(device)  # [1,C,D,H,W]
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred

# ---------------------------
# Visualization
# ---------------------------
def make_overlay_image(mri_vol: np.ndarray, label_vol: np.ndarray, slice_idx: int) -> np.ndarray:
    """
    Side-by-side: left = MRI, right = MRI + colored mask.
    label_map classes: 0=bg, 1,2,3 (tumor subregions)
    Colors: 1=green, 2=blue, 3=red
    Returns H x (2W) x 3 uint8.
    """
    # pick axial slice (D,H,W) -> [H,W] by last axis
    slice_idx = int(np.clip(slice_idx, 0, mri_vol.shape[-1] - 1))
    mri_slice = mri_vol[:, :, slice_idx]
    mask_slice = label_vol[:, :, slice_idx]

    # normalize MRI -> [0,1]
    mmin, mmax = float(mri_slice.min()), float(mri_slice.max())
    mri_norm = (mri_slice - mmin) / (mmax - mmin + 1e-8)
    base = np.stack([mri_norm]*3, axis=-1)  # RGB

    overlay = base.copy()
    overlay[mask_slice == 1] = [0.0, 1.0, 0.0]  # green
    overlay[mask_slice == 2] = [0.0, 0.0, 1.0]  # blue
    overlay[mask_slice == 3] = [1.0, 0.0, 0.0]  # red

    blended = 0.7 * base + 0.3 * overlay
    side_by_side = np.concatenate([(base * 255).astype(np.uint8),
                                   (blended * 255).astype(np.uint8)], axis=1)
    return side_by_side

# ---------------------------
# Metrics
# ---------------------------
def remap_brats_gt(gt: np.ndarray) -> np.ndarray:
    """Map BraTS GT labels {0,1,2,4} -> {0,1,2,3}."""
    out = np.zeros_like(gt, dtype=np.uint8)
    out[gt == 1] = 1
    out[gt == 2] = 2
    out[gt == 4] = 3
    return out

def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    den = a.sum() + b.sum() + 1e-8
    return float(2.0 * inter / den)

def compute_metrics(pred_labels: np.ndarray, gt_labels: np.ndarray) -> str:
    gt = remap_brats_gt(gt_labels.astype(np.uint8))
    txt = []
    # overall tumor
    d_all = dice_score(pred_labels > 0, gt > 0)
    txt.append(f"Dice (Any Tumor): {d_all:.3f}")
    # per-class dice
    for c, name in [(1, "Class1"), (2, "Class2"), (3, "Class3")]:
        d_c = dice_score(pred_labels == c, gt == c)
        txt.append(f"Dice ({name}={c}): {d_c:.3f}")
    return " | ".join(txt)

# ---------------------------
# Smart input handler
# ---------------------------
def resolve_input_paths(file_path: str):
    """
    Accepts:
      - .zip (extracts and reads modalities inside)
      - .nii / .nii.gz (single volume)
      - folder path (tries to find 4 modalities + optional seg)
    Returns: (mode, data)
      mode == "single": data = path_to_single_nii
      mode == "multi":  data = dict with keys among {'t1','t1ce','t2','flair','seg'}
    """
    if not file_path:
        raise ValueError("No input provided.")

    p = file_path
    if os.path.isfile(p) and zipfile.is_zipfile(p):
        folder = extract_zip_to_tmp(p)
        modals = find_modalities_in_folder(folder)
        if not modals:
            raise ValueError("No NIfTI files found inside the zip.")
        return "multi", modals

    if os.path.isdir(p):
        modals = find_modalities_in_folder(p)
        if not modals:
            raise ValueError("No NIfTI files found in the folder.")
        return "multi", modals

    if os.path.isfile(p) and is_nii(p):
        return "single", p

    raise ValueError("Unsupported input. Provide a .zip, a folder path, or a .nii/.nii.gz file.")

# ---------------------------
# Gradio callback
# ---------------------------
def run_smart(input_file_path, manual_path, slice_idx):
    try:
        # choose source
        src = (input_file_path or "").strip() if input_file_path else ""
        if not src and manual_path:
            src = manual_path.strip()
        if not src:
            return None, "❌ Please upload a file or enter a path.", None

        mode, data = resolve_input_paths(src)

        if mode == "single":
            vol3d, aff, hdr = load_nifti(data)
            vol3d = norm_z(vol3d)
            model_in = prepare_input(vol3d, want_channels=len(Cfg.modalities))
            bg_for_view = vol3d  # show this as the MRI background
        else:  # multi
            model_in, bg_for_view = prepare_multi_input(data)
            # keep an affine/header for saving predicted labels
            any_mod = next((data[k] for k in ["t1ce", "flair", "t2", "t1"] if k in data), None)
            _, aff, hdr = load_nifti(any_mod) if any_mod else (None, np.eye(4), None)

        pred_labels = run_model(model_in)  # [D,H,W]

        # Make visualization (axial slice)
        vis = make_overlay_image(bg_for_view, pred_labels, slice_idx=int(slice_idx))

        # Metrics if GT available (only in multi mode and if 'seg' present)
        metrics_txt = ""
        if mode == "multi" and "seg" in data:
            gt_vol, _, _ = load_nifti(data["seg"])
            metrics_txt = compute_metrics(pred_labels, gt_vol)

        # Save predicted label volume as NIfTI for download
        out_path = Path(tempfile.gettempdir()) / "pred_mask.nii.gz"
        nib.save(nib.Nifti1Image(pred_labels.astype(np.uint8), aff), str(out_path))

        return vis, metrics_txt, str(out_path)

    except Exception as e:
        import traceback
        return None, f"⚠ Error:\n{str(e)}\n{traceback.format_exc()}", None

# ---------------------------
# UI
# ---------------------------
with gr.Blocks(css="""
    body {background-color: #ffffff; color: #333; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .gr-block {background-color: #fefefe; border-radius: 12px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);}
    .gr-button {background-color: #4caf50; color: white; font-weight: bold; border-radius: 10px; font-size: 16px; padding: 12px;}
    .gr-button:hover {background-color: #45a049;}
    .gr-slider {accent-color: #ff5722; margin: 8px 0;}
    .gr-file {border: 2px dashed #2196f3; border-radius: 10px; padding: 12px; background-color: #e0f7fa;}
    .gr-textbox {background-color: #f0f0f0; color: #333; border-radius: 8px; font-size: 14px;}
    .gr-image {border: 3px solid #2196f3; border-radius: 10px;}
    h1 {color: #2a9d8f; text-align: center; font-size: 36px; background: linear-gradient(to right, #2a9d8f, #8fd3f4); -webkit-background-clip: text; color: transparent;}
""") as demo:
    gr.Markdown("<h1>🧠 3D-BrainScope: Tumor Segmentation Portal</h1>")
    gr.Markdown("Upload a **.zip** of a patient (T1/T1ce/T2/FLAIR [+ seg]) or a **single .nii/.nii.gz**, or paste a **folder/file path**.")

    with gr.Row():
        file_in = gr.File(
            label="Upload .zip / .nii / .nii.gz",
            file_types=[".zip", ".nii", ".nii.gz"],
            type="filepath"
        )
        path_in = gr.Textbox(label="Or enter folder/file path (e.g., D:\\cases\\BraTS\\case_0001)")

    slice_slider = gr.Slider(0, 127, value=60, step=1, label="Axial Slice Index (auto-clipped)")
    run_btn = gr.Button("Run Segmentation")

    overlay_output = gr.Image(label="MRI | MRI + Segmentation", type="numpy")
    metrics_output = gr.Textbox(label="Metrics (if seg available)", interactive=False)
    download_btn = gr.File(label="Download Predicted Mask (NIfTI)")

    run_btn.click(
        run_smart,
        inputs=[file_in, path_in, slice_slider],
        outputs=[overlay_output, metrics_output, download_btn]
    )

if __name__ == "__main__":
    demo.launch()
