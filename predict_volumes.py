from dataset import BraTSDataset
import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Cfg
from main import MultiEncoderRMDUNet  # your model class

# --- Config ---
ckpt_path = os.path.join(Cfg.ckpt_dir, "best.pth")  # change to latest.pth if needed
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

model = MultiEncoderRMDUNet(
    in_modalities=len(Cfg.modalities),
    base_ch=16,
    num_stages=4,
    num_classes=Cfg.num_classes,
    rmd_enable=Cfg.rmd_enable
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# --- Dataset for prediction ---
dataset = BraTSDataset(split="val", eval_mode=True)   # change split to 'test' if you want
print(f"Predicting on {len(dataset)} cases...")

def create_overlay(mri_vol, pred_mask, out_path, axis=-1):
    """Save a PNG with MRI slice and overlay side by side"""
    # pick middle slice
    slice_idx = mri_vol.shape[axis] // 2
    
    # extract slices
    if axis == 0:
        mri_slice = mri_vol[slice_idx, :, :]
        mask_slice = pred_mask[slice_idx, :, :]
    elif axis == 1:
        mri_slice = mri_vol[:, slice_idx, :]
        mask_slice = pred_mask[:, slice_idx, :]
    else:
        mri_slice = mri_vol[:, :, slice_idx]
        mask_slice = pred_mask[:, :, slice_idx]
    
    # normalize MRI
    mri_norm = (mri_slice - np.min(mri_slice)) / (np.max(mri_slice) - np.min(mri_slice) + 1e-8)
    
    # RGB MRI
    mri_rgb = np.stack([mri_norm]*3, axis=-1)
    
    # overlay mask
    overlay = mri_rgb.copy()
    overlay[mask_slice > 0] = [1, 0, 0]   # red tumor
    
    blended = (0.7*mri_rgb + 0.3*overlay)
    
    # plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(mri_rgb, cmap="gray")
    axs[0].set_title("Original MRI")
    axs[0].axis("off")
    
    axs[1].imshow(blended)
    axs[1].set_title("MRI + Segmentation")
    axs[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --- Predict ---
with torch.no_grad():
    for idx in tqdm(range(len(dataset))):
        case_id = dataset.ids[idx] if hasattr(dataset, "ids") else f"case_{idx:04d}"
        data, case_id = dataset[idx]
        data = data.unsqueeze(0).to(device)  # add batch dim

        logits = model(data)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save segmentation mask
        img_path = dataset.get_case_path(idx, modality="t1")  # adjust if dataset method differs
        ref_nii = nib.load(img_path)
        pred_nii = nib.Nifti1Image(pred, affine=ref_nii.affine, header=ref_nii.header)
        out_path = os.path.join(output_dir, f"{case_id}_pred.nii.gz")
        nib.save(pred_nii, out_path)

        # Save overlay visualization
        overlay_path = os.path.join(output_dir, f"{case_id}_overlay.png")
        mri_vol = ref_nii.get_fdata()
        create_overlay(mri_vol, pred, overlay_path)

print(f"✅ Predictions + overlays saved to: {output_dir}")
