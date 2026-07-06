import os
import sys
import math
import time
import random
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from split_utils import load_split_manifest, resolve_case_paths

# ----------------------------
# ====== Configuration ========
# ----------------------------
class Cfg:
    # Paths (Windows-friendly). Change root to your dataset folder.
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.environ.get("BRATS_DATA_ROOT", os.path.join(project_root, "archive", "BraTS2021_Training_Data"))
    ckpt_dir  = os.environ.get("BRATS_CKPT_DIR", os.path.join(project_root, "checkpoints"))
    log_every = 1

    # Data
    modalities = ["t1", "t1ce", "t2", "flair"]
    patch_size = (128, 128, 128)   # training crop
    samples_per_volume = 2         # random tumor-biased samples per case per epoch
    tumor_bias = 0.7               # probability of sampling a tumor-centered patch
    intensity_norm = "z"           # "z" or None

    # Train
    num_classes = 4                # labels: 0,1,2,4 -> remapped to 0..3
    epochs = 100
    batch_size = 1                 # keep small for 3D
    lr = 1e-3
    weight_decay = 1e-5
    amp = True                     # mixed precision
    num_workers = 4
    split_manifest = os.environ.get("BRATS_SPLIT_MANIFEST", os.path.join(project_root, "splits", "patient_splits.json"))
    split_fold = int(os.environ.get("BRATS_SPLIT_FOLD", "0"))
    seed = 42

    # Random Multi-View Drop (RMD)
    rmd_enable = True
    rmd_min_keep = 2               # at least this many modalities kept
    rmd_prob = 0.7                 # with this prob we apply RMD for a batch

    # Inference (sliding window)
    sw_input_size = (160, 160, 160)  # slightly larger than patch, adjust to VRAM
    sw_overlap = 0.5

# ----------------------------
# ====== Reproducibility =====
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(Cfg.seed)

# ----------------------------
# ====== Data Utilities ======
# ----------------------------
def load_nifti(path: str) -> np.ndarray:
    arr = nib.load(path).get_fdata()
    return arr

def remap_seg(seg: np.ndarray) -> np.ndarray:
    """BraTS labels are {0,1,2,4}. Map to {0,1,2,3} for CE loss."""
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1   # NCR/NET
    out[seg == 2] = 2   # ED
    out[seg == 4] = 3   # ET
    return out

def zscore_norm(img: np.ndarray) -> np.ndarray:
    m = img[img > 0].mean() if (img > 0).any() else img.mean()
    s = img[img > 0].std() if (img > 0).any() else img.std()
    if s < 1e-8: 
        return img
    out = img.copy()
    mask = img > 0
    out[mask] = (img[mask] - m) / (s + 1e-8)
    return out

def bbox_from_mask(mask: np.ndarray, margin=8) -> Tuple[slice, slice, slice]:
    """Tight crop box around non-zero mask with margin."""
    coords = np.array(np.nonzero(mask))
    if coords.size == 0:
        # fallback to full
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)
    return slice(mins[0], maxs[0]), slice(mins[1], maxs[1]), slice(mins[2], maxs[2])

def crop_or_pad_3d(arr: np.ndarray, out_size: Tuple[int,int,int], center: Optional[Tuple[int,int,int]]=None) -> np.ndarray:
    """Crop or pad to out_size around center if provided, otherwise center crop."""
    D,H,W = arr.shape
    d,h,w = out_size

    if center is None:
        cz, cy, cx = D//2, H//2, W//2
    else:
        cz, cy, cx = center

    # compute start indices
    sz = max(0, min(D - d, cz - d//2))
    sy = max(0, min(H - h, cy - h//2))
    sx = max(0, min(W - w, cx - w//2))

    patch = arr[sz:sz+d, sy:sy+h, sx:sx+w]

    # pad if needed
    pdz = max(0, d - patch.shape[0])
    pdy = max(0, h - patch.shape[1])
    pdx = max(0, w - patch.shape[2])
    if pdz or pdy or pdx:
        patch = np.pad(patch, ((0,pdz),(0,pdy),(0,pdx)), mode='constant', constant_values=0)
    return patch

def choose_tumor_center(seg_remap: np.ndarray) -> Optional[Tuple[int,int,int]]:
    """Pick a voxel from any tumor label (1,2,3)."""
    idxs = np.argwhere(seg_remap > 0)
    if idxs.size == 0:
        return None
    z,y,x = idxs[np.random.randint(0, idxs.shape[0])]
    return int(z), int(y), int(x)

# ----------------------------
# ====== Dataset =============
# ----------------------------
class BraTSDataset(Dataset):
    def __init__(self, root: str, modalities: List[str], patch_size=(128,128,128),
                 samples_per_volume=1, tumor_bias=0.7, intensity_norm="z", train=True,
                 case_ids: Optional[List[str]] = None):
        self.root = root
        self.modalities = modalities
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.tumor_bias = tumor_bias
        self.intensity_norm = intensity_norm
        self.train = train

        if case_ids is None:
            raise ValueError("case_ids is required; datasets may not discover or split patients implicitly")
        self.case_ids = list(case_ids)
        if len(self.case_ids) != len(set(self.case_ids)):
            raise AssertionError("Duplicate patient IDs passed to BraTSDataset")
        self.cases = resolve_case_paths(root, self.case_ids)
        self.index = []  # (case_path, sample_id)
        for c in self.cases:
            for s in range(self.samples_per_volume if self.train else 1):
                self.index.append((c, s))

    def __len__(self):
        return len(self.index)

    def _load_case(self, case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        vols = []
        base = os.path.basename(case_dir)
        for m in self.modalities:
            path = os.path.join(case_dir, f"{base}_{m}.nii.gz")
            img = load_nifti(path)
            if self.intensity_norm == "z":
                img = zscore_norm(img)
            vols.append(img.astype(np.float32))

        segp = os.path.join(case_dir, f"{base}_seg.nii.gz")
        seg = load_nifti(segp)
        seg = remmap = remap_seg(seg).astype(np.int64)

        vols = np.stack(vols, axis=0)   # (C, D, H, W)
        return vols, remmap

    def __getitem__(self, i: int):
        case_dir, _ = self.index[i]
        vols, seg = self._load_case(case_dir)  # vols: (4, D,H,W), seg: (D,H,W)

        if self.train:
            # tumor-biased center or random center
            center = None
            if np.random.rand() < self.tumor_bias:
                c = choose_tumor_center(seg)
                if c is not None:
                    center = c
            # crop each channel and seg around same center
            C, D, H, W = vols.shape
            patch_vols = []
            for ci in range(C):
                patch_vols.append(crop_or_pad_3d(vols[ci], self.patch_size, center))
            x = np.stack(patch_vols, axis=0)  # (C, d,h,w)
            y = crop_or_pad_3d(seg, self.patch_size, center)
        else:
            # for validation we center-crop/pad to a fixed size (fast)
            C, D, H, W = vols.shape
            patch_vols = []
            for ci in range(C):
                patch_vols.append(crop_or_pad_3d(vols[ci], Cfg.sw_input_size, None))
            x = np.stack(patch_vols, axis=0)
            y = crop_or_pad_3d(seg, Cfg.sw_input_size, None)

        x = torch.from_numpy(x)                   # (C, d,h,w)
        y = torch.from_numpy(y).long()            # (d,h,w)
        return x, y

# ----------------------------
# ====== Model ===============
# ----------------------------
class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Encoder3d(nn.Module):
    def __init__(self, in_ch, chs: List[int]):
        super().__init__()
        layers = []
        c_prev = in_ch
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        for c in chs:
            self.stages.append(nn.Sequential(
                ConvBlock3d(c_prev, c),
                ConvBlock3d(c, c)
            ))
            self.downs.append(nn.MaxPool3d(2))
            c_prev = c
    def forward(self, x):
        feats = []
        for stage, down in zip(self.stages, self.downs):
            x = stage(x)
            feats.append(x)
            x = down(x)
        return feats, x  # list of skip features, and bottleneck input

class Bottleneck3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3d(in_ch, out_ch),
            ConvBlock3d(out_ch, out_ch),
        )
    def forward(self, x): return self.block(x)

class LearnableResidualSkip(nn.Module):
    """
    Learnable Residual Skip (LRS):
    - Combine decoder feature 'x' and encoder skip 's' using learnable scalars (or vectors).
    - y = x + w * s, where w is learnable per-channel scale.
    """
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, ch, 1, 1, 1))  # learnable scale for skip
    def forward(self, x, skip):
        return x + self.alpha * skip

class Decoder3d(nn.Module):
    def __init__(self, chs: List[int], out_ch: int, num_classes: int):
        super().__init__()
        # chs are encoder stage channels; we reverse for decoding
        rev = chs[::-1]
        self.upconvs = nn.ModuleList()
        self.fuse = nn.ModuleList()  # LRS per level
        self.blocks = nn.ModuleList()
        c_prev = rev[0]
        for c in rev:
            self.upconvs.append(nn.ConvTranspose3d(c_prev, c, kernel_size=2, stride=2))
            self.fuse.append(LearnableResidualSkip(c))
            self.blocks.append(nn.Sequential(
                ConvBlock3d(c, c),
                ConvBlock3d(c, c)
            ))
            c_prev = c
        self.head = nn.Conv3d(c_prev, num_classes, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips_agg: List[torch.Tensor]):
        """
        skips_agg: list of aggregated skips from all encoders at each level (same order as encoder forward).
        """
        x = bottleneck
        for i, (up, lrs, blk) in enumerate(zip(self.upconvs, self.fuse, self.blocks)):
            x = up(x)
            # match spatial size if off by one
            if x.shape[-1] != skips_agg[-1 - i].shape[-1]:
                # simple safe pad/crop matching
                ds, hs, ws = skips_agg[-1 - i].shape[-3:]
                x = F.interpolate(x, size=(ds, hs, ws), mode="trilinear", align_corners=False)
            x = lrs(x, skips_agg[-1 - i])
            x = blk(x)
        return self.head(x)


class ModalityAttentionFusion(nn.Module):
    """
    Modality-aware channel attention fusion.

    Applies a shared SE-style MLP to each available modality feature, normalizes
    attention logits across modalities per channel, then aggregates features by
    the learned weights.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.last_attention = None

    def forward(self, features: List[torch.Tensor], avail_mask: Optional[torch.Tensor] = None,
                like: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(features) == 0:
            if like is None:
                raise ValueError("Cannot infer zero-modality fusion shape without a reference tensor")
            return torch.zeros_like(like)
        if len(features) == 1 and avail_mask is None:
            self.last_attention = None
            return features[0]

        stacked = torch.stack(features, dim=0)  # (M, B, C, D, H, W)
        pooled = stacked.mean(dim=(3, 4, 5))    # (M, B, C)
        logits = self.mlp(pooled)               # (M, B, C)

        if avail_mask is not None:
            mask = avail_mask.to(device=stacked.device, dtype=torch.bool)
            if mask.ndim == 1:
                if mask.numel() != len(features):
                    raise ValueError("avail_mask length must match number of modality features")
                mask = mask[:, None, None]
            elif mask.ndim == 2:
                if mask.shape != (stacked.shape[1], len(features)):
                    raise ValueError("batched avail_mask must have shape (B, M)")
                mask = mask.transpose(0, 1)[:, :, None]
            else:
                raise ValueError("avail_mask must have shape (M,) or (B, M)")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=0).masked_fill(~mask, 0.0)
            normalizer = weights.sum(dim=0, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            weights = weights / normalizer
        else:
            weights = torch.softmax(logits, dim=0)

        self.last_attention = weights.detach()
        return (stacked * weights[..., None, None, None]).sum(dim=0)


class MultiEncoderRMDUNet(nn.Module):
    """
    Multi-encoder (one per modality) + Random Multi-View Drop (RMD) + LRS decoder.
    Encoders: separate small stacks per modality.
    RMD: during training, randomly drop K encoders (mask their outputs & skips).
    Aggregation: modality-aware channel attention fusion of available encoder features.
    """
    def __init__(self, in_modalities: int, base_ch=16, num_stages=4, num_classes=4,
                 rmd_enable=True, fusion_reduction: int = 4):
        super().__init__()
        self.in_modalities = in_modalities
        self.rmd_enable = rmd_enable

        # channels per stage
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8][:num_stages]

        # One encoder per modality
        self.encoders = nn.ModuleList([
            Encoder3d(1, chs) for _ in range(in_modalities)
        ])

        # bottleneck operates after modality-aware fusion of deepest encoder outputs
        self.bottleneck = Bottleneck3d(chs[-1], chs[-1])
        self.skip_attention = nn.ModuleList([
            ModalityAttentionFusion(ch, reduction=fusion_reduction) for ch in chs
        ])
        self.bott_attention = ModalityAttentionFusion(
            channels=base_ch * (2 ** (num_stages - 1)),
            reduction=fusion_reduction,
        )

        # decoder with LRS
        self.decoder = Decoder3d(chs, out_ch=chs[0], num_classes=num_classes)

    def _aggregate_skips(self, list_of_skips: List[List[torch.Tensor]], keep_mask: torch.Tensor):
        """
        list_of_skips: list over modalities -> list over levels -> tensor (B,C,D,H,W)
        keep_mask: (M,) or (B,M) bool tensor for modalities kept
        Return attention-fused modalities at each level.
        """
        M = len(list_of_skips)
        L = len(list_of_skips[0])
        agg = []
        for l in range(L):
            if keep_mask.ndim == 1:
                feats = [list_of_skips[m][l] for m in range(M) if keep_mask[m]]
                agg.append(self.skip_attention[l](feats, like=list_of_skips[0][l]))
            else:
                feats = [list_of_skips[m][l] for m in range(M)]
                agg.append(self.skip_attention[l](feats, avail_mask=keep_mask, like=list_of_skips[0][l]))
        return agg

    def _aggregate_skips_mean_unused(self, list_of_skips: List[List[torch.Tensor]], keep_mask: torch.Tensor):
        return self._aggregate_skips(list_of_skips, keep_mask)
        """
        list_of_skips: list over modalities -> list over levels -> tensor (B,C,D,H,W)
        keep_mask: (M,) bool tensor for modalities kept
        Return mean over kept modalities at each level.
        """
        M = len(list_of_skips)
        L = len(list_of_skips[0])
        agg = []
        for l in range(L):
            feats = []
            for m in range(M):
                if keep_mask[m]:
                    feats.append(list_of_skips[m][l])
            # mean aggregation; if all dropped, fallback to zeros (shouldn’t happen if we enforce min_keep)
            if len(feats) == 0:
                agg.append(torch.zeros_like(list_of_skips[0][l]))
            else:
                agg.append(torch.stack(feats, dim=0).mean(dim=0))
        return agg

    def forward(self, x: torch.Tensor, avail_mask: torch.Tensor = None):
        """
        x: (B, C=modalities, D, H, W)
        avail_mask: optional bool tensor, shape (C,) or (B,C), marking true modalities.
        """
        B, C, D, H, W = x.shape
        assert C == self.in_modalities

        device = x.device
        if avail_mask is not None:
            keep = avail_mask.to(device=device, dtype=torch.bool)
            if keep.ndim == 1:
                if keep.numel() != C:
                    raise ValueError("avail_mask with shape (C,) must match input modality count")
            elif keep.ndim == 2:
                if keep.shape != (B, C):
                    raise ValueError("avail_mask with shape (B,C) must match input batch and modality count")
            else:
                raise ValueError("avail_mask must have shape (C,) or (B,C)")
        elif self.training and self.rmd_enable and (random.random() < Cfg.rmd_prob):
            idx = list(range(C))
            random.shuffle(idx)
            k = random.randint(Cfg.rmd_min_keep, C)
            keep = torch.zeros(C, dtype=torch.bool, device=device)
            keep[idx[:k]] = True
        else:
            keep = torch.ones(C, dtype=torch.bool, device=device)

        all_skips = []
        bottlenecks = []
        for m in range(C):
            xm = x[:, m:m+1, ...]
            if keep.ndim == 1 and not keep[m]:
                xm = torch.zeros_like(xm)
            elif keep.ndim == 2:
                xm = xm * keep[:, m].view(B, 1, 1, 1, 1).to(dtype=xm.dtype)
            skips, bott_in = self.encoders[m](xm)
            all_skips.append(skips)
            bottlenecks.append(bott_in)

        if keep.ndim == 1:
            kept_bott = [bottlenecks[m] for m in range(C) if keep[m]]
            bott_fused = self.bott_attention(kept_bott, like=bottlenecks[0])
        else:
            kept_bott = [bottlenecks[m] for m in range(C)]
            bott_fused = self.bott_attention(kept_bott, avail_mask=keep, like=bottlenecks[0])

        agg_skips = self._aggregate_skips(all_skips, keep)
        z = self.bottleneck(bott_fused)
        logits = self.decoder(z, agg_skips)
        return logits

# ----------------------------
# ====== Loss & Metrics ======
# ----------------------------
class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight=None, dice_eps=1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.num_classes = num_classes
        self.eps = dice_eps

    def dice_term(self, logits, target):
        # one-hot target
        num_classes = logits.shape[1]
        with torch.no_grad():
            tgt = F.one_hot(target, num_classes=num_classes).permute(0,4,1,2,3).float()
        prob = torch.softmax(logits, dim=1)
        inter = (prob * tgt).sum(dim=(2,3,4))
        denom = (prob + tgt).sum(dim=(2,3,4)) + self.eps
        dice = (2*inter/denom).mean()  # mean over classes and batch
        return 1 - dice

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_term(logits, target)

@torch.no_grad()
def dice_per_region(logits, target) -> Dict[str, float]:
    """
    Compute BraTS-style region dice from argmax prediction:
      WT (whole tumor) = labels {1,2,3}
      TC (tumor core)  = labels {1,3}
      ET (enhancing)   = label  {3}
    """
    pred = torch.argmax(logits, dim=1)  # (B,d,h,w)
    def _dice(a, b):
        inter = (a & b).sum().item()
        den = a.sum().item() + b.sum().item()
        return (2*inter/den) if den > 0 else 1.0

    scores = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    B = pred.shape[0]
    for b in range(B):
        p = pred[b]
        t = target[b]
        p_wt = (p>0);              t_wt = (t>0)
        p_tc = (p==1)|(p==3);      t_tc = (t==1)|(t==3)
        p_et = (p==3);             t_et = (t==3)
        scores["WT"] += _dice(p_wt, t_wt)
        scores["TC"] += _dice(p_tc, t_tc)
        scores["ET"] += _dice(p_et, t_et)
    for k in scores:
        scores[k] /= B
    return scores

# ----------------------------
# ====== Training Loop =======
# ----------------------------
def make_loaders() -> Tuple[DataLoader, DataLoader]:
    splits = load_split_manifest(Cfg.split_manifest, Cfg.split_fold)
    train_set = BraTSDataset(
        root=Cfg.data_root,
        modalities=Cfg.modalities,
        patch_size=Cfg.patch_size,
        samples_per_volume=Cfg.samples_per_volume,
        tumor_bias=Cfg.tumor_bias,
        intensity_norm=Cfg.intensity_norm,
        train=True,
        case_ids=splits["train"],
    )
    val_set = BraTSDataset(
        root=Cfg.data_root,
        modalities=Cfg.modalities,
        patch_size=Cfg.patch_size,
        samples_per_volume=1,
        tumor_bias=0.0,
        intensity_norm=Cfg.intensity_norm,
        train=False,
        case_ids=splits["val"],
    )

    train_loader = DataLoader(train_set, batch_size=Cfg.batch_size, shuffle=True,
                              num_workers=Cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=Cfg.num_workers, pin_memory=True)
    print(f"Patients: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}; training patches={len(train_set)}")
    return train_loader, val_loader

def train():
    os.makedirs(Cfg.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Data loaders
    train_loader, val_loader = make_loaders()

    # Model, optimizer, loss
    model = MultiEncoderRMDUNet(
        in_modalities=len(Cfg.modalities),
        base_ch=16,
        num_stages=4,
        num_classes=Cfg.num_classes,
        rmd_enable=Cfg.rmd_enable
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=Cfg.amp)
    criterion = DiceCELoss(num_classes=Cfg.num_classes)

    # --- Resume from checkpoint if exists ---
    start_epoch = 1
    best_val = math.inf
    latest_ckpt = os.path.join(Cfg.ckpt_dir, "latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", math.inf)
        print(f"Resuming from epoch {start_epoch}")

    # ----- Training Loop -----
    for epoch in range(start_epoch, Cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=Cfg.amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if step % Cfg.log_every == 0:
                print(f"Epoch {epoch} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        all_scores = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=Cfg.amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item()
                scores = dice_per_region(logits, yb)
                for k in all_scores: all_scores[k] += scores[k]

        val_loss /= max(1, len(val_loader))
        for k in all_scores: all_scores[k] /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"[Epoch {epoch}/{Cfg.epochs}] Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"WT {all_scores['WT']:.3f} TC {all_scores['TC']:.3f} ET {all_scores['ET']:.3f} | {dt:.1f}s")

        # ----- Save latest checkpoint -----
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "best_val": best_val
        }, latest_ckpt)

        # ----- Save best checkpoint separately -----
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(Cfg.ckpt_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict()
            }, best_path)
            print(f"✓ Saved best to: {best_path}")

        # ----- Save epoch-specific checkpoint (optional) -----
        epoch_ckpt = os.path.join(Cfg.ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict()
        }, epoch_ckpt)


        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        all_scores = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=Cfg.amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item()
                scores = dice_per_region(logits, yb)
                for k in all_scores: all_scores[k] += scores[k]
        val_loss /= max(1, len(val_loader))
        for k in all_scores: all_scores[k] /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"[Epoch {epoch}/{Cfg.epochs}] Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"WT {all_scores['WT']:.3f} TC {all_scores['TC']:.3f} ET {all_scores['ET']:.3f} | {dt:.1f}s")

        # save ckpt
        ckpt_path = os.path.join(Cfg.ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict()}, ckpt_path)

        # best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(Cfg.ckpt_dir, "best.pth")
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict()}, best_path)
            print(f"✓ Saved best to: {best_path}")

# ======================
# DataLoader helper
# ======================
def get_dataloaders(batch_size=1):
    """
    Returns train_loader, val_loader for BraTS dataset.
    Adjust paths, transforms, and dataset class if needed.
    """
    from torch.utils.data import DataLoader
    # Import your dataset class here
    from dataset import BraTSDataset  # adjust if different

    # Replace these with your actual dataset paths
    return make_loaders()


if __name__ == "__main__":
    train()
