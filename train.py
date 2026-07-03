import os
import math
import time
import torch
from config import Cfg
from main import MultiEncoderRMDUNet, make_loaders as make_patient_loaders
from loss_metrics import DiceCELoss, dice_per_region

def make_loaders():
    return make_patient_loaders()

def train():
    os.makedirs(Cfg.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = make_loaders()
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

    for epoch in range(start_epoch, Cfg.epochs + 1):
        # ---- Training loop exactly as in main.py ----
        ...
