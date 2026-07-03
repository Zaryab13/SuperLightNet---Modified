# evaluation.py
import os
import torch
import argparse

# import *directly* from your existing code
from main import Cfg, MultiEncoderRMDUNet, make_loaders, dice_per_region

def evaluate(ckpt_name="best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # we only need the val loader (your make_loaders already builds it with center-crops)
    _, val_loader = make_loaders()

    # build the same model definition you trained
    model = MultiEncoderRMDUNet(
        in_modalities=len(Cfg.modalities),
        base_ch=16,
        num_stages=4,
        num_classes=Cfg.num_classes,
        rmd_enable=Cfg.rmd_enable,
    ).to(device)

    # resolve checkpoint path and load weights
    ckpt_path = ckpt_name if os.path.isabs(ckpt_name) else os.path.join(Cfg.ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # disable Random Multi-View Drop at eval time
    model.rmd_enable = False

    # accumulate Dice over the whole val loader
    totals = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    n = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            scores = dice_per_region(logits, yb)
            for k in totals:
                totals[k] += scores[k]
            n += 1

    # mean dice
    for k in totals:
        totals[k] /= max(1, n)

    print(f"Val mean Dice — WT: {totals['WT']:.3f} | TC: {totals['TC']:.3f} | ET: {totals['ET']:.3f}")
    return totals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best.pth", help="best.pth or latest.pth or a full path")
    args = parser.parse_args()
    evaluate(args.ckpt)
