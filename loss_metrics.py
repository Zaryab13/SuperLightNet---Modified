import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    # same as in main.py
    ...

@torch.no_grad()
def dice_per_region(logits, target):
    # same as in main.py
    ...
