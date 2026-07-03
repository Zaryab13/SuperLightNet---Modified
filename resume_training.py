import os
import torch
from project_main import MultiEncoderRMDUNet, get_dataloaders  # get both model and dataloader

# --- Config ---
project_root = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.environ.get("BRATS_RESUME_CHECKPOINT", os.path.join(project_root, "checkpoints", "epoch_007.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1   # match original
num_epochs = 100 # total desired epochs

# --- Recreate model (match training settings) ---
model = MultiEncoderRMDUNet(
    in_modalities=4,      # BraTS modalities: T1, T1ce, T2, FLAIR
    base_ch=16,
    num_stages=4,
    num_classes=4,        # change to 3 if your original training used 3
    rmd_enable=True
)
model = model.to(device)

# --- Optimizer ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # match training

# --- Load checkpoint ---
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1

print(f"✅ Resumed from epoch {checkpoint['epoch']} → will start at {start_epoch}")

# --- Data ---
train_loader, val_loader = get_dataloaders(batch_size=batch_size)

# --- Continue training ---
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done.")

    # save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(project_root, "checkpoints", f"epoch_{epoch:03d}.pth"))
