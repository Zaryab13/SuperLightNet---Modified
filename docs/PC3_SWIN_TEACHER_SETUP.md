# PC3 clean SwinUNETR teacher setup

This branch prepares the C9-C-AMENDED leakage-clean SwinUNETR teacher. The run starts from random initialization and uses only the existing 1,000-case train and 125-case validation splits. It never instantiates the 126-case test dataset.

## 1. Clone or switch to the teacher branch

In Anaconda Prompt on PC3:

```bat
cd /d D:\
git clone <YOUR_GITHUB_REPOSITORY_URL> "SuperLightNet - Modified"
cd /d "D:\SuperLightNet - Modified"
git fetch origin
git switch teacher-training
git pull --ff-only origin teacher-training
```

If the repository is already cloned:

```bat
cd /d "D:\path\to\SuperLightNet - Modified"
git fetch origin
git switch teacher-training
git pull --ff-only origin teacher-training
```

## 2. Create the PC3 environment

The pinned stack uses the official PyTorch 2.6.0 CUDA 12.4 wheels and MONAI 1.5.2. MONAI 1.5.2 is required because `SwinUNETR` no longer accepts `img_size`.

```bat
conda create -n swin_teacher_clean python=3.11 -y
conda activate swin_teacher_clean
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install monai==1.5.2 numpy==1.26.4 nibabel==5.3.2 scipy==1.15.2 einops==0.8.1 nvidia-ml-py==12.570.86
```

Verify the environment:

```bat
python -c "import torch, monai, nibabel, pynvml; print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available()); print('monai',monai.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected essentials:

```text
torch 2.6.0+cu124
monai 1.5.2
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## 3. Put the BraTS2021 data in the manifest location

The split manifest expects:

```text
archive\BraTS2021_Training_Data\BraTS2021_XXXXX\
```

Copy the dataset to that location without changing `splits/patient_splits.json`. Then verify the manifest, case directories, and split counts:

```bat
python -c "import json,pathlib,hashlib; p=pathlib.Path('splits/patient_splits.json'); m=json.loads(p.read_text()); root=(p.parent.parent/m['dataset_root']).resolve(); s=m['splits']; print('manifest_sha256',hashlib.sha256(p.read_bytes()).hexdigest()); print('root',root,'exists',root.is_dir()); print({k:len(v) for k,v in s.items()}); missing=[x for k in ('train','val') for x in s[k] if not (root/x).is_dir()]; print('missing_train_val_dirs',len(missing))"
```

Expected split counts:

```text
train=1000, val=125, test=126
missing_train_val_dirs=0
```

The script checks test IDs for disjointness but never constructs or opens test cases.

## 4. Confirm the GPU is free

```bat
nvidia-smi
```

Do not launch if another compute process holds more than 1 GiB. The training script repeats this check and exits with `GPU BUSY - not starting` before it creates output directories.

## 5. Start the run

Use unbuffered output so progress appears immediately:

```bat
python -u scripts\train_swin_teacher_clean.py --num_workers 4 --val_workers 0
```

The first startup must print:

```text
No pretrained weights loaded - training from random initialisation.
LEAKAGE-CHECK PASSED
Trainable parameter count: 15705621
Output channel order: (TC, WT, ET)
```

Outputs are isolated to:

```text
checkpoints\swin_teacher_clean\best.pth
checkpoints\swin_teacher_clean\last.pth
results\swin_teacher_clean\training_log.csv
```

## 6. Resume after a normal interruption

```bat
conda activate swin_teacher_clean
cd /d "D:\path\to\SuperLightNet - Modified"
python -u scripts\train_swin_teacher_clean.py --num_workers 4 --val_workers 0 --resume
```

Resume is accepted only from `checkpoints\swin_teacher_clean\last.pth` and only when its manifest, MONAI version, seed, architecture, and random-initialization provenance match.

## 7. OOM fallback

Do not reduce `feature_size`. If the first 128-cubed training attempt raises CUDA OOM, preserve the failed artifacts for diagnosis, move them out of the fixed output names, and restart from random initialization at 96 cubed:

```bat
ren checkpoints\swin_teacher_clean swin_teacher_clean_oom128
ren results\swin_teacher_clean swin_teacher_clean_oom128
python -u scripts\train_swin_teacher_clean.py --train_roi_size 96,96,96 --num_workers 4 --val_workers 0
```

Validation remains 128 cubed with 0.5 overlap. Only the training patch changes.

## 8. Monitor without modifying the run

```bat
nvidia-smi -l 10
```

In a second Anaconda Prompt:

```bat
conda activate swin_teacher_clean
cd /d "D:\path\to\SuperLightNet - Modified"
powershell -NoProfile -Command "Get-Content results\swin_teacher_clean\training_log.csv -Tail 10 -Wait"
```

The run stops cleanly after 150 epochs or 66 accumulated training hours, whichever occurs first. Validation runs every five epochs, `last.pth` is replaced atomically every epoch, and `best.pth` is replaced atomically whenever mean validation Dice improves.
