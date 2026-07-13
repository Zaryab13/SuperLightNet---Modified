# PC3 clean SegResNet teacher setup

This branch prepares the leakage-clean, cross-family SegResNet teacher. It starts from random initialization and reuses the clean Swin teacher's train/validation recipe. It uses the existing 1,000-case training and 125-case validation splits; the 126-case test dataset is never instantiated.

Do not run this job while the clean Swin teacher or any other CUDA job is active. The trainer checks GPU process memory and aborts with `GPU BUSY - not starting` if another process holds more than 1 GiB.

## 1. Pull the teacher-training branch

In Anaconda Prompt on PC3:

```bat
cd /d "D:\Ahmad Jawad\New folder\SuperLightNet---Modified"
git fetch origin
git switch teacher-training
git pull --ff-only origin teacher-training
```

## 2. Create the complete PC3 environment

The repository requirements pin Python-compatible PyTorch CUDA 12.4 wheels, MONAI 1.5.2, and the supporting libraries used by the teacher pipeline.

One complete Anaconda Prompt command:

```bat
conda create -n segresnet_teacher_clean python=3.11 -y && conda activate segresnet_teacher_clean && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt && python -c "import torch,monai,nibabel,pynvml; from monai.networks.nets import SegResNet; m=SegResNet(spatial_dims=3,in_channels=4,out_channels=3,init_filters=16,blocks_down=(1,2,2,4),blocks_up=(1,1,1),dropout_prob=0.2); n=sum(p.numel() for p in m.parameters() if p.requires_grad); print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available()); print('monai',monai.__version__); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print('segresnet_trainable_params',n); assert n==4702227"
```

Expected essentials:

```text
torch 2.6.0+cu124
monai 1.5.2
available True
gpu NVIDIA GeForce RTX 4090
segresnet_trainable_params 4702227
```

If the environment already exists, update and verify it with:

```bat
conda activate segresnet_teacher_clean
python -m pip install -r requirements.txt
python -c "import torch,monai; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),monai.__version__)"
```

## 3. Verify the BraTS manifest and data location

The manifest expects patient directories beneath:

```text
archive\BraTS2021_Training_Data\BraTS2021_XXXXX\
```

Run:

```bat
python -c "import json,pathlib,hashlib; p=pathlib.Path('splits/patient_splits.json'); m=json.loads(p.read_text()); root=(p.parent.parent/m['dataset_root']).resolve(); s=m['splits']; h=hashlib.sha256(p.read_bytes()).hexdigest(); print('manifest_sha256',h); print('root',root,'exists',root.is_dir()); print({k:len(v) for k,v in s.items()}); missing=[x for k in ('train','val') for x in s[k] if not (root/x).is_dir()]; print('missing_train_val_dirs',len(missing)); assert h=='9b411a68dc8ac8878d5e01f40b966d451d926c68afa6b960bf40a38422a55881'; assert (len(s['train']),len(s['val']),len(s['test']))==(1000,125,126); assert not missing"
```

The script checks test IDs only for disjointness. It never constructs a test dataset or opens test images.

## 4. Confirm the GPU is free

```bat
nvidia-smi
```

Wait until any existing Swin training or other CUDA process has ended. The SegResNet job must have the GPU to itself.

## 5. Start SegResNet training

```bat
conda activate segresnet_teacher_clean
cd /d "D:\Ahmad Jawad\New folder\SuperLightNet---Modified"
python -u scripts\train_segresnet_teacher_clean.py --num_workers 4 --val_workers 0
```

Startup must include:

```text
LEAKAGE-CHECK PASSED
Only the model differs from the Swin teacher recipe.
MONAI version: 1.5.2
Output channel order: (TC, WT, ET)
No pretrained weights loaded - training from random initialisation.
Trainable parameter count: 4702227
```

Outputs are isolated to:

```text
checkpoints\segresnet_teacher_clean\best.pth
checkpoints\segresnet_teacher_clean\last.pth
results\segresnet_teacher_clean\training_log.csv
```

Validation runs every five epochs and reports TC, WT, ET, and their mean in `training_log.csv`. `last.pth` is saved every epoch; `best.pth` is replaced whenever mean validation Dice improves. The run exits normally at 150 epochs or 66 accumulated hours, whichever occurs first.

## 6. Resume after a normal interruption

```bat
conda activate segresnet_teacher_clean
cd /d "D:\Ahmad Jawad\New folder\SuperLightNet---Modified"
python -u scripts\train_segresnet_teacher_clean.py --num_workers 4 --val_workers 0 --resume
```

Resume accepts only the isolated SegResNet `last.pth` with matching architecture, parameter count, seed, MONAI version, region order, and split-manifest provenance.

## 7. OOM fallback

Do not change `init_filters`. If the 128-cubed training patch raises CUDA OOM, restart from the last atomic checkpoint with a 96-cubed training patch:

```bat
python -u scripts\train_segresnet_teacher_clean.py --train_roi_size 96,96,96 --num_workers 4 --val_workers 0 --resume
```

Validation remains at 128 cubed with overlap 0.5; only the training patch changes.

## 8. Monitor progress

GPU utilization:

```bat
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 1
```

Training log from a second Anaconda Prompt:

```bat
conda activate segresnet_teacher_clean
cd /d "D:\Ahmad Jawad\New folder\SuperLightNet---Modified"
powershell -NoProfile -Command "Get-Content results\segresnet_teacher_clean\training_log.csv -Tail 10 -Wait"
```
