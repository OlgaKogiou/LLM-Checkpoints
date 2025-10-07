# CheckFreq + CoorDL — Quick Setup (CUDA10-era container)

This README gets you from zero to a working **smoke test** run of CheckFreq using a small Imagenette dataset.  
It uses a known-good NVIDIA PyTorch container and the **native PyTorch dataloader** (no DALI required).

> If you already have an ImageNet-style dataset, you can mount it instead of downloading Imagenette. See §7.

---

## 0) Prerequisites (host)

- Linux with NVIDIA GPU + recent driver
- Docker + NVIDIA runtime (`nvidia-container-toolkit`)
- Sanity check:

```bash
# Check Ubuntu version
lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.5 LTS
Release:        22.04
Codename:       jammy
```

```bash
# Ubuntu 22.04 base (recommended)
sudo docker pull nvidia/cuda:11.8.0-base-ubuntu22.04
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Or Ubuntu 20.04
sudo docker pull nvidia/cuda:11.8.0-base-ubuntu20.04
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```
---

## 1) Create a workspace & clone the repos (host)

```bash
# Workspace root
mkdir -p ~/checkfreq_env && cd ~/checkfreq_env

# Code
git clone https://github.com/msr-fiddle/CheckFreq.git
git clone https://github.com/msr-fiddle/CoorDL.git
```

---

## 2) Put a small dataset in place (Imagenette via wget)

You can use any ImageNet-style dataset. For a fast smoke test, download **Imagenette** and unpack it under
`~/checkfreq_env/imagenet`. Inside the container this will appear at `/work/imagenet`.

```bash
# --- Imagenette (small demo dataset) -----------------------------
ROOT=~/checkfreq_env
mkdir -p "$ROOT/imagenet"

# Pick ONE size (uncomment the one you want):
IMAGENETTE_URL=https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz   # ~150MB, 320px
# IMAGENETTE_URL=https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz # ~50MB, 160px

TMPDIR=$(mktemp -d)
wget -q --show-progress -O "$TMPDIR/imagenette.tgz" "$IMAGENETTE_URL"
tar -xzf "$TMPDIR/imagenette.tgz" -C "$ROOT/imagenet" --strip-components=1
rm -rf "$TMPDIR"

# Quick sanity check on the host:
echo "[host] train classes:" && ls -1 "$ROOT/imagenet/train" | head
echo "[host] val classes:"   && ls -1 "$ROOT/imagenet/val"   | head
```

---

## 3) Start a known-good container (host)

We’ll use **NVIDIA PyTorch 19.05** (Python 3.6). This matches the environment we validated.

```bash
docker run --rm -it --gpus all --ipc=host   -v ~/checkfreq_env:/work   nvcr.io/nvidia/pytorch:19.05-py3 bash
```

Inside the container the workspace is mounted at `/work`.

---

## 4) Inside the container: basic prep

```bash
# Ensure we’re in the workspace
cd /work
ls -al CheckFreq CoorDL

# Optional: lightly update pip/wheel/setuptools
pip install --upgrade pip wheel setuptools
```

---

## 5) Run the smoke test (native dataloader; no DALI)

This is the simplest, reliable path. It runs **ResNet18** for **1 epoch** on Imagenette.

```bash
cd /work/CheckFreq

python -m torch.distributed.launch --nproc_per_node=4   models/image_classification/pytorch-imagenet-cf.py   -a resnet18   -b 128   --workers 3   --epochs 1   --deterministic   --noeval   --barrier   --checkfreq   --chk-prefix ./chk_smoke/   --data /work/imagenet | tee /work/run_checkfreq_imagenette.log
```

**Expected:** you’ll see “Using native dataloader”, epoch logs with `Loss`, `Prec@1/5`, and a final duration summary.

> If you don’t have 4 GPUs, reduce `--nproc_per_node` accordingly (e.g., `--nproc_per_node=1`).

---

## 6) (Optional) About DALI

DALI wheels are tied to specific CUDA versions and features differ between old releases.  
For the quick smoke test, the **native dataloader already works**. If you do want DALI later, ensure the DALI build
matches the container’s CUDA exactly; otherwise you may see missing ops (e.g., `ImageDecoderRandomCrop`).

---

## 7) Using your own dataset instead of Imagenette

Mount your dataset and point `--data` to it:

```bash
# Host: run a container and also mount your dataset
docker run --rm -it --gpus all --ipc=host   -v ~/checkfreq_env:/work   -v /path/to/your/imagenet:/data   nvcr.io/nvidia/pytorch:19.05-py3 bash

# In-container, run with:
python -m torch.distributed.launch --nproc_per_node=4   models/image_classification/pytorch-imagenet-cf.py   -a resnet18 -b 128 --workers 3 --epochs 1   --deterministic --noeval --barrier   --checkfreq --chk-prefix ./chk_smoke/   --data /data
```

Your dataset should be in **ImageNet-style** directory layout:
```
/path/to/your/imagenet/
  train/
    class1/  class2/  ...
  val/
    class1/  class2/  ...
```

---

## 8) Verify things **without deleting any Docker images** (host-safe)

```bash
# Confirm the image is present
docker images | grep -E 'nvcr.io/nvidia/pytorch\s+19\.05-py3'

# Prove GPUs are visible (no containers kept)
docker run --rm --gpus all nvcr.io/nvidia/pytorch:19.05-py3 nvidia-smi

# Inspect image layers (read-only)
docker history nvcr.io/nvidia/pytorch:19.05-py3

# See if related images are around
docker images | grep -E 'nvidia/dali|manylinux|pytorch'
```

These commands **do not** prune or delete anything.

---

## 9) Troubleshooting

- **OOM / too little GPU RAM**: lower `-b` (batch size) and/or `--nproc_per_node`.
- **Only 1 GPU available**: run with `--nproc_per_node=1`.
- **Slow dataloading**: raise/lower `--workers` (try 2–6) depending on CPU/IO.
- **DALI operator missing**: stick with the native dataloader, or install a DALI wheel that exactly matches CUDA in the container.
- **Dataset not found**: check mount and path; for Imagenette we expect `/work/imagenet/{train,val}` inside the container.

---

## References / Credits

- CheckFreq: <https://github.com/msr-fiddle/CheckFreq>
- CoorDL: <https://github.com/msr-fiddle/CoorDL>
- Imagenette by fast.ai: <https://github.com/fastai/imagenette>
