
# Installing FlashAttention-2 for Tuo (MI300A / ROCm)

## Clone the Repository

```bash
git clone https://github.com/ROCm/flash-attention.git
```

## Enter the Repository

```bash
cd flash-attention
```

## Initialize Submodules

```bash
git submodule update --init --recursive
```

## Load ROCm Module

```bash
module load rocm/6.4.0
```

## Set ROCm Environment Variables

```bash
export ROCM_PATH=/opt/rocm-6.4.0
export ROCM_HOME=/opt/rocm-6.4.0
```

## Set GPU Architecture

For MI300A GPUs on Tuo:

```bash
export GPU_ARCHS="gfx942"
```

## Install FlashAttention-2

```bash
pip install . --no-build-isolation
```

---

# Verifying Installation

You can verify the installation with:

```bash
python -c "import flash_attn; print('FlashAttention installed successfully')"
```
