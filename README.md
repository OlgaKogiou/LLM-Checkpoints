CheckFreq — Adaptive Checkpointing for PyTorch (BLOOM-3B example)

This repo contains a minimal CheckFreq setup:

cf_checkpoint.py – snapshot/serialize/restore logic

cf_manager.py – orchestration (MANUAL/AUTO), async persist (process/thread)

cf_iterator.py – iterator wrapper that decides when to checkpoint

disk_bw.py – optional storage bandwidth probe (non-critical)

telemetry.py – CSV logging for checkpoint timing

input_data.txt – tiny toy corpus (you can replace with your own)

Example training script used below: models/nlp/bloom_cf.py (paths in commands assume the layout used in our logs).

1) Quick start with Docker

You need an NVIDIA GPU machine with Docker + nvidia-container-toolkit.

# Get a PyTorch CUDA image (any recent nvidia/pytorch tag works)
docker pull nvcr.io/nvidia/pytorch:24.06-py3


Run the container and mount this repo plus cache/output folders:

# From the repo root on your host:
mkdir -p hf_cache chk_bloom

docker run --gpus all --rm -it \
  -v $PWD:/work/CheckFreq \
  -v $PWD/hf_cache:/work/hf_cache \
  -v $PWD/chk_bloom:/work/chk_bloom \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:24.06-py3 bash


Inside the container:

# Packages (no internet? pre-populate hf_cache on host)
pip install -U transformers accelerate datasets safetensors --no-cache-dir

# Recommended env
export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/CheckFreq:/work/CheckFreq/src:$PYTHONPATH
# Use all visible GPUs
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))
# Prefer threads for async checkpointing under DDP
export CF_USE_THREAD=1

2) Data

Use the provided toy file or mount your own:

# toy data included in repo
ls -lh /work/CheckFreq/input_data.txt


If you have a different file, pass --train-file /path/to/your.txt in the run command.

3) MANUAL mode (fixed interval)

Choose a frequency (in steps) and run with --manual-freq > 0.

torchrun --nproc_per_node=$(python -c 'import torch; print(torch.cuda.device_count())') \
  /work/CheckFreq/models/nlp/bloom_cf.py \
  --model bigscience/bloom-3b \
  --train-file /work/CheckFreq/input_data.txt \
  --seq-len 256 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 1 \
  --workers 2 \
  --lr 2e-5 \
  --chk-prefix /work/chk_bloom \
  --manual-freq 4 \        # <— checkpoint every 4 steps
  --arch-name bloom3b | tee /work/chk_bloom/run.log


What you should see:

Console prints like MUST CHECKPOINT NOW AT ITER ….

Files under /work/chk_bloom/ such as lm_v_0_sync.chk, lm_v_1_sync.chk, and an epoch/ subdir when epoch boundary is reached.

Telemetry CSV: /work/chk_bloom/cf_telem_rank*.csv with cpu_disk_write and engine_wait rows.

4) AUTO mode (adaptive)

In AUTO, the iterator profiles iteration time/memory, then chooses a checkpoint cadence. Enable by passing --manual-freq 0.

Tip: The built-in profiler targets ~100 iterations. If your epoch is short, consider using more epochs or a larger dataset so AUTO can complete profiling before it picks a frequency. (It will still save at epoch end either way.)

# clear any prior cache so AUTO re-profiles
rm -f /.cache_* ./.STR_BW /work/CheckFreq/.cache_* 2>/dev/null || true

torchrun --nproc_per_node=$(python -c 'import torch; print(torch.cuda.device_count())') \
  /work/CheckFreq/models/nlp/bloom_cf.py \
  --model bigscience/bloom-3b \
  --train-file /work/CheckFreq/input_data.txt \
  --seq-len 256 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --epochs 2 \               # <— give AUTO enough steps to profile
  --workers 2 \
  --lr 2e-5 \
  --chk-prefix /work/chk_bloom \
  --manual-freq 0 \          # <— AUTO
  --arch-name bloom3b | tee -a /work/chk_bloom/run.log


What you should see:

PROFILE step … messages during the first ~100 steps.

A line like Chosen freq = X, percent_ov=Y%.

Async messages: [... START ASYNC] / [... END ASYNC] when checkpoints persist in the background.

Telemetry appended in /work/chk_bloom/cf_telem_rank*.csv.

5) Verifying outputs
# Checkpoint files
ls -lh /work/chk_bloom
ls -lh /work/chk_bloom/epoch

# Telemetry rows (not just header)
tail -n +1 /work/chk_bloom/cf_telem_rank*.csv | sed -n '1,10p'


CSV columns include:

type, ts, rank, path, engine_elapsed_s, engine_bytes, ..., bubble_time_s, note


cpu_disk_write: engine_elapsed_s ≈ serialize/write time

engine_wait : fsync wait time

These are useful for comparing checkpoint overhead across methods.

6) Resume from latest checkpoint (optional)

The training script calls into CFManager.restore() if you wire it in. Typical flow:

from cf_checkpoint import CFCheckpoint
from cf_manager import CFManager, CFMode

chk = CFCheckpoint(model=model, optim=optimizer)
cf_mgr = CFManager(chk_dir="/work/chk_bloom", chk=chk, mode=CFMode.MANUAL, chk_prefix="lm_v_")
extra = cf_mgr.restore(latest=True, gpu=local_rank)
# If you also saved iterator/dataloader state in additional_snapshot,
# apply it here (e.g., cf_dl.load_state_dict(extra['dl_state'])).

7) Notes & tips

Threads vs processes: In distributed runs (torchrun), the code defaults to threads for async persist (set CF_USE_THREAD=1) to avoid pickling DDP objects with spawn.

DALI: Optional. If not installed, the iterator falls back to native DataLoader.

Storage bandwidth probe: disk_bw.py is best-effort; absence of hdparm simply yields 0.0 and does not affect correctness.

Hugging Face cache: First run will download bigscience/bloom-3b into /work/hf_cache. Persist this volume between runs to avoid re-download.

Reproducibility: You may want to set seeds in your script:

import torch, random, numpy as np
random.seed(0); np.random.seed(0)
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

8) What to report in your paper

Mode: MANUAL (frequency N) vs AUTO (chosen frequency).

Metrics: throughput (tokens/sec or steps/sec), mean/median cpu_disk_write and engine_wait from telemetry, wall-clock to epoch end, and checkpoint sizes.

Setup: GPUs, PyTorch/CUDA versions, storage type (NVMe/NFS), and whether async used threads or processes.

9) Troubleshooting

No checkpoints appear: confirm --chk-prefix exists and you’re running on rank 0 (the iterator triggers saves from worker 0).

AUTO never prints “Chosen freq”: run longer (more steps/epochs) so profiling completes.

DDP spawn error inside async process: ensure CF_USE_THREAD=1 (already set above).

Out of space: set overwrite=True in CFManager or periodically prune old .chk files.
