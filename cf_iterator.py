# cf_iterator.py  (drop-in)
import torch
import time
from collections.abc import Iterable
import math
from statistics import mean
from cf_manager import CFMode
import json
import os

# Optional storage bandwidth (safe fallback if missing)
try:
    from disk_bw import get_storage_bandwidth
except Exception:
    def get_storage_bandwidth(*args, **kwargs):
        return 0.0

# ---- DALI is OPTIONAL ----
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator  # noqa: F401
    HAVE_DALI = True
except Exception:
    HAVE_DALI = False

# Pytorch native dataloader
from torch.utils.data.dataloader import DataLoader


class CFIterator:
    def __init__(
            self, 
            dataloader, 
            bs=128, 
            dali=False,            # default False for NLP
            steps_this_epoch=0, 
            epoch=0, 
            arch='resnet18',
            worker_id=0,
            chk_freq=0, 
            steps_to_run=-1,
            max_overhead=5,
            dynamic=False,
            persist=True,
            cf_manager=None):

        if not isinstance(dataloader, Iterable):
            raise ValueError("Dataloader of type {} is not iterable".format(type(dataloader)))

        self._dataloader = dataloader
        print("Using CF Iterator")
        self._steps_this_epoch = steps_this_epoch
        self._samples_this_epoch = 0
        self._epoch = epoch
        self._chk_freq = chk_freq
        self._batch_size = bs

        # only use DALI if actually available
        self._dali = bool(dali and HAVE_DALI)
        if dali and not HAVE_DALI and worker_id == 0:
            print("[CFIterator] DALI requested but not installed. Falling back to native DataLoader.")

        self._persist = persist
        self._steps_to_run = steps_to_run

        # robust step count (works for both DALI iterator and DataLoader)
        self._max_steps_per_epoch = self._infer_max_steps_per_epoch()
        self._total_steps = self._epoch * self._max_steps_per_epoch + self._steps_this_epoch

        self._worker_id = worker_id
        self._steps_this_run = 0
        self._exit = False
        self._cf_manager = cf_manager
        self._chk_fn = None
        self._start_monitor = False
        self._stop_monitor = False
        self._max_overhead = max_overhead
        self._arch = arch
        self._cache_file = ".cache_" + self._arch + "_" + str(self._batch_size)
        self._steps_since_chk = 0
        self.t_start = 0
        self._dynamic = dynamic
        self._chk_mode = CFMode.MANUAL
        if self._cf_manager is not None:
            self._chk_mode = self._cf_manager.mode

        if self._worker_id == 0:
            if self._chk_mode == CFMode.MANUAL and self._chk_freq == 0:
                print("No iter level checkpointing chosen in MANUAL mode")
            elif self._chk_mode == CFMode.MANUAL and self._chk_freq > 0:
                print("Freq {} chosen in MANUAL mode".format(self._chk_freq))

        self._profile_iter_count = 0
        self._prev_iter_end = time.time()
        self._iter_dur = []
        self._mem_util = []
        self._avg_iter_dur = 0
        self._monitor_iter_count = -1
        self._skip_few_monitors = False
        self._use_thread = False  # safer default

        # If AUTO mode, fetch cache if present
        if self._worker_id == 0 and self._chk_mode == CFMode.AUTO:
            if os.path.exists(self._cache_file):
                self.load_params_from_cache()

        print("Epoch={}, this ep steps={}, max_steps={}, total_steps={}, steps_to_run={}".format(
            self._epoch, self._steps_this_epoch, self._max_steps_per_epoch, self._total_steps, self._steps_to_run))

    def _infer_max_steps_per_epoch(self):
        """Compute number of batches/epoch."""
        # Prefer len(dataloader) if available (already #batches)
        try:
            n_batches = len(self._dataloader)
            if isinstance(n_batches, int) and n_batches > 0:
                return n_batches
        except Exception:
            pass

        # DALI-style: samples in _size
        if hasattr(self._dataloader, "_size"):
            try:
                return int(math.ceil(self._dataloader._size / float(self._batch_size)))
            except Exception:
                pass

        # Fallback: try dataset length
        try:
            return int(math.ceil(len(self._dataloader.dataset) / float(self._batch_size)))
        except Exception:
            return 1

    def load_params_from_cache(self):
        found_all = True
        with open(self._cache_file) as f:
            data = json.load(f)
            for key, val in data.items():
                if key == 'avg_iter_dur':
                    self._avg_iter_dur = val
                elif key == 'chk_freq':
                    self._chk_freq = val
                elif key == 'self._chk_fn':
                    self.populate_chk_fn(val)
                elif key == 'self._use_thread':
                    self._use_thread = val
                else:
                    found_all = False

        if found_all:
            self._profile_done = True
            self._skip_few_monitors = True
            self._monitor_iter_count = 0
            print("Loaded : iter_dur = {}s, freq={}, fn={}, th={}".format(
                self._avg_iter_dur, self._chk_freq, self._chk_fn, self._use_thread))

    def cache_params(self):
        data = {
            'avg_iter_dur' : self._avg_iter_dur,
            'chk_freq' : self._chk_freq,
            'self._chk_fn' : self.get_chk_fn_str(),
            'self._use_thread' : self._use_thread
        }
        print("CACHING : ")
        print(data)
        with open(self._cache_file, 'w') as f:
            json.dump(data, f)
            os.fsync(f.fileno())

    def get_chk_fn_str(self):
        if self._chk_fn == getattr(self._cf_manager, "save_cpu"):
            return "cpu"
        return "gpu"

    def populate_chk_fn(self, dev):
        if dev == "cpu":
            self._chk_fn = getattr(self._cf_manager, "save_cpu")
        else:
            self._chk_fn = getattr(self._cf_manager, "save")

    def __iter__(self):
        self._iterator = iter(self._dataloader)
        return self

    def __next__(self):
        # --- profiling for AUTO mode on worker 0 ---
        if self._worker_id == 0 and self._chk_mode == CFMode.AUTO and not getattr(self, "_profile_done", False):
            self._profile_iter_count += 1
            dev = max(0, torch.cuda.current_device())
            if 5 <= self._profile_iter_count < 100:
                print("PROFILE step {}".format(self._profile_iter_count))
                self._iter_dur.append(time.time()-self._prev_iter_end)
                self._mem_util.append(torch.cuda.max_memory_allocated(dev))
            elif self._profile_iter_count == 100:
                self._complete_profile(dev)

        # --- time to checkpoint? ---
        elif self._worker_id == 0 and self._chk_freq > 0 and self._steps_since_chk == self._chk_freq:
            print("MUST CHECKPOINT NOW AT ITER {}, steps {}".format(self._steps_this_epoch, self._steps_since_chk))
            if self._chk_mode == CFMode.MANUAL:
                self._cf_manager.save(synchronous=True, additional_snapshot=self.state_dict(), persist=self._persist)
            else:
                self._chk_fn(additional_snapshot=self.state_dict(), use_thread=self._use_thread)
            self._steps_since_chk = 0

            if getattr(self, "_profile_done", False) and not self._start_monitor and self._chk_mode == CFMode.AUTO and not self._stop_monitor:
                self._iter_dur = []
                self._start_monitor = True

        # --- ongoing overhead monitor block ---
        if self._worker_id == 0 and self._start_monitor and not self._stop_monitor:
            self._monitor_iter_count += 1
            if self._monitor_iter_count == 1:
                self.t_start = time.time()
            elif self._monitor_iter_count == self._chk_freq + 1:
                t_dur = time.time() - self.t_start
                print("Duration between checkpoints = {:.2f}s".format(t_dur))
            if 0 < self._monitor_iter_count <= self._chk_freq:
                self._iter_dur.append(time.time()-self._prev_iter_end)
            elif self._monitor_iter_count == self._chk_freq + 1:
                if self._skip_few_monitors and len(self._iter_dur) > 3:
                    self._iter_dur = self._iter_dur[2:-1]
                    self._skip_few_monitors = False
                current_iter_mean = mean(self._iter_dur)
                current_total = sum(self._iter_dur)
                orig_total = self._avg_iter_dur * len(self._iter_dur)
                num_items = len(self._iter_dur) - int(len(self._iter_dur)/3)
                if num_items > 3:
                    new_avg = mean(self._iter_dur[num_items:])
                    print("New avg = {:.3f}s".format(new_avg))
                overhead = max(0, current_iter_mean - self._avg_iter_dur)
                overhead_full = current_total - orig_total
                overhead_percent = overhead_full / max(1e-8, orig_total) * 100
                print("NEW OVERHEAD IS  ={:.2f}, FULL={:.2f}".format(overhead, overhead_full))
                print(self._iter_dur)
                print("OLD ITER={:.2f}, NEW_ITER={:.2f}".format(self._avg_iter_dur, current_iter_mean))
                print("OLD TOTAL={:.2f}, NEW_TOTAL={:.2f}, percent={:.2f}%".format(orig_total, current_total, overhead_percent))
                if overhead_percent > self._max_overhead:
                    self._chk_freq += 2
                    self.cache_params()
                    print("Changed chk freq to {}".format(self._chk_freq))
                elif not self._dynamic:
                    self._start_monitor = False
                    self._stop_monitor = True
                self._iter_dur = []
                self._monitor_iter_count = 0

        # crash/stop simulation
        if self._steps_to_run >= 0 and self._steps_this_run == self._steps_to_run:
            self._exit = True
            print("Epoch={}, this ep steps={}, total_steps={}, steps_this_run={}".format(
                self._epoch, self._steps_this_epoch, self._total_steps, self._steps_this_run))
            raise StopIteration

        self._prev_iter_end = time.time()

        try:
            val = next(self._iterator)
        except StopIteration:
            print("Tracking epoch end in CF DL. Steps this run = {}, this epoch={}, samples this epoch={}".format(
                self._steps_this_run, self._steps_this_epoch, self._samples_this_epoch))
            self._epoch += 1
            self._steps_this_epoch = 0
            self._samples_this_epoch = 0
            self._steps_since_chk = 0
            print("Epoch set to {}".format(self._epoch))
            # Force epoch checkpoint
            if self._worker_id == 0 and self._chk_mode == CFMode.MANUAL:
                self._cf_manager.save(synchronous=True, additional_snapshot=self.state_dict(),
                                      persist=self._persist, is_epoch=True, epoch=self._epoch)
            elif self._worker_id == 0:
                if self._chk_fn is not None:
                    self._chk_fn(additional_snapshot=self.state_dict(), is_epoch=True, epoch=self._epoch)
                else:
                    print('[CFIterator] No checkpoint function set; skipping epoch-end checkpoint.')
            raise StopIteration

        self._total_steps += 1
        self._steps_this_epoch += 1
        self._samples_this_epoch += self._batch_size
        self._steps_this_run += 1
        self._steps_since_chk += 1

        return val

    def _complete_profile(self, dev):
        self._avg_iter_dur = mean(self._iter_dur) if len(self._iter_dur) else 0.0
        try:
            self._avg_free_mem = (torch.cuda.get_device_properties(dev).total_memory -
                                  mean(self._mem_util)) / 1024 / 1024
        except Exception:
            self._avg_free_mem = 0.0
        print("Average iter dur = {:.3f}, free mem = {:.2f}MB".format(self._avg_iter_dur, self._avg_free_mem))
        self._chk_size = self._cf_manager.get_chk_size
        print("Size of chk = {:.3f}MB".format(self._chk_size))
        disk_path = self._cf_manager.chk_dir
        if disk_path.startswith("./"):
            disk_path = os.getcwd()
        print("Disk path = {}".format(disk_path))
        _ = get_storage_bandwidth(disk_path)  # optional

        # profile CPU snapshot time (thread vs proc)
        _, t_ct = self._cf_manager.save_cpu(profile_snap=True, use_thread=True)
        _, t_cp = self._cf_manager.save_cpu(profile_snap=True, use_thread=False)
        if t_ct <= t_cp:
            t_c = t_ct; self._use_thread = True
        else:
            t_c = t_cp; self._use_thread = False
        print("t_cp={:.4f}, t_ct={:.4f}".format(t_cp, t_ct))
        self._chk_fn = getattr(self._cf_manager, "save_cpu")
        overhead = t_c

        # consider GPU snapshot if memory allows
        t_g = t_s = 0.0
        if self._chk_size < self._avg_free_mem:
            t_gt, t_ft = self._cf_manager.save(profile_full=True, use_thread=True)
            t_gp, t_fp = self._cf_manager.save(profile_full=True, use_thread=False)
            if t_fp <= t_ft:
                t_f = t_fp; t_g = t_gp; self._use_thread = False
            else:
                t_f = t_ft; t_g = t_gt; self._use_thread = True
            print("t_fp={:.4f}, t_ft={:.4f}, t_gp={:.4f}, t_gt={:.4f}".format(t_fp, t_ft, t_gp, t_gt))
            t_s = t_f - t_g
            if overhead >= t_g:
                self._chk_fn = getattr(self._cf_manager, "save")
                overhead = t_g
        else:
            t_f = t_c  # simplest

        print("t_c={:.4f}, t_g={:.4f}, t_s={:.4f}, t_w={:.4f}".format(t_c, t_g, t_s, self._avg_iter_dur))
        print("Chosen function is : {}, overhead={}".format(self._chk_fn, overhead))
        _, t_f = self._chk_fn(profile_full=True, use_thread=self._use_thread)

        t_i = max(1e-8, self._avg_iter_dur)
        self._chk_freq = max(int(math.ceil((t_f - overhead) / t_i)), 1)
        percent_overhead = overhead / (self._chk_freq * t_i) * 100.0

        self.cache_params()
        self._profile_done = True
        self._steps_since_chk = 0
        print("Chosen freq = {}, percent_ov={:.2f}%".format(self._chk_freq, percent_overhead))

    def state_dict(self):
        return {
            'iter' : self._steps_this_epoch,
            'steps_so_far' : self._total_steps, 
            'start_index' : self._samples_this_epoch,
            'epoch' : self._epoch
        }

    def load_state_dict(self, chk_map):
        if chk_map is None: return
        self._steps_this_epoch = chk_map['iter']
        self._total_steps = chk_map['steps_so_far']
        self._samples_this_epoch = chk_map['start_index']
        self._epoch = chk_map['epoch']

    def reset(self):
        # DALI has reset(); DataLoader does not.
        if hasattr(self._dataloader, "reset"):
            return self._dataloader.reset()
        return None

    @property
    def _size(self):
        # Kept for compatibility
        if hasattr(self._dataloader, "_size"):
            return self._dataloader._size
        try:
            return len(self._dataloader.dataset)
        except Exception:
            return self._infer_max_steps_per_epoch() * self._batch_size
