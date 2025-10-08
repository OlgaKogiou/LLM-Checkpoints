# src/telemetry.py
import os, time, csv, shutil
import torch

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

_HDR = [
    "type","ts","rank","path","engine_elapsed_s","engine_bytes",
    "gpu_alloc_before","gpu_alloc_after",
    "gpu_reserved_before","gpu_reserved_after",
    "gpu_nvml_used_before","gpu_nvml_used_after",
    "cpu_rss_before","cpu_rss_after",
    "disk_total","disk_used","disk_free",
    "bubble_time_s","note"
]

class TelemetryLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(_HDR)

    def _gpu_mem(self):
        alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        nvml_used = 0
        if _NVML_OK and torch.cuda.is_available():
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                nvml_used = pynvml.nvmlDeviceGetMemoryInfo(h).used
            except Exception:
                pass
        return alloc, reserved, nvml_used

    def _cpu_mem(self):
        if psutil:
            return psutil.Process().memory_info().rss
        # /proc/self/statm fallback (pages * page_size)
        try:
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return 0

    def _disk(self, path):
        try:
            usage = shutil.disk_usage(os.path.abspath(os.path.dirname(path) or "."))
            return usage.total, usage.used, usage.free
        except Exception:
            return 0,0,0

    def record(self, ev_type, path, rank, elapsed_s, nbytes, note="", bubble_time_s=""):
        ts = time.time()
        ga_b, gr_b, nv_b = self._gpu_mem()
        cr_b = self._cpu_mem()
        dt_b, du_b, df_b = self._disk(path)
        # small sleep to let CUDA accounting tick when called twice quickly
        ga_a, gr_a, nv_a = self._gpu_mem()
        cr_a = self._cpu_mem()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ev_type, f"{ts:.6f}", rank, path,
                f"{elapsed_s:.9f}", int(nbytes),
                int(ga_b), int(ga_a),
                int(gr_b), int(gr_a),
                int(nv_b), int(nv_a),
                int(cr_b), int(cr_a),
                int(dt_b), int(du_b), int(df_b),
                f"{bubble_time_s}" if bubble_time_s=="" else f"{float(bubble_time_s):.9f}",
                note
            ])
