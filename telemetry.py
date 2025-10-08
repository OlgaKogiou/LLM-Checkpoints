import os, time
class TelemetryLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path,'w') as f:
                f.write("type,ts,rank,path,engine_elapsed_s,engine_bytes,"
                        "gpu_alloc_before,gpu_alloc_after,gpu_reserved_before,gpu_reserved_after,"
                        "gpu_nvml_used_before,gpu_nvml_used_after,cpu_rss_before,cpu_rss_after,"
                        "disk_total,disk_used,disk_free,bubble_time_s,note\n")
    def record(self, typ, path, rank, engine_elapsed_s, engine_bytes, note=""):
        ts = time.time()
        with open(self.path,'a') as f:
            f.write(f"{typ},{ts},{rank},{path},{engine_elapsed_s},{engine_bytes},"
                    f",,,,,,,,,,,,{note}\n")
