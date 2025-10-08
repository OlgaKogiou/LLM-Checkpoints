import torch
import os
import enum
import logging
import copy
from collections import OrderedDict
from collections.abc import Mapping
import time
from telemetry import TelemetryLogger
"""
Checkpointing and restoring logic
"""

class CFCheckpoint:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.tracking_map = OrderedDict()
        self.latest_snapshot = None
        self.pipeline_snapshot = None

        for name, ref in kwargs.items():
            if hasattr(ref, 'state_dict'):
                self.tracking_map[name] = ref
            else:
                self.logger.info("Skipping object `{}` in CF Checkpointing. No state_dict() method exposed".format(name))

        self.num_tracking = len(self.tracking_map.keys())
        if self.num_tracking == 0:
            raise ValueError("Nothing to track")

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def _snapshot(self, active_snapshot, additional_state=None):
        if active_snapshot == 1:
            self.logger.info("Snapshot is not None. Checkpoint underway")
            return False

        self.latest_snapshot = OrderedDict()

        for name, ref in self.tracking_map.items():
            if name not in self.latest_snapshot:
                self.latest_snapshot[name] = copy.deepcopy(ref.state_dict())
            else:
                self.logger.info("Repeated entry for {}".format(name))
                return False

        if isinstance(additional_state, Mapping):
            self.latest_snapshot.update(additional_state)

        return True

    def _serialize_and_persist(
        self,
        filepath,
        snapshot,
        active_snapshot,
        lock,
        linkpath=None,
        iter_chk=None,
        epoch_chk=None,
        overwrite=True
    ):
        print("[{}] START ASYNC".format(time.time()))

        with lock:
            if active_snapshot.value == 0:
                self.logger.error("Cannot persist. Empty snapshot")
                return

        # Use a CUDA stream (behavior preserved)
        try:
            s = torch.cuda.Stream()
            torch.cuda.stream(s)
        except Exception:
            pass

        # Write timing
        w_t0 = time.time()
        torch.save(snapshot, filepath)
        write_elapsed = time.time() - w_t0
        try:
            telem = TelemetryLogger(os.path.join(os.path.dirname(filepath), f"cf_telem_rank{os.environ.get('LOCAL_RANK','0')}.csv"))
            import os as _os
            _size_bytes = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0
            telem.record('cpu_disk_write', filepath, int(os.environ.get('LOCAL_RANK', 0)), write_elapsed, _size_bytes, 'Header+lean_state_dict write')
        except Exception:
            pass

        # Clear the snapshot flag
        with lock:
            active_snapshot.value = 0

        # Fsync timing
        fs_elapsed = 0.0
        try:
            fs_t0 = time.time()
            f = open(filepath, 'a+')
            os.fsync(f.fileno())
            f.close()
            fs_elapsed = time.time() - fs_t0
            try:
                telem = TelemetryLogger(os.path.join(os.path.dirname(filepath), f"cf_telem_rank{os.environ.get('LOCAL_RANK','0')}.csv"))
                import os as _os
                _size_bytes = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0
                telem.record('engine_wait', filepath, int(os.environ.get('LOCAL_RANK', 0)), fs_elapsed, _size_bytes, 'CPU->Disk completion (blocking)')
            except Exception:
                pass
        except Exception:
            pass

        update_stats(
            filepath,
            iter_chk=iter_chk,
            overwrite=overwrite,
            epoch_chk=epoch_chk,
            linkpath=linkpath
        )
        print("[{}] END ASYNC".format(time.time()))

    def _serialize_and_persist_direct(
        self,
        snapshot,
        filepath,
        additional_state=None,
        persist=True,
        linkpath=None,
        iter_chk=None,
        epoch_chk=None,
        overwrite=True
    ):
        # Build a fresh state dict on demand
        snap_ptr = {}
        for name, ref in self.tracking_map.items():
            snap_ptr[name] = ref.state_dict()

        if isinstance(additional_state, Mapping):
            snap_ptr.update(additional_state)

        w_t0 = time.time()
        torch.save(snap_ptr, filepath)
        write_elapsed = time.time() - w_t0

        fs_elapsed = 0.0
        if persist:
            fs_t0 = time.time()
            f = open(filepath, 'a+')
            os.fsync(f.fileno())
            f.close()
            fs_elapsed = time.time() - fs_t0

        try:
            telem = TelemetryLogger(os.path.join(os.path.dirname(filepath), f"cf_telem_rank{os.environ.get('LOCAL_RANK','0')}.csv"))
            import os as _os
            _size_bytes = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0
            telem.record('cpu_disk_write', filepath, int(os.environ.get('LOCAL_RANK', 0)), write_elapsed, _size_bytes, 'Header+lean_state_dict write')
            if persist:
                telem.record('engine_wait', filepath, int(os.environ.get('LOCAL_RANK', 0)), fs_elapsed, _size_bytes, 'CPU->Disk completion (blocking)')
        except Exception:
            pass

        update_stats(
            filepath,
            iter_chk=iter_chk,
            overwrite=overwrite,
            epoch_chk=epoch_chk,
            linkpath=linkpath
        )

    def _snapshot_and_persist_async(
        self,
        filepath,
        active_snapshot,
        in_progress_snapshot,
        lock,
        snap_ptr,
        additional_state=None,
        persist=True,
        linkpath=None,
        iter_chk=None,
        epoch_chk=None,
        overwrite=True,
        profile=False
    ):
        s = time.time()
        print("[{}] START ASYNC".format(time.time()))
        if active_snapshot.value == 1:
            print("ERROR! Snapshot active")
            return

        print("In progress snapshot val = {}".format(in_progress_snapshot.value))

        # CPU snapshot
        snapshot = {}
        for name, ref in snap_ptr.items():
            snapshot[name] = _to_cpu(ref)
        print("Time for CPU snapshot = {}s".format(time.time() - s))

        with lock:
            in_progress_snapshot.value = 0
            active_snapshot.value = 1
        print("In progress snapshot val = {}".format(in_progress_snapshot.value))
        print("[{}] START ASYNC PERSIST".format(time.time()))

        if isinstance(additional_state, Mapping):
            snapshot.update(additional_state)

        if profile:
            with lock:
                active_snapshot.value = 0
            return

        # Write timing
        w_t0 = time.time()
        torch.save(snapshot, filepath)
        write_elapsed = time.time() - w_t0
        try:
            telem = TelemetryLogger(os.path.join(os.path.dirname(filepath), f"cf_telem_rank{os.environ.get('LOCAL_RANK','0')}.csv"))
            import os as _os
            _size_bytes = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0
            telem.record('cpu_disk_write', filepath, int(os.environ.get('LOCAL_RANK', 0)), write_elapsed, _size_bytes, 'Header+lean_state_dict write')
        except Exception:
            pass

        with lock:
            active_snapshot.value = 0

        # Fsync timing
        fs_t0 = time.time()
        f = open(filepath, 'a+')
        os.fsync(f.fileno())
        f.close()
        fs_elapsed = time.time() - fs_t0
        try:
            telem = TelemetryLogger(os.path.join(os.path.dirname(filepath), f"cf_telem_rank{os.environ.get('LOCAL_RANK','0')}.csv"))
            import os as _os
            _size_bytes = _os.path.getsize(filepath) if _os.path.exists(filepath) else 0
            telem.record('engine_wait', filepath, int(os.environ.get('LOCAL_RANK', 0)), fs_elapsed, _size_bytes, 'CPU->Disk completion (blocking)')
        except Exception:
            pass

        update_stats(
            filepath,
            iter_chk=iter_chk,
            overwrite=overwrite,
            epoch_chk=epoch_chk,
            linkpath=linkpath
        )
        print("Time to checkpoint = {}s".format(time.time() - s))
        print("[{}] END ASYNC".format(time.time()))

    def _restore(self, filepath='model.chk', gpu=0):
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage.cuda(gpu))
        for name, ref in self.tracking_map.items():
            try:
                ref.load_state_dict(checkpoint[name])
                del checkpoint[name]
            except ValueError:
                print("Corrupt checkpoint")

        if len(checkpoint.keys()) > 0:
            return checkpoint
        return None


def update_stats(filepath, iter_chk=None, epoch_chk=None, overwrite=True, linkpath=None):
    if iter_chk is not None:
        fname = fname_from_path(filepath)
        if fname not in iter_chk:
            iter_chk.append(fname)

        if overwrite and len(iter_chk) > 1:
            del_filepath = os.path.join(os.path.dirname(filepath), iter_chk[0] + '.chk')
            if os.path.exists(del_filepath):
                os.remove(del_filepath)
            del iter_chk[0]

    if linkpath is not None and epoch_chk is not None:
        epoch_chk.append(fname_from_path(linkpath))


def _to_cpu(ele, snapshot=None):
    if snapshot is None:
        snapshot = {}
    if hasattr(ele, 'cpu'):
        snapshot = ele.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = _to_cpu(v, None)
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v, None)
    return snapshot


def fname_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]
