# disk_bw.py
import os, subprocess

_str_bw_file = "./.STR_BW"

def get_storage_bandwidth(disk="/"):
    # Optional: try to reuse cached value
    v = _strProfileExists()
    if v is not None: return v

    # Try to resolve backing device and run hdparm -t
    try:
        paths = disk.split('/')
        mnt_paths = [s for s in paths if s.startswith("mnt")]
        disk_token = mnt_paths[0] if mnt_paths else ""
        p = subprocess.Popen(['grep', disk_token, '/proc/mounts'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = subprocess.check_output(['cut', '-d', ' ', '-f', '1'], stdin=p.stdout)
        p.wait()
        device = output.decode('utf-8').strip()
        if not device:
            return 0.0
        p = subprocess.Popen(['hdparm', '-t', device],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = p.communicate()
        result = out.decode('utf-8')
        str_bw = float(result.split()[-2])  # MB/sec
        os.environ['STR_BW'] = str(str_bw)
        with open(_str_bw_file, 'w+') as wf: wf.write(str(str_bw))
        return str_bw
    except Exception:
        return 0.0

def _strProfileExists():
    if 'STR_BW' in os.environ: 
        try: return float(os.environ['STR_BW'])
        except: return None
    if os.path.exists(_str_bw_file):
        try:
            with open(_str_bw_file, 'r') as rf:
                return float(rf.readline().strip())
        except: 
            return None
    return None
