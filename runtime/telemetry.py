"""telemetry.py - periodic device utilisation and memory sampling.

Values are GPU-wide (MPS does not report per client). They are exposed in
/state for operators and used by one admission guard: a GPU whose measured
memory use already exceeds the ledger's expectation is flagged, because
something outside the runtime is using it.
"""
import glob, subprocess, threading, time


def sample_nvidia():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, util, mem = [x.strip() for x in line.split(",")]
        res[int(i)] = {"util_pct": float(util), "mem_used": int(mem) * 2**20}
    return res


def sample_amd(gpus):
    res = {}
    for g, info in gpus.items():
        dev = f"/sys/class/drm/renderD{info['render_minor']}/device"
        try:
            res[g] = {"util_pct": float(open(f"{dev}/gpu_busy_percent").read()),
                      "mem_used": int(open(f"{dev}/mem_info_vram_used").read())}
        except OSError:
            pass
    return res


class Telemetry(threading.Thread):
    def __init__(self, vendor, backend, period=5):
        super().__init__(daemon=True)
        self.vendor, self.backend, self.period = vendor, backend, period
        self.latest, self.ts = {}, 0

    def run(self):
        while True:
            try:
                self.latest = (sample_nvidia() if self.vendor == "nvidia"
                               else sample_amd(self.backend.gpus))
                self.ts = time.time()
            except Exception as e:                          # never kill the runtime
                print(f"[telemetry] sample failed: {e}")
            time.sleep(self.period)
