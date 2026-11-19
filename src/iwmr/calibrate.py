#!/usr/bin/env python3
"""calibrate.py - check, before the IWMR starts, that the per-container GPU
metrics can be trusted on this node.

The figures a container reports about itself are calculated by libvitroobs.so
(container/vitro_observer.c), because under MPS the driver attributes all
utilisation to the MPS server. This test runs one inference container alone on
a GPU and compares what the library says with what the driver says:

  sm_limit      SMs the container sees        vs  share x SMs of the GPU
  memory        footprint (cap - free)        vs  driver's per-process memory
  utilisation   calculated busy time          vs  nvidia-smi, GPU-wide
  attribution   with overlapping requests, every kernel launch is assigned to
                an inference (none unattributed, unobserved or unresolved)

It also measures the context overhead (memory a container uses beyond what its
model allocates), which the admission ledger needs.

The GPU has to be idle, which it is before the runtime starts. If workloads are
already running (runtime restart) the last stored result is used instead. A
result is stored per GPU and reused while GPU model, driver and image are the
same; --force runs it again.

  sudo python3 iwmr/calibrate.py --all --model /srv/models/calib.onnx
  python3 iwmr/calibrate.py --gpu 0 --model calib.onnx --state-dir /tmp/vitro

Use a model that keeps the GPU busy, e.g. `tests/make_test_model.py out.onnx 4096`;
with a light model the utilisation check is inconclusive.

Exit status: 0 if every GPU is verified (now or from a stored result), else 1.
A failed check must not stop the runtime: limits are enforced by MPS, not by
the library. It only means the metrics of that GPU are reported as unverified.

The pass thresholds below are PROVISIONAL: they come from the first runs on one
GPU and have to be revisited with results from the target hardware.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "container"))
import vitro_stats                                    # noqa: E402

SM_TOLERANCE = 1            # SMs
MEM_TOLERANCE_MIB = 16
UTIL_TOLERANCE_PCT = 10     # percentage points
MIN_LOAD_PCT = 20           # below this the utilisation comparison says nothing
MIB = 2**20


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def smi(gpu, fields, what="gpu"):
    return sh("nvidia-smi", "-i", str(gpu), f"--query-{what}={fields}", "--format=csv,noheader,nounits")


def gpu_util(gpu):
    return float(smi(gpu, "utilization.gpu"))


SM_COUNT = """
import ctypes
cu = ctypes.CDLL("libcuda.so.1")
dev, ctx, sms = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_int()
bad = (cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), 0) or
       cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) or cu.cuCtxSetCurrent(ctx) or
       cu.cuDeviceGetAttribute(ctypes.byref(sms), 16, dev))
print(-1 if bad else sms.value)
"""


def mps_args(a, gpu, share):
    """docker run arguments that make a container an MPS client of `gpu`."""
    return ["--gpus", f"device={gpu}", "--ipc", "host",
            "--security-opt", "label=disable",        # host paths are mounted, no relabelling
            "-e", "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps",
            "-v", f"{a.mps_root}/gpu{gpu}:/tmp/nvidia-mps",
            "-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={share}"]


def total_sms(a, gpu):
    """SM count of the whole GPU, as an MPS client at 100 % sees it.

    Asked from inside a container, not from this script: an MPS server serves
    one user at a time and takes minutes to switch, so the question has to come
    from the same user as the workload containers.
    """
    try:
        out = subprocess.run(["docker", "run", "--rm", *mps_args(a, gpu, 100), a.image,
                              "python3", "-c", SM_COUNT], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    last = out.stdout.strip().splitlines()[-1:] or [""]
    return int(last[0]) if last[0].isdigit() else None


class Container:
    """One inference container under MPS on `gpu`, alone, with its stats file on the host."""

    def __init__(self, a, gpu, run_dir):
        self.name, self.stats = f"vitro-calib-gpu{gpu}", os.path.join(run_dir, "stats")
        sh("docker", "rm", "-f", self.name, check=False)
        os.makedirs(run_dir, exist_ok=True)
        if os.path.exists(self.stats):
            os.remove(self.stats)
        sh("docker", "run", "-d", "--name", self.name, *mps_args(a, gpu, a.share),
           "-e", f"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0={a.mem_mib}M",
           "-v", f"{os.path.abspath(a.model)}:/models/model.onnx:ro",
           "-v", f"{run_dir}:/run/vitro", a.image)
        ip = sh("docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self.name)
        self.url = f"http://{ip}:8000"

    def get(self, path):
        return json.load(urllib.request.urlopen(self.url + path, timeout=30))

    def wait_ready(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                return self.get("/health")
            except OSError:
                if sh("docker", "inspect", "-f", "{{.State.Running}}", self.name, check=False) != "true":
                    break
                time.sleep(1)
        raise RuntimeError("container did not become ready:\n" + self.logs())

    def infer(self, body):
        req = urllib.request.Request(self.url + "/infer", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120)).get("metrics")

    def read(self):
        s = vitro_stats.read(self.stats)
        if s is None:
            raise RuntimeError("no stats file from the observer library:\n" + self.logs())
        return s

    def driver_mem_mib(self, gpu):
        """Memory the driver attributes to this container's processes."""
        pids = set(sh("docker", "top", self.name, "-eo", "pid").split()[1:])
        rows = [r.split(",") for r in smi(gpu, "pid,used_memory", "compute-apps").splitlines()]
        mine = [float(m) for p, m in rows if p.strip() in pids]
        return sum(mine) if mine else None

    def logs(self):
        return sh("docker", "logs", "--tail", "30", self.name, check=False)

    def remove(self):
        sh("docker", "rm", "-f", self.name, check=False)


def load(c, gpu, body, clients, seconds):
    """Send inferences from `clients` threads; sample the driver's GPU-wide utilisation."""
    before, results, errors, samples = c.read(), [], [], []
    stop = time.time() + seconds

    def client():
        while time.time() < stop:
            try:
                results.append(c.infer(body))
            except Exception as e:                    # noqa: BLE001 - reported below
                errors.append(str(e))
                return

    threads = [threading.Thread(target=client) for _ in range(clients)]
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(1.0)                                   # nvidia-smi averages over the last second
    while time.time() < stop - 0.3:
        samples.append(gpu_util(gpu))
        time.sleep(0.5)
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    time.sleep(2.5)                                   # busy_ms is published 2 s late
    after = c.read()
    if errors or not results or None in results:
        raise RuntimeError(f"inference failed: {errors[:1] or 'no metrics in the response'}")
    d = {k: after[k] - before[k] for k in ("kernels", "kernels_unattributed", "kernels_unobserved",
                                           "busy_ms", "gpu_ms", "smw_ms")}
    n = len(results)
    return {
        "inferences": n,
        "util_calculated_pct": round(100 * d["busy_ms"] / (elapsed * 1000), 1),
        "util_driver_pct": round(sum(samples) / len(samples), 1) if samples else None,
        "smw_util_pct": round(100 * d["smw_ms"] / (elapsed * 1000), 1),
        "kernels": d["kernels"], "unattributed": d["kernels_unattributed"],
        "unobserved": d["kernels_unobserved"],
        "unresolved": sum(r["unresolved"] for r in results),
        "per_inference": {k: round(sum(r[k] for r in results) / n, 2)
                          for k in ("gpu_ms", "wall_ms", "util_pct", "smw_util_pct")},
    }, after


def calibrate(a, gpu, key):
    run_dir = os.path.join(a.state_dir, "calibration", f"run-gpu{gpu}")
    full = total_sms(a, gpu)
    c = Container(a, gpu, run_dir)
    try:
        health = c.wait_ready(a.ready_timeout)
        if not health.get("observer"):
            raise RuntimeError("the image does not preload libvitroobs.so")
        shape = next(iter(health["inputs"].values()))["shape"]
        name = next(iter(health["inputs"]))
        body = json.dumps({"inputs": {name: [[0.5] * int(shape[-1])]}}).encode()
        c.infer(body)                                 # warm-up: first run sets up cuDNN/cuBLAS

        single, _ = load(c, gpu, body, 1, a.seconds)
        overlap, stats = load(c, gpu, body, a.clients, a.seconds / 2)
        drv_mem = c.driver_mem_mib(gpu)
    finally:
        c.remove()

    checks = {}
    want = None if full is None else a.share / 100 * full
    checks["sm_limit"] = {
        "library": stats["sm_count"], "expected": want, "gpu_total": full,
        "status": "skip" if want is None else
                  "pass" if abs(stats["sm_count"] - want) <= SM_TOLERANCE and stats["sm_count"] <= full else "fail"}
    foot = stats["mem_footprint_bytes"] / MIB
    checks["memory"] = {
        "library_mib": round(foot, 1), "driver_mib": drv_mem,
        "status": "skip" if drv_mem is None else
                  "pass" if abs(foot - drv_mem) <= MEM_TOLERANCE_MIB else "fail"}
    worst = "pass"
    for label, run in (("single", single), ("overlap", overlap)):
        calc, drv = run["util_calculated_pct"], run["util_driver_pct"]
        status = ("skip" if drv is None or drv < MIN_LOAD_PCT else
                  "pass" if abs(calc - drv) <= UTIL_TOLERANCE_PCT else "fail")
        run["status"] = status
        worst = "fail" if "fail" in (worst, status) else "skip" if "skip" in (worst, status) else "pass"
    checks["utilisation"] = {"single": single, "overlap": overlap, "status": worst}
    lost = sum(r[k] for r in (single, overlap) for k in ("unattributed", "unobserved", "unresolved"))
    checks["attribution"] = {"kernels": single["kernels"] + overlap["kernels"], "not_attributed": lost,
                             "status": "pass" if lost == 0 else "fail"}

    return {
        "gpu": gpu, "key": key, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "passed": all(ch["status"] == "pass" for ch in checks.values()),
        "checks": checks,
        "derived": {"ctx_overhead_mib": round(stats["mem_overhead_bytes"] / MIB, 1),
                    "model_mib": round(stats["mem_model_bytes"] / MIB, 1)},
        "settings": {"share_pct": a.share, "mem_cap_mib": a.mem_mib, "seconds": a.seconds,
                     "clients": a.clients, "model": os.path.abspath(a.model)},
        "thresholds": {"provisional": True, "sm": SM_TOLERANCE, "mem_mib": MEM_TOLERANCE_MIB,
                       "util_pct": UTIL_TOLERANCE_PCT, "min_load_pct": MIN_LOAD_PCT},
    }


def busy(gpu):
    """Workload containers (gs-*) running, or the GPU already loaded."""
    if sh("docker", "ps", "-q", "--filter", "name=^gs-", check=False):
        return "workload containers are running"
    samples = []
    for _ in range(3):                                # one reading can be stale
        samples.append(gpu_util(gpu))
        time.sleep(1)
    return f"GPU is at {min(samples):.0f} % utilisation" if min(samples) > 10 else None


def summary(r):
    ch = r["checks"]
    u = ch["utilisation"]
    return "\n".join([
        f"  sm_limit     {ch['sm_limit']['status']:<5} library {ch['sm_limit']['library']}, "
        f"expected {ch['sm_limit']['expected']} of {ch['sm_limit']['gpu_total']}",
        f"  memory       {ch['memory']['status']:<5} library {ch['memory']['library_mib']} MiB, "
        f"driver {ch['memory']['driver_mib']} MiB",
        f"  utilisation  {u['status']:<5} single: {u['single']['util_calculated_pct']} % vs driver "
        f"{u['single']['util_driver_pct']} %; overlap: {u['overlap']['util_calculated_pct']} % vs "
        f"{u['overlap']['util_driver_pct']} %",
        f"  attribution  {ch['attribution']['status']:<5} {ch['attribution']['not_attributed']} of "
        f"{ch['attribution']['kernels']} kernels not attributed",
        f"  context overhead {r['derived']['ctx_overhead_mib']} MiB; per inference (single): "
        f"{u['single']['per_inference']}",
    ])


def main():
    p = argparse.ArgumentParser(description="Verify the per-container GPU metrics on this node.")
    p.add_argument("--gpu", type=int, action="append", help="GPU index (repeatable)")
    p.add_argument("--all", action="store_true", help="all GPUs except $VALIDATION_GPU")
    p.add_argument("--model", help="ONNX model that loads the GPU")
    p.add_argument("--image", default="onnx-infer:latest")
    p.add_argument("--share", type=int, default=50, help="compute share in %% (default 50)")
    p.add_argument("--mem-mib", type=int, default=1024, help="memory cap (default 1024)")
    p.add_argument("--seconds", type=float, default=10, help="length of the load run (default 10)")
    p.add_argument("--clients", type=int, default=4, help="overlapping requests (default 4)")
    p.add_argument("--ready-timeout", type=float, default=180)
    p.add_argument("--state-dir", default="/var/lib/gpushare")
    p.add_argument("--mps-root", default="/var/run/nvidia-mps")
    p.add_argument("--force", action="store_true", help="ignore a stored result")
    a = p.parse_args()
    if not a.model or not os.path.isfile(a.model):
        p.error("--model must be an existing ONNX file")
    gpus = a.gpu or []
    if a.all:
        skip = os.environ.get("VALIDATION_GPU", "")
        gpus = [int(g) for g in sh("nvidia-smi", "--query-gpu=index", "--format=csv,noheader").split()
                if g != skip]
    if not gpus:
        p.error("give --gpu N or --all")

    out_dir = os.path.join(a.state_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    image_id = sh("docker", "image", "inspect", "-f", "{{.Id}}", a.image)
    verified = True
    for gpu in gpus:
        name, driver = [x.strip() for x in smi(gpu, "name,driver_version").split(",")]
        key = {"gpu_name": name, "driver": driver, "image": image_id}
        path = os.path.join(out_dir, f"gpu{gpu}.json")
        stored = json.load(open(path)) if os.path.exists(path) else None
        current = stored is not None and stored.get("key") == key
        print(f"GPU {gpu} ({name}, driver {driver})")
        if current and stored["passed"] and not a.force:
            print(f"  verified earlier ({stored['time']}); stored result reused")
            continue
        why = busy(gpu)
        if why:
            print(f"  not tested: {why}. " + ("Stored result is for another GPU/driver/image."
                                              if stored and not current else "No usable stored result.")
                  + " Metrics of this GPU are UNVERIFIED.")
            verified = False
            continue
        try:
            result = calibrate(a, gpu, key)
        except Exception as e:                        # noqa: BLE001 - one GPU must not stop the rest
            print(f"  test could not run: {e}\n  Metrics of this GPU are UNVERIFIED.")
            verified = False
            continue
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(summary(result))
        print(f"  -> {'VERIFIED' if result['passed'] else 'NOT VERIFIED'}; result in {path}")
        verified &= result["passed"]
    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
