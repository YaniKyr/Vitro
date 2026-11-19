#!/usr/bin/env python3
"""util_sum.py - do the utilisations of several containers add up in nvidia-smi?

Starts N inference containers under MPS and gives each one a workload that
keeps the GPU busy for a target fraction of the time (default 20 %). Each
container is first run alone, to confirm that it produces the target, then all
run together and the GPU-wide figure of nvidia-smi is compared with the sum.

The load is paced: an inference of d ms is followed by a pause, so that one
starts every d / target ms on average (intervals are randomised +-50 %, so the
containers do not stay in step). In the solo run d follows the measured
inference time, which holds the container at the target; the joint run keeps
the interval the solo run ended with, i.e. every container gets the same work
as when it was alone.
With --saturate there is no pause: the only limit is the SM share.

  util_sum.py --model m.onnx                       3 containers, 20 % each
  util_sum.py --model m.onnx --share 100           the same without an SM limit
  util_sum.py --model m.onnx --saturate            20 % of the SMs each, full load
"""
import argparse, json, subprocess, os, random, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "iwmr"))
import calibrate as cal                                # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--model", required=True)
p.add_argument("--n", type=int, default=3)
p.add_argument("--target", type=float, default=20, help="utilisation per container, %%")
p.add_argument("--share", type=int, default=20, help="MPS active thread percentage per container")
p.add_argument("--saturate", action="store_true", help="no pacing: full load")
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--mem-mib", type=int, default=400)
p.add_argument("--seconds", type=float, default=30, help="joint run (solo runs take a third)")
p.add_argument("--image", default="onnx-infer:latest")
p.add_argument("--mps-root", default="/var/run/nvidia-mps")
p.add_argument("--ns", help="MPS v3: comma list of <server>/<namespace>, one per container; "
               "limits then come from the namespace, not from the environment")
p.add_argument("--json")
a = p.parse_args()
NS = a.ns.split(",") if a.ns else []


def mps_args(i, share):
    """docker arguments that make container i (1-based) an MPS client.

    With --ns only the namespace's own directory is mounted, and the container
    is told the memory limit of the namespace (see container/entrypoint.sh).
    Without it the limits are set in the container's environment (legacy).
    """
    if not NS:
        return [*cal.mps_args(a, a.gpu, share), "-e", f"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0={a.mem_mib}M"]
    ns = NS[(i - 1) % len(NS)]
    server, name = ns.split("/")
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=f"{a.mps_root}/gpu{a.gpu}", CUDA_MPS_PROTOCOL_VERSION="3")
    cap = subprocess.run(["nvidia-cuda-mps-control", "namespace", "get", name, server, "pinned-memory-limit"],
                         capture_output=True, text=True, env=env).stdout.strip()
    return ["--gpus", f"device={a.gpu}", "--ipc", "host", "--security-opt", "label=disable",
            "-v", f"{a.mps_root}/gpu{a.gpu}/{ns}:/run/nvidia-mps",
            "-e", "CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps", "-e", f"VITRO_MPS_NAMESPACE={ns}",
            *(["-e", f"VITRO_MEM_CAP={cap}"] if cap[:1].isdigit() else [])]



def start(i, run_root):
    c = cal.Container.__new__(cal.Container)
    run_dir = os.path.join(run_root, f"run-{i}")
    c.name, c.stats = f"vitro-util-{i}", os.path.join(run_dir, "stats")
    cal.sh("docker", "rm", "-f", c.name, check=False)
    os.makedirs(run_dir, exist_ok=True)
    cal.sh("docker", "run", "-d", "--name", c.name, *mps_args(i, a.share),
           "-v", f"{os.path.abspath(a.model)}:/models/model.onnx:ro",
           "-v", f"{run_dir}:/run/vitro", a.image)
    ip = cal.sh("docker", "inspect", "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", c.name)
    c.url = f"http://{ip}:8000"
    return c


def phase(cs, seconds, adapt=False):
    """Paced load on the containers in `cs`; nvidia-smi sampled twice a second."""
    before = {c.name: c.read() for c in cs}
    done, err, samples = {c.name: [] for c in cs}, [], []
    stop = time.time() + seconds

    def client(c):
        nxt = time.time() + random.uniform(0, c.period)
        while True:
            time.sleep(max(0, nxt - time.time()))
            if time.time() >= stop:
                return
            try:
                done[c.name].append(c.infer(body))
                if adapt and c.period:
                    c.d += 0.1 * (done[c.name][-1]["gpu_ms"] - c.d)
                    c.period = c.d / a.target / 10
            except Exception as e:                     # noqa: BLE001
                err.append(f"{c.name}: {e}")
                return
            nxt = max(nxt + c.period * random.uniform(0.5, 1.5), time.time())

    ts = [threading.Thread(target=client, args=(c,)) for c in cs]
    t0 = time.time()
    [t.start() for t in ts]
    time.sleep(1.5)                                    # nvidia-smi looks back over the last second
    while time.time() < stop:
        samples.append(cal.gpu_util(a.gpu))
        time.sleep(0.5)
    [t.join() for t in ts]
    el = time.time() - t0
    time.sleep(2.5)                                    # busy_ms is published 2 s late
    out = {"smi_mean_pct": round(sum(samples) / len(samples), 1), "smi_max_pct": max(samples),
           "smi_min_pct": min(samples), "errors": err, "containers": {}}
    for c in cs:
        s, r = c.read(), done[c.name]
        n = len(r) or 1
        out["containers"][c.name] = {
            "util_calc_pct": round(100 * (s["busy_ms"] - before[c.name]["busy_ms"]) / (el * 1000), 1),
            "inferences": len(r), "gpu_ms": round(sum(x["gpu_ms"] for x in r) / n, 1)}
    out["util_calc_sum_pct"] = round(sum(v["util_calc_pct"] for v in out["containers"].values()), 1)
    return out


cs, res = [], {"settings": vars(a)}
run_root = tempfile.mkdtemp(prefix="vitro-util-")
try:
    cs = [start(i, run_root) for i in range(1, a.n + 1)]
    for c in cs:
        h = c.wait_ready(240)
    name, shape = next(iter(h["inputs"])), next(iter(h["inputs"].values()))["shape"]
    body = json.dumps({"inputs": {name: [[0.5] * int(shape[-1])]}}).encode()
    res["containers"] = {}
    for c in cs:                                       # d: duration of one inference, alone
        end = time.time() + 2                          # warm up: an idle GPU clocks down
        while time.time() < end:
            c.infer(body)
        d = sorted(c.infer(body)["gpu_ms"] for _ in range(21))[10]
        c.d, c.period = d, 0 if a.saturate else d / (a.target / 100) / 1000
        res["containers"][c.name] = {"sm_count": c.read()["sm_count"]}
    time.sleep(2)
    res["idle_smi_pct"] = cal.gpu_util(a.gpu)
    res["solo"] = {c.name: phase([c], a.seconds / 3, adapt=True) for c in cs}
    for c in cs:
        res["containers"][c.name]["period_ms"] = round(c.period * 1000, 1)
    res["together"] = phase(cs, a.seconds)
except Exception as e:                                 # noqa: BLE001
    res["error"] = str(e)
finally:
    for c in cs:
        c.remove()

if a.json:
    with open(a.json, "w") as f:
        json.dump(res, f, indent=2)
if "error" in res:
    sys.exit(f"ERROR: {res['error']}")
print(f"{a.n} containers, share {a.share} % ({next(iter(res['containers'].values()))['sm_count']} SMs), "
      + ("full load" if a.saturate else f"paced to {a.target:g} % each") + f"; idle GPU at {res['idle_smi_pct']:g} %")
solo_sum = 0
for n, r in res["solo"].items():
    k = r["containers"][n]
    solo_sum += r["smi_mean_pct"]
    print(f"  alone    {n}: nvidia-smi {r['smi_mean_pct']} % (min {r['smi_min_pct']:g}, max {r['smi_max_pct']:g}); "
          f"calculated {k['util_calc_pct']} %; {k['gpu_ms']} ms per inference, "
          f"period {res['containers'][n]['period_ms']} ms")
t = res["together"]
print(f"  together: nvidia-smi {t['smi_mean_pct']} % (min {t['smi_min_pct']:g}, max {t['smi_max_pct']:g}); "
      f"sum of the solo figures {solo_sum:.1f} %; calculated per container "
      + ", ".join(f"{v['util_calc_pct']} %" for v in t["containers"].values())
      + f" = {t['util_calc_sum_pct']} %; ms per inference "
      + ", ".join(f"{v['gpu_ms']}" for v in t["containers"].values()))
if t["errors"]:
    print("  errors:", t["errors"])
