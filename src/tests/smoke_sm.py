#!/usr/bin/env python3
"""smoke_sm.py - smoke test: inference containers that demand a number of SMs.

Each number on the command line is one container under MPS that demands that
many SMs. The test starts them, checks that each got the SMs it asked for and
that inference works, and measures the inference speed (time until one pass
through the model is completed).

  smoke_sm.py --model m.onnx 14        one container with 14 SMs
  smoke_sm.py --model m.onnx 2         one container with 2 SMs
  smoke_sm.py --model m.onnx 14 2      both at once, sharing the GPU

Reported per container:
  ready_s      docker run -> model loaded and server answering
  first_ms     first inference (sets up cuBLAS/cuDNN; not in the figures below)
  latency_ms   request sent -> result received, one request at a time
  model_ms     the same pass as timed inside the server (wall) and on the GPU
  inf_per_s    completed inferences per second

Exit status: 0 if every container got its SMs and every inference succeeded.
Needs the onnx-infer image and an MPS daemon for the GPU (see deploy/).
"""
import argparse, json, subprocess, math, os, statistics, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "iwmr"))
import calibrate as cal                                # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("sms", type=int, nargs="+", help="SMs demanded, one container each")
p.add_argument("--model", required=True)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--mem-mib", type=int, default=400)
p.add_argument("--seconds", type=float, default=10)
p.add_argument("--image", default="onnx-infer:latest")
p.add_argument("--mps-root", default="/var/run/nvidia-mps")
p.add_argument("--ns", help="MPS v3: comma list of <server>/<namespace>, one per container; "
               "limits then come from the namespace, not from the environment")
p.add_argument("--json", help="also write the result to this file")
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



def start(i, sms, share, run_root):
    c = cal.Container.__new__(cal.Container)
    run_dir = os.path.join(run_root, f"run-{i}")
    c.name, c.stats, c.sms, c.share = f"vitro-smoke-{sms}sm-{i}", os.path.join(run_dir, "stats"), sms, share
    cal.sh("docker", "rm", "-f", c.name, check=False)
    os.makedirs(run_dir, exist_ok=True)
    c.t_start = time.time()
    cal.sh("docker", "run", "-d", "--name", c.name, *mps_args(i, share),
           "-v", f"{os.path.abspath(a.model)}:/models/model.onnx:ro",
           "-v", f"{run_dir}:/run/vitro", a.image)
    ip = cal.sh("docker", "inspect", "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", c.name)
    c.url = f"http://{ip}:8000"
    return c


def timed(c, body):
    t = time.perf_counter()
    m = c.infer(body)
    return (time.perf_counter() - t) * 1000, m


def run(c, body, stop, out):
    lat, wall, gpu = [], [], []
    t0 = time.time()
    try:
        while time.time() < stop:
            ms, m = timed(c, body)
            lat.append(ms), wall.append(m["wall_ms"]), gpu.append(m["gpu_ms"])
    except Exception as e:                             # noqa: BLE001 - reported as a failure
        out["error"] = str(e)
    el = time.time() - t0
    if lat:
        q = statistics.quantiles(lat, n=20) if len(lat) > 1 else lat * 19
        out.update(inferences=len(lat), inf_per_s=round(len(lat) / el, 2),
                   latency_ms={"mean": round(statistics.mean(lat), 1), "p50": round(q[9], 1),
                               "p95": round(q[18], 1), "max": round(max(lat), 1)},
                   model_ms={"wall": round(statistics.mean(wall), 1),
                             "gpu": round(statistics.mean(gpu), 1)})


total = cal.total_sms(a, a.gpu)
if total is None:
    sys.exit("could not read the SM count of the GPU through MPS")
if sum(a.sms) > total:
    print(f"note: {sum(a.sms)} SMs demanded, GPU has {total}: the containers overlap")
cs, res, ok = [], {}, True
run_root = tempfile.mkdtemp(prefix="vitro-smoke-")
try:
    for i, sms in enumerate(a.sms, 1):
        cs.append(start(i, sms, min(100, math.ceil(100 * sms / total)), run_root))
    for c in cs:
        h = c.wait_ready(240)
        res[c.name] = {"sms_demanded": c.sms, "share_pct": c.share,
                       "ready_s": round(time.time() - c.t_start, 1)}
    name, shape = next(iter(h["inputs"])), next(iter(h["inputs"].values()))["shape"]
    body = json.dumps({"inputs": {name: [[0.5] * int(shape[-1])]}}).encode()
    live = []
    for c in cs:
        try:
            res[c.name]["first_ms"] = round(timed(c, body)[0], 1)
        except Exception as e:                         # noqa: BLE001 - e.g. over the memory limit
            detail = e.read().decode()[:200] if hasattr(e, "read") else ""
            res[c.name].update(error=f"first inference: {e} {detail}".strip())
            res[c.name]["pass"] = ok = False
            continue
        live.append(c)
        s = c.read()
        res[c.name].update(sms_granted=s["sm_count"],
                           mem_cap_mib=round(s["mem_cap_bytes"] / cal.MIB),
                           mem_footprint_mib=round(s["mem_footprint_bytes"] / cal.MIB, 1))
    stop = time.time() + a.seconds
    ts = [threading.Thread(target=run, args=(c, body, stop, res[c.name])) for c in live]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for c in live:
        r = res[c.name]
        try:
            s = c.read()
            r["mem_peak_mib"] = round((s["mem_model_peak_bytes"] + s["mem_overhead_bytes"]) / cal.MIB, 1)
        except RuntimeError:
            pass
        r["pass"] = r["sms_granted"] == r["sms_demanded"] and "error" not in r and r.get("inferences", 0) > 0
        ok &= r["pass"]
except Exception as e:                                 # noqa: BLE001
    ok = False
    res["error"] = str(e)
finally:
    for c in cs:
        c.remove()

out = {"gpu_sms": total, "model": os.path.abspath(a.model), "seconds": a.seconds,
       "passed": ok, "containers": res}
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=2)
print(f"GPU {a.gpu}: {total} SMs; {len(a.sms)} container(s), {a.seconds:g} s of load")
if "error" in res:
    print("  ERROR:", res.pop("error"))
for n, r in res.items():
    if "latency_ms" not in r:
        print(f"  FAIL {n}: {r}")
        continue
    l, m = r["latency_ms"], r["model_ms"]
    print(f"  {'PASS' if r['pass'] else 'FAIL'} {n}: SMs {r['sms_granted']}/{r['sms_demanded']} "
          f"(share {r['share_pct']} %), ready {r['ready_s']} s, first {r['first_ms']} ms\n"
          f"       latency mean {l['mean']} / p50 {l['p50']} / p95 {l['p95']} / max {l['max']} ms; "
          f"in server {m['wall']} ms, on GPU {m['gpu']} ms; {r['inf_per_s']} inf/s "
          f"({r['inferences']} inferences)" + (f"; error: {r['error']}" if "error" in r else ""))
print("PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
