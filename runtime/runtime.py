#!/usr/bin/env python3
"""runtime.py - Inference Workload Management Runtime (IWMR).

One container per GPU node. It is the only management component: it exposes
the API, keeps the workload registry, admits and places workloads (best-fit),
launches inference containers with vendor-specific limits, routes inference
requests with an in-flight journal, watches exits and kernel-log faults,
recovers workloads, samples telemetry and validates images.

  docker run ... iwmr      (see deploy/run_runtime.sh for the required flags)
  API: http://<node>:8080/docs
"""
import json, os, threading, time
import anyio, docker, requests
from fastapi import FastAPI, HTTPException, Request

from backends.nvidia import NvidiaBackend
from backends.amd import AmdBackend
from ledger import Ledger
from router import Router
from recovery import Recovery
from faults import FaultMonitor
from telemetry import Telemetry
from validator import Validator

VENDOR = os.environ.get("GPUSHARE_VENDOR", "nvidia")
NETWORK = os.environ.get("GPUSHARE_NETWORK", "gpushare")
STATE = "/var/lib/gpushare"                  # same path on host and in the runtime
LABEL = "gpushare.workload_id"
VALIDATION_GPU = int(os.environ["VALIDATION_GPU"]) if os.environ.get("VALIDATION_GPU") else None
READY_TIMEOUT_S = int(os.environ.get("READY_TIMEOUT_S", "600"))
REQUIRED = ("workload_id", "image", "compute_share", "mem_bytes")

backend = NvidiaBackend() if VENDOR == "nvidia" else AmdBackend()
dock = docker.from_env()
ledger = Ledger()
registry = {}                                # wid -> {req, status, gpu, container, allocation}
lock = threading.RLock()
app = FastAPI(title="Inference Workload Management Runtime")

amd_minor = (backend.gpus[VALIDATION_GPU]["render_minor"]
             if VENDOR == "amd" and VALIDATION_GPU is not None else None)
validator = Validator(VENDOR, VALIDATION_GPU, amd_minor)
router = Router(registry, lambda wid: f"http://gs-{wid}:{registry[wid]['req'].get('port', 8000)}")
telemetry = Telemetry(VENDOR, backend)


# ------------------------------------------------------------------ placement and launch
def launch(req, alloc):
    wid = req["workload_id"]
    status_dir = f"{STATE}/{wid}"
    os.makedirs(status_dir, exist_ok=True)
    try:
        os.remove(f"{status_dir}/exit.json")
    except FileNotFoundError:
        pass
    try:                                                 # leftover from a previous run
        dock.containers.get(f"gs-{wid}").remove(force=True)
    except docker.errors.NotFound:
        pass
    kw = backend.container_kwargs(req, alloc)
    kw["environment"].update(req.get("env", {}), GPUSHARE_WORKLOAD_ID=wid)
    kw["volumes"][status_dir] = {"bind": "/run/gpushare", "mode": "rw"}
    for host, cont in req.get("volumes", {}).items():    # e.g. read-only model weights
        kw["volumes"][host] = {"bind": cont, "mode": "ro"}
    c = dock.containers.run(req["image"], command=req.get("command"), detach=True,
                            name=f"gs-{wid}", network=NETWORK,
                            labels={LABEL: wid, "gpushare.gpu": str(alloc.gpu)},
                            restart_policy={"Name": "no"}, **kw)
    return c.id


def place(req):
    """Admission + best-fit placement + launch. Raises RuntimeError if nothing fits."""
    wid = req["workload_id"]
    with lock:
        for g in ledger.candidates(req):
            try:
                alloc = backend.allocate(req, g)         # AMD may refuse: CU fragmentation
            except ValueError:
                continue
            try:
                cid = launch(req, alloc)
            except docker.errors.APIError as e:
                backend.release(wid)
                raise RuntimeError(f"docker run failed: {e}")
            ledger.commit(req, g, alloc.compute_share)
            rec = registry.setdefault(wid, {})
            rec.update(req=req, status="starting", gpu=g, container=cid,
                       allocation=alloc.__dict__, started=time.time())
            threading.Thread(target=wait_ready, args=(wid, cid), daemon=True).start()
            return rec
        raise RuntimeError(f"no feasible GPU: {ledger.explain(req)}")


def wait_ready(wid, cid):
    t0 = time.time()
    while time.time() - t0 < READY_TIMEOUT_S:
        rec = registry.get(wid)
        if not rec or rec.get("container") != cid or rec["status"] != "starting":
            return
        try:
            if requests.get(router.host(wid) + "/health", timeout=2).ok:
                rec["status"] = "ready"
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    if registry.get(wid, {}).get("container") == cid:
        registry[wid]["status"] = "unhealthy"


# ------------------------------------------------------------------ background watchers
def watch_exits():
    for ev in dock.events(decode=True, filters={"type": "container", "event": "die", "label": LABEL}):
        a = ev["Actor"]["Attributes"]
        wid, gpu = a.get(LABEL), int(a.get("gpushare.gpu", -1))
        try:
            report = json.load(open(f"{STATE}/{wid}/exit.json"))
        except (OSError, ValueError):
            report = {"class": "unknown", "exit_code": int(a.get("exitCode", -1))}
        with lock:
            backend.release(wid)
            ledger.release(wid)
            rec = registry.get(wid)
        try:
            dock.containers.get(ev["id"]).remove()
        except docker.errors.APIError:
            pass
        if rec is None or rec.get("container") != ev["id"]:
            continue
        if rec["status"] == "stopping":
            registry.pop(wid, None)
            continue
        threading.Thread(target=recovery.on_exit, args=(wid, gpu, report), daemon=True).start()


def on_kernel_fault(evt):
    if evt["device_level"]:
        g = ledger.gpu_of_pci(evt["pci"])
        if g is not None:
            ledger.gpus[g].healthy = False               # drain; containers exit on their own
        recovery._log(None, "device_error", f"kernel: {evt['raw'][:120]}")
        return
    if evt.get("container"):
        try:
            wid = dock.containers.get(evt["container"]).labels.get(LABEL)
        except docker.errors.NotFound:
            wid = None
        if wid:
            recovery.mark_kernel_culprit(wid)


def watch_health(period=10):
    while True:
        for g in list(ledger.gpus):
            h = backend.health(g)
            if not h.get("ok") and ledger.gpus[g].healthy:
                ledger.gpus[g].healthy = False
                recovery._log(None, "backend", f"gpu {g} unhealthy: {h}")
        time.sleep(period)


recovery = Recovery(ledger, registry, place, validator)


@app.on_event("startup")
def startup():
    os.makedirs(STATE, exist_ok=True)
    for g in backend.inventory():
        if g["gpu"] != VALIDATION_GPU:
            ledger.add_gpu(g["gpu"], g["compute_units"], g["mem_bytes"], g.get("pci", ""))
    for c in dock.containers.list(all=True, filters={"label": LABEL}):
        c.remove(force=True)                             # registry is not persistent: start clean
    for t in (watch_exits, watch_health):
        threading.Thread(target=t, daemon=True).start()
    FaultMonitor(on_kernel_fault).start()
    telemetry.start()


# ------------------------------------------------------------------ API
@app.post("/workloads")
def submit(req: dict):
    missing = [k for k in REQUIRED if k not in req]
    if missing:
        raise HTTPException(422, f"missing fields: {missing}")
    if req["workload_id"] in registry and registry[req["workload_id"]]["status"] not in ("dropped", "failed", "stopped"):
        raise HTTPException(409, "workload_id already exists")
    if req.get("isolation", "shared") == "shared" and not validator.approved(req.get("image_digest")):
        raise HTTPException(403, "image digest not validated; validate it or use isolation=exclusive")
    try:
        return place(req)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/workloads")
def list_workloads():
    return {w: {k: r.get(k) for k in ("status", "gpu", "allocation", "restarts")}
            for w, r in registry.items()}


@app.get("/workloads/{wid}")
def get_workload(wid: str):
    if wid not in registry:
        raise HTTPException(404)
    return registry[wid]


@app.delete("/workloads/{wid}")
def stop(wid: str):
    rec = registry.get(wid)
    if not rec:
        raise HTTPException(404)
    rec["status"] = "stopping"
    try:
        dock.containers.get(f"gs-{wid}").stop(timeout=20)
    except docker.errors.NotFound:
        registry.pop(wid, None)
    return {"stopping": wid}


@app.post("/workloads/{wid}/infer")
async def infer(wid: str, request: Request):
    if wid not in registry:
        raise HTTPException(404)
    payload = await request.json()
    code, body = await anyio.to_thread.run_sync(
        router.forward, wid, payload, request.headers.get("X-Request-ID"))
    if code >= 400:
        raise HTTPException(code, body)
    return body


@app.post("/images/validate")
def validate(body: dict):
    try:
        return {"job": validator.submit(body["image"])}
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/images/validate/{job}")
def validation_job(job: str):
    return validator.jobs.get(job) or {}


@app.get("/state")
def state():
    gpus = {}
    for g, s in ledger.gpus.items():
        share, mem = s.used()
        t = telemetry.latest.get(g, {})
        gpus[g] = {"healthy": s.healthy, "share_used": round(share, 3),
                   "mem_reserved": mem, "workloads": list(s.workloads), **t,
                   "unexpected_use": t.get("mem_used", 0) > mem + 2**30}
    return {"vendor": VENDOR, "gpus": gpus, "validation_gpu": VALIDATION_GPU,
            "inflight_requests": router.inflight(), "recent_events": recovery.events[-20:]}


@app.post("/gpus/{g}/undrain")
def undrain(g: int):
    ledger.gpus[g].healthy = True
    return {"gpu": g, "healthy": True}


@app.get("/health")
def health():
    return {"ok": True, "vendor": VENDOR, "workloads": len(registry)}
