"""oom_handling.py - recoverable-error handling inside the inference server.

An allocation that hits the MPS / hipmemcap limit raises an OOM error but leaves
the GPU context valid, so the server can shed load instead of dying. Fatal GPU
faults are NOT caught here: the context is unusable and guard.py reports them.
Works unchanged on PyTorch CUDA and PyTorch ROCm builds.
"""
import os, time
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
model = None                              # loaded at startup from the read-only weights volume
OOM_EVENTS = 0
MAX_BATCH = int(os.environ.get("MAX_BATCH", "32"))


@app.on_event("startup")
def load():
    global model
    model = torch.load(os.environ["MODEL_PATH"], weights_only=False).eval().to("cuda")


@app.post("/infer")
def infer(payload: dict):
    global OOM_EVENTS, MAX_BATCH
    x = torch.tensor(payload["inputs"], device="cuda")
    try:
        with torch.inference_mode():
            y = model(x[:MAX_BATCH])
        return {"outputs": y.cpu().tolist()}
    except torch.OutOfMemoryError:
        OOM_EVENTS += 1
        torch.cuda.empty_cache()              # return cached blocks to the capped pool
        MAX_BATCH = max(1, MAX_BATCH // 2)    # degrade instead of crashing
        return JSONResponse({"error": "gpu_oom", "retry": True},
                            status_code=503, headers={"Retry-After": "1"})


@app.get("/health")
def health():
    # Polled by the IWMR runtime (readiness); oom_events shows memory pressure.
    return {"ok": True, "oom_events": OOM_EVENTS, "max_batch": MAX_BATCH,
            "mem_reserved": torch.cuda.memory_reserved(), "ts": time.time()}
