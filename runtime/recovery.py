"""recovery.py - decide what happens after a workload container exits.

Rules are fixed (no learning):
  ok/stopped        -> nothing
  oom               -> raise memory cap x1.25 (up to the GPU size), re-place
  gpu_fault_victim  -> re-place at once (router replays its in-flight requests)
  gpu_fault_culprit -> DROPPED: container not restarted, image approval revoked
                       (it can only return after re-validation); its in-flight
                       requests are failed back to the clients
  device_error      -> drain the GPU, re-place
  app_error/unknown -> victim if the same GPU had a culprit in the incident
                       window, otherwise restart with exponential back-off
Kernel-log attribution (faults.py) can upgrade an exit to culprit.
"""
import threading, time
from collections import defaultdict

WINDOW_S = 15
OOM_GROWTH = 1.25
MAX_RESTARTS = 5


class Recovery:
    def __init__(self, ledger, registry, place_fn, validator):
        self.ledger, self.registry, self.place = ledger, registry, place_fn
        self.validator = validator
        self.recent = defaultdict(list)       # gpu -> [(ts, wid, class)]
        self.kernel_culprits = {}             # wid -> ts (from faults.py)
        self.events = []                      # audit trail, exposed in /state
        self.lock = threading.Lock()

    def mark_kernel_culprit(self, wid):
        self.kernel_culprits[wid] = time.time()

    def on_exit(self, wid, gpu, report):
        rec = self.registry.get(wid)
        if rec is None:
            return
        req, now = rec["req"], time.time()
        cls = report.get("class", "unknown")
        if now - self.kernel_culprits.pop(wid, 0) < WINDOW_S and cls != "oom":
            cls = "gpu_fault_culprit"                    # kernel log named this process
        with self.lock:
            self.recent[gpu] = [r for r in self.recent[gpu] if now - r[0] < WINDOW_S]
            self.recent[gpu].append((now, wid, cls))
            incident = {c for _, _, c in self.recent[gpu]}

        if cls in ("ok", "stopped"):
            rec["status"] = "stopped"
            return self._log(wid, cls, "no action")
        if cls == "oom":
            cap = self.ledger.gpus[gpu].mem_bytes
            req["mem_bytes"] = min(int(req["mem_bytes"] * OOM_GROWTH), cap)
            return self._restart(wid, cls, "memory cap raised")
        if cls == "gpu_fault_culprit":
            self.validator.revoke(req.get("image_digest", req["image"]))
            rec["status"] = "dropped"                    # router fails its pending requests
            return self._log(wid, cls, "culprit dropped; image approval revoked")
        if cls == "gpu_fault_victim" or "gpu_fault_culprit" in incident:
            return self._restart(wid, cls, "victim of a co-located fault")
        if cls == "device_error":
            self.ledger.gpus[gpu].healthy = False
            return self._restart(wid, cls, f"gpu {gpu} drained")
        return self._restart(wid, cls, "application error", backoff=True)

    def _restart(self, wid, cls, reason, backoff=False):
        rec = self.registry[wid]
        rec["status"] = "recovering"
        rec["restarts"] = rec.get("restarts", 0) + 1
        self._log(wid, cls, reason)
        if rec["restarts"] > MAX_RESTARTS:
            rec["status"] = "failed"
            return self._log(wid, cls, "restart limit reached")
        delay = min(30, 2 ** rec["restarts"]) if backoff else 0
        last = ""
        for attempt in range(6):                       # wait for capacity (e.g. after a drain)
            time.sleep(delay)
            try:
                self.place(rec["req"])
                return self._log(wid, cls, "re-placed")
            except RuntimeError as e:
                delay = 10
                last = str(e)
        rec["status"] = "failed"
        return self._log(wid, cls, f"no capacity: {last}")

    def _log(self, wid, cls, action):
        self.events.append({"ts": time.time(), "workload_id": wid, "class": cls, "action": action})
        del self.events[:-200]
        return action
