"""router.py - inference request router with an in-flight journal.

Clients send inference requests to the runtime, never to containers directly.
Each request is journaled until its response returns. If the target container
dies (fault, OOM exit, neighbour's MPS teardown), the request stays in the
journal, the router waits for the recovery controller to bring the workload
back, and replays it. Clients see added latency instead of an error.
"""
import threading, time, uuid
import requests

RECOVERY_WAIT_S = 180          # how long a request may wait for its workload to return
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 120


class Router:
    def __init__(self, registry, network_host):
        self.registry = registry          # wid -> record with "status", "req"
        self.host = network_host          # callable(wid) -> "http://gs-<wid>:<port>"
        self.journal = {}                 # request_id -> {wid, ts, attempts}
        self.lock = threading.Lock()

    def inflight(self):
        with self.lock:
            out = {}
            for r in self.journal.values():
                out[r["wid"]] = out.get(r["wid"], 0) + 1
            return out

    def _wait_ready(self, wid):
        t0 = time.time()
        while time.time() - t0 < RECOVERY_WAIT_S:
            rec = self.registry.get(wid)
            if rec is None or rec["status"] in ("failed", "stopping", "dropped"):
                return False
            if rec["status"] == "ready":
                return True
            time.sleep(0.5)
        return False

    def forward(self, wid, payload, request_id=None):
        rid = request_id or str(uuid.uuid4())
        with self.lock:
            self.journal[rid] = {"wid": wid, "ts": time.time(), "attempts": 0}
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                self.journal[rid]["attempts"] = attempt
                if not self._wait_ready(wid):
                    return 503, {"error": "workload unavailable", "request_id": rid}
                path = self.registry[wid]["req"].get("infer_path", "/infer")
                try:
                    r = requests.post(self.host(wid) + path, json=payload, timeout=REQUEST_TIMEOUT_S,
                                      headers={"X-Request-ID": rid})
                except requests.ConnectionError:
                    time.sleep(1)           # container gone: the exit watcher starts recovery
                    continue
                if r.status_code == 503 and r.headers.get("Retry-After"):
                    time.sleep(float(r.headers["Retry-After"]))   # OOM back-pressure
                    continue
                return r.status_code, r.json()
            return 503, {"error": "retries exhausted", "request_id": rid}
        finally:
            with self.lock:
                self.journal.pop(rid, None)
