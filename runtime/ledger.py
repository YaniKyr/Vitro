"""ledger.py - admission and deterministic placement for the GPUs of this node.

MPS percentages, CU masks and memory caps are ceilings, not reservations. The
ledger admits a workload on a GPU only if the sum of granted compute shares
stays <= 1 and the sum of memory caps plus per-client context overhead stays
within device memory. Under that rule the ceilings behave as reservations.

Placement is plain best-fit: among feasible GPUs, take the one that is left
with the least free compute share (ties: lowest index). No learning, no scoring.
"""
import os
from dataclasses import dataclass, field

CTX_OVERHEAD = int(os.environ.get("CTX_OVERHEAD_MB", "512")) * 2**20   # measure per GPU model
COMPUTE_HEADROOM = float(os.environ.get("COMPUTE_HEADROOM", "0.0"))     # e.g. 0.1 kept free
MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS_PER_GPU", "48"))


@dataclass
class Gpu:
    index: int
    compute_units: int
    mem_bytes: int
    pci: str = ""
    healthy: bool = True
    workloads: dict = field(default_factory=dict)  # wid -> (share, mem, exclusive)

    def used(self):
        return (sum(w[0] for w in self.workloads.values()),
                sum(w[1] + CTX_OVERHEAD for w in self.workloads.values()))


class Ledger:
    def __init__(self):
        self.gpus = {}

    def add_gpu(self, index, compute_units, mem_bytes, pci=""):
        self.gpus[index] = Gpu(index, compute_units, mem_bytes, pci)

    def fits(self, req, g):
        gpu = self.gpus[g]
        if not gpu.healthy:
            return False, "gpu unhealthy (drained)"
        if any(w[2] for w in gpu.workloads.values()):
            return False, "gpu held by an exclusive workload"
        if req.get("isolation") == "exclusive" and gpu.workloads:
            return False, "exclusive placement needs an empty gpu"
        if len(gpu.workloads) >= MAX_CLIENTS:
            return False, "client limit"
        share, mem = gpu.used()
        if share + req["compute_share"] > 1.0 - COMPUTE_HEADROOM + 1e-9:
            return False, "compute share exhausted"
        if mem + req["mem_bytes"] + CTX_OVERHEAD > gpu.mem_bytes:
            return False, "device memory exhausted"
        return True, "ok"

    def candidates(self, req):
        """Feasible GPUs in best-fit order."""
        ok = [g for g in sorted(self.gpus) if self.fits(req, g)[0]]
        return sorted(ok, key=lambda g: (1.0 - self.gpus[g].used()[0] - req["compute_share"], g))

    def explain(self, req):
        return {g: self.fits(req, g)[1] for g in sorted(self.gpus)}

    def commit(self, req, g, granted_share):
        self.gpus[g].workloads[req["workload_id"]] = (
            granted_share, req["mem_bytes"], req.get("isolation") == "exclusive")

    def release(self, wid):
        for gpu in self.gpus.values():
            gpu.workloads.pop(wid, None)

    def gpu_of_pci(self, pci_fragment):
        for g in self.gpus.values():
            if g.pci and pci_fragment and pci_fragment in g.pci:
                return g.index
        return None
