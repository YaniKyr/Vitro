"""Vendor-neutral backend interface used by the runtime (IWMR)."""
from dataclasses import dataclass, field


@dataclass
class Allocation:
    workload_id: str
    gpu: int                      # host GPU index
    compute_share: float          # granted share, 0..1 (after rounding)
    mem_bytes: int                # granted device-memory cap
    detail: dict = field(default_factory=dict)   # vendor specifics (e.g. CU set)


class Backend:
    vendor = "base"

    def inventory(self) -> list:
        """[{gpu, uuid, name, compute_units, mem_bytes, fault_domain}, ...]"""
        raise NotImplementedError

    def allocate(self, req: dict, gpu: int) -> Allocation:
        """Reserve resources on `gpu` for request `req`; raise ValueError if impossible."""
        raise NotImplementedError

    def release(self, workload_id: str) -> None:
        raise NotImplementedError

    def container_kwargs(self, req: dict, alloc: Allocation) -> dict:
        """docker-py run() kwargs that enforce the allocation inside the container."""
        raise NotImplementedError

    def health(self, gpu: int) -> dict:
        """Backend health for one GPU (e.g. MPS daemon reachable)."""
        return {"ok": True}
