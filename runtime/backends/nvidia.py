"""NVIDIA backend: MPS active-thread percentage + MPS pinned device-memory limit."""
import math, os, subprocess
import docker
from .base import Allocation, Backend

MPS_ROOT = "/var/run/nvidia-mps"
SM_COUNT_HELPER = os.environ.get("SM_COUNT_HELPER", "/opt/iwmr/tools/mps_sm_count.py")


def _pipe_dir(gpu):
    return f"{MPS_ROOT}/gpu{gpu}"


class NvidiaBackend(Backend):
    vendor = "nvidia"
    MAX_CLIENTS = 48            # per MPS server on Volta+; check your CUDA version

    def __init__(self):
        self.allocs = {}        # workload_id -> Allocation
        self._inv = None        # static topology, discovered once

    def inventory(self):
        if self._inv is None:
            self._inv = self._discover()
        return [dict(g) for g in self._inv]

    def _discover(self):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        gpus = []
        for line in out.strip().splitlines():
            idx, uuid, name, mem_mib, pci = [s.strip() for s in line.split(",")]
            gpus.append({"gpu": int(idx), "uuid": uuid, "name": name,
                         "compute_units": self._sm_count(int(idx)),
                         "mem_bytes": int(mem_mib) * 2**20,
                         "fault_domain": f"mps-gpu{idx}",
                         "pci": pci.lower()[-12:]})   # "0000:3b:00.0" form
        return gpus

    def _sm_count(self, gpu):
        # Full SM count: query as an MPS client with a 100 % limit.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   CUDA_MPS_PIPE_DIRECTORY=_pipe_dir(gpu),
                   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100")
        r = subprocess.run(["python3", SM_COUNT_HELPER], env=env,
                           capture_output=True, text=True)
        return int(r.stdout.strip() or 0)

    def allocate(self, req, gpu):
        on_gpu = [a for a in self.allocs.values() if a.gpu == gpu]
        if len(on_gpu) >= self.MAX_CLIENTS:
            raise ValueError("MPS client limit reached")
        pct = max(1, min(100, math.ceil(req["compute_share"] * 100)))
        a = Allocation(req["workload_id"], gpu, pct / 100, int(req["mem_bytes"]),
                       {"active_thread_percentage": pct})
        self.allocs[a.workload_id] = a
        return a

    def release(self, workload_id):
        self.allocs.pop(workload_id, None)

    def container_kwargs(self, req, alloc):
        pipe = _pipe_dir(alloc.gpu)
        mem_mib = alloc.mem_bytes // 2**20
        return {
            "device_requests": [docker.types.DeviceRequest(
                device_ids=[str(alloc.gpu)], capabilities=[["gpu"]])],
            "ipc_mode": "host",                              # required for MPS clients
            "volumes": {pipe: {"bind": pipe, "mode": "rw"}},
            "environment": {
                "CUDA_MPS_PIPE_DIRECTORY": pipe,
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(alloc.detail["active_thread_percentage"]),
                # device index as seen INSIDE the container (one GPU -> 0)
                "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": f"0={mem_mib}M",
            },
        }

    def health(self, gpu):
        if not os.path.isdir(_pipe_dir(gpu)):
            return {"ok": False, "reason": "no MPS daemon for this GPU"}
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=_pipe_dir(gpu))
        r = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n",
                           env=env, capture_output=True, text=True, timeout=5)
        return {"ok": r.returncode == 0, "mps_servers": r.stdout.split()}
