"""AMD backend: disjoint CU sets via HSA_CU_MASK + hipMalloc interposer for memory."""
import glob, grp, math, os
from .base import Allocation, Backend


def _has_group(name):
    try:
        grp.getgrnam(name)
        return True
    except KeyError:
        return False

KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"
MEMCAP_LIB = "/opt/gpushare/lib/libhipmemcap.so"


def _props(path):
    out = {}
    for line in open(path):
        k, _, v = line.partition(" ")
        if v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


class AmdBackend(Backend):
    vendor = "amd"

    def __init__(self):
        self.allocs = {}                 # workload_id -> Allocation
        self.gpus = self._discover()     # gpu index -> topology info
        self.free = {g: [True] * info["compute_units"] for g, info in self.gpus.items()}

    def _discover(self):
        gpus, idx = {}, 0
        for node in sorted(glob.glob(f"{KFD_NODES}/*"), key=lambda p: int(p.rsplit("/", 1)[1])):
            p = _props(f"{node}/properties")
            if p.get("simd_count", 0) == 0:
                continue                                  # CPU node
            minor = p["drm_render_minor"]
            try:
                vram = int(open(f"/sys/class/drm/renderD{minor}/device/mem_info_vram_total").read())
            except OSError:
                vram = 0
            pci = os.path.basename(os.path.realpath(f"/sys/class/drm/renderD{minor}/device"))
            gpus[idx] = {"render_minor": minor, "pci": pci.lower()[-12:],
                         "compute_units": p["simd_count"] // p["simd_per_cu"],
                         "num_xcc": p.get("num_xcc", 1), "mem_bytes": vram}
            idx += 1
        return gpus

    def inventory(self):
        return [{"gpu": g, "uuid": f"renderD{i['render_minor']}", "name": "amd-gpu",
                 "compute_units": i["compute_units"], "mem_bytes": i["mem_bytes"],
                 "free_compute_units": sum(self.free[g]),
                 "fault_domain": f"amd-gpu{g}", "pci": i["pci"]} for g, i in self.gpus.items()]

    def _find_range(self, gpu, n):
        """First-fit contiguous CU range, preferring one that stays inside one XCC."""
        free, total = self.free[gpu], len(self.free[gpu])
        per_xcc = total // self.gpus[gpu]["num_xcc"]
        starts = range(0, total - n + 1)
        if n <= per_xcc:   # try ranges that do not cross an XCC boundary first
            starts = sorted(starts, key=lambda s: (s // per_xcc != (s + n - 1) // per_xcc, s))
        for s in starts:
            if all(free[s:s + n]):
                return s
        raise ValueError("no contiguous CU range free (fragmentation)")

    def allocate(self, req, gpu):
        total = self.gpus[gpu]["compute_units"]
        n = max(2, math.ceil(req["compute_share"] * total))
        n += n % 2                                         # keep CU pairs together
        start = self._find_range(gpu, n)
        for c in range(start, start + n):
            self.free[gpu][c] = False
        a = Allocation(req["workload_id"], gpu, n / total, int(req["mem_bytes"]),
                       {"cu_first": start, "cu_last": start + n - 1})
        self.allocs[a.workload_id] = a
        return a

    def release(self, workload_id):
        a = self.allocs.pop(workload_id, None)
        if a:
            for c in range(a.detail["cu_first"], a.detail["cu_last"] + 1):
                self.free[a.gpu][c] = True

    def container_kwargs(self, req, alloc):
        minor = self.gpus[alloc.gpu]["render_minor"]
        return {
            "devices": ["/dev/kfd:/dev/kfd", f"/dev/dri/renderD{minor}:/dev/dri/renderD{minor}"],
            # numeric host GIDs: group names may not exist inside the image
            "group_add": [str(grp.getgrnam(g).gr_gid) for g in ("video", "render") if _has_group(g)],
            "volumes": {MEMCAP_LIB: {"bind": MEMCAP_LIB, "mode": "ro"}},
            "environment": {
                # one GPU visible inside the container -> device index 0
                "HSA_CU_MASK": f"0:{alloc.detail['cu_first']}-{alloc.detail['cu_last']}",
                "HIPMEM_LIMIT_BYTES": str(alloc.mem_bytes),
                "LD_PRELOAD": MEMCAP_LIB,
            },
        }
