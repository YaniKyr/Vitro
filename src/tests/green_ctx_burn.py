#!/usr/bin/env python3
"""Split the GPU into SM groups with green contexts and burn in all of them at once.

Usage:  python green_ctx_burn.py            # 8 groups, at least 2 SMs each
        python green_ctx_burn.py 16 1       # 16 groups, at least 1 SM each
        python green_ctx_burn.py 16 1 --ignore-cosched   # same, lifting the co-scheduling rule

One process, one thread per group. Each group runs the same kernel as mps_burn.py,
first alone and then together with all the others, and records which SMs its
blocks ran on (%smid). Without MPS these are physical SM ids, so the table shows
whether the groups really are disjoint.
"""
import ctypes, sys, threading, time

from mps_burn import PTX, ITERS, BLOCKS, THREADS, HIST_SLOTS, ranges

SOLO_S, TOGETHER_S = 3, 10
SM, DEFAULT_STREAM, NON_BLOCKING, IGNORE_COSCHED = 1, 1, 1, 1


class DevResource(ctypes.Structure):   # CUdevResource in cuda.h, 144 bytes (ABI v1)
    _fields_ = [("type", ctypes.c_int),
                ("_internal_padding", ctypes.c_ubyte * 92),
                ("smCount", ctypes.c_uint),
                ("minSmPartitionSize", ctypes.c_uint),
                ("smCoscheduledAlignment", ctypes.c_uint),
                ("_oversize", ctypes.c_ubyte * 36)]


cu = ctypes.CDLL("libcuda.so.1")


def call(name, *args):
    rc = getattr(cu, name)(*args)
    if rc:
        msg = ctypes.c_char_p()
        cu.cuGetErrorString(rc, ctypes.byref(msg))
        sys.exit(f"{name} failed: {msg.value.decode() if msg.value else rc}")


class Group:
    """One green context with its own stream, kernel and SM histogram."""

    def __init__(self, dev, resource):
        desc, self.green, self.ctx = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        self.stream, mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        out, self.hist, got = ctypes.c_uint64(), ctypes.c_uint64(), DevResource()
        call("cuDevResourceGenerateDesc", ctypes.byref(desc), ctypes.byref(resource), 1)
        call("cuGreenCtxCreate", ctypes.byref(self.green), desc, dev, DEFAULT_STREAM)
        call("cuGreenCtxGetDevResource", self.green, ctypes.byref(got), SM)
        self.sms = got.smCount
        call("cuCtxFromGreenCtx", ctypes.byref(self.ctx), self.green)
        call("cuGreenCtxStreamCreate", ctypes.byref(self.stream), self.green, NON_BLOCKING, 0)
        call("cuCtxPushCurrent_v2", self.ctx)
        call("cuModuleLoadData", ctypes.byref(mod), ctypes.c_char_p(PTX))
        call("cuModuleGetFunction", ctypes.byref(self.fn), mod, b"burn")
        call("cuMemAlloc_v2", ctypes.byref(out), ctypes.c_size_t(4))
        call("cuMemAlloc_v2", ctypes.byref(self.hist), ctypes.c_size_t(4 * HIST_SLOTS))
        call("cuMemsetD32_v2", self.hist, 0, ctypes.c_size_t(HIST_SLOTS))
        call("cuCtxPopCurrent_v2", None)
        self.out, self.iters = ctypes.c_uint64(out.value), ctypes.c_uint32(ITERS)
        self.args = (ctypes.c_void_p * 3)(ctypes.addressof(self.out), ctypes.addressof(self.iters),
                                          ctypes.addressof(self.hist))
        self.ms = {}

    def burn(self, label, seconds, start=None):
        """Launch the kernel in this group's stream for `seconds`; store ms per launch."""
        call("cuCtxSetCurrent", self.ctx)
        if start:
            start.wait()
        launches, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            call("cuLaunchKernel", self.fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, self.stream, self.args, None)
            call("cuStreamSynchronize", self.stream)
            launches += 1
            if launches == 1 and label == "alone":
                self.first = self.smids()              # SMs of a single launch
        self.ms[label] = 1000 * (time.time() - t0) / launches

    def smids(self):
        counts = (ctypes.c_uint32 * HIST_SLOTS)()
        call("cuCtxSetCurrent", self.ctx)
        call("cuMemcpyDtoH_v2", counts, self.hist, ctypes.c_size_t(4 * HIST_SLOTS))
        return {i for i, n in enumerate(counts) if n}


def main(want=8, min_sms=2, flags=0):
    dev, ctx, name, whole = ctypes.c_int(), ctypes.c_void_p(), ctypes.create_string_buffer(256), DevResource()
    call("cuInit", 0)
    call("cuDeviceGet", ctypes.byref(dev), 0)
    call("cuDeviceGetName", name, len(name), dev)
    call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
    call("cuCtxSetCurrent", ctx)
    call("cuDeviceGetDevResource", dev, ctypes.byref(whole), SM)
    print(f"{name.value.decode()}: {whole.smCount} SMs, minimum partition {whole.minSmPartitionSize},"
          f" alignment {whole.smCoscheduledAlignment}")

    parts, n, rest = (DevResource * want)(), ctypes.c_uint(want), DevResource()
    call("cuDevSmResourceSplitByCount", parts, ctypes.byref(n), ctypes.byref(whole),
         ctypes.byref(rest), flags, min_sms)
    print(f"asked for {want} groups of >= {min_sms} SMs: driver made {n.value}, {rest.smCount} SMs left over")

    groups = [Group(dev, parts[i]) for i in range(n.value)]

    for g in groups:                                   # each group with the GPU to itself
        t = threading.Thread(target=g.burn, args=("alone", SOLO_S))
        t.start(); t.join()

    start = threading.Event()                          # all groups at the same time
    threads = [threading.Thread(target=g.burn, args=("together", TOGETHER_S, start)) for g in groups]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    print("\ngroup  SMs  ids in 1 launch  ids over run  ms/launch alone  together  slowdown")
    seen, overlap = set(), set()
    for i, g in enumerate(groups):
        ids = g.smids()
        overlap |= seen & ids
        seen |= ids
        print(f"{i:>5}  {g.sms:>3}  {ranges(g.first):>15}  {ranges(ids):>12}  {g.ms['alone']:>15.2f}  {g.ms['together']:>8.2f}"
              f"  {100 * (g.ms['together'] / g.ms['alone'] - 1):>+7.1f}%")
    print(f"\nSMs used by more than one group: {ranges(overlap)}")
    print(f"launches per second, all groups together: {sum(1000 / g.ms['together'] for g in groups):.0f}")


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(*nums[:2], flags=IGNORE_COSCHED if "--ignore-cosched" in sys.argv else 0)
