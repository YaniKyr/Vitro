#!/usr/bin/env python3
"""Dummy GPU workload for MPS testing (no torch needed).

Usage:  python mps_burn.py 25 50 25         # 3 concurrent clients at 25%, 50%, 25% SMs
        python mps_burn.py 25:300 50:600    # same, each client also holds 300 / 600 MiB

Each client also records which SMs its thread blocks ran on (PTX register %smid),
so the table shows how many distinct SMs a client really touched. Add --smids to
print the SM ids of every client and how evenly its blocks were spread over them.
Under MPS the ids are numbered per client (every client starts at 0), so they do
not identify physical SMs and cannot be compared between clients.

Every driver call is checked. A client that gets an error (e.g. out of memory at
its CUDA_MPS_PINNED_DEVICE_MEM_LIMIT) prints it to stderr, exits 1, and is shown
in the table with the error instead of timings.
"""
import ctypes, json, os, subprocess, sys, time

# GPU kernel: each thread loops ITERS times doing x = x*1 + 1 (dummy load).
# Thread 0 of every block first adds 1 to hist[%smid], the SM the block runs on.
PTX = rb"""
.version 7.0
.target sm_52
.address_size 64
.visible .entry burn(.param .u64 out, .param .u32 iters, .param .u64 hist)
{
  .reg .f32 %f<3>; .reg .u32 %r<6>; .reg .u64 %rd<4>; .reg .pred %p;
  ld.param.u64 %rd1, [out]; ld.param.u32 %r1, [iters]; ld.param.u64 %rd2, [hist];
  mov.u32 %r2, %tid.x; cvt.rn.f32.u32 %f1, %r2;
  setp.ne.u32 %p, %r2, 0;
  @%p bra BURN;
  mov.u32 %r4, %smid;                       // SM this block is running on
  and.b32 %r4, %r4, 1023;                   // stay inside the HIST_SLOTS table
  cvt.u64.u32 %rd3, %r4; shl.b64 %rd3, %rd3, 2; add.u64 %rd3, %rd2, %rd3;
  atom.global.add.u32 %r5, [%rd3], 1;       // hist[smid] += 1
BURN:
  mov.f32 %f2, 0f3F800000; mov.u32 %r3, 0;
LOOP:
  fma.rn.f32 %f1, %f1, %f2, %f2;
  add.u32 %r3, %r3, 1;
  setp.lt.u32 %p, %r3, %r1;
  @%p bra LOOP;
  setp.eq.f32 %p, %f1, 0fBF800000;
  @%p st.global.f32 [%rd1], %f1;
  ret;
}
"""
DURATION, ITERS, BLOCKS, THREADS = 10, 20000, 1024, 256
HIST_SLOTS = 1024                             # must match the mask in the kernel


def fail(cu, rc, what):
    """Report a driver error the way a real server would: message on stderr, exit 1."""
    msg = ctypes.c_char_p()
    cu.cuGetErrorString(rc, ctypes.byref(msg))
    text = msg.value.decode() if msg.value else f"error {rc}"
    print(f"{what}: {text}", file=sys.stderr, flush=True)
    print(json.dumps({"pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"),
                      "mem": int(os.environ.get("BURN_MEM_MIB", "0")), "error": text}))
    sys.exit(1)


def worker():
    """One MPS client: hold BURN_MEM_MIB of memory, run the kernel for DURATION seconds."""
    cu = ctypes.CDLL("libcuda.so.1")
    dev, ctx, mod, fn = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    sms, ptr, held, hist = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    counts = (ctypes.c_uint32 * HIST_SLOTS)()
    mem_mib = int(os.environ.get("BURN_MEM_MIB", "0"))

    def call(name, *args):
        rc = getattr(cu, name)(*args)
        if rc:
            fail(cu, rc, name)

    call("cuInit", 0)
    call("cuDeviceGet", ctypes.byref(dev), 0)
    call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)           # becomes MPS client here
    call("cuCtxSetCurrent", ctx)
    call("cuDeviceGetAttribute", ctypes.byref(sms), 16, dev)           # 16 = SM count
    call("cuModuleLoadData", ctypes.byref(mod), ctypes.c_char_p(PTX))  # compile kernel
    call("cuModuleGetFunction", ctypes.byref(fn), mod, b"burn")
    call("cuMemAlloc_v2", ctypes.byref(ptr), ctypes.c_size_t(4))
    if mem_mib:                                                        # dummy model memory
        call("cuMemAlloc_v2", ctypes.byref(held), ctypes.c_size_t(mem_mib * 2**20))

    call("cuMemAlloc_v2", ctypes.byref(hist), ctypes.c_size_t(4 * HIST_SLOTS))
    call("cuMemsetD32_v2", ctypes.c_uint64(hist.value), 0, ctypes.c_size_t(HIST_SLOTS))

    out, iters = ctypes.c_uint64(ptr.value), ctypes.c_uint32(ITERS)
    args = (ctypes.c_void_p * 3)(ctypes.addressof(out), ctypes.addressof(iters),
                                 ctypes.addressof(hist))

    def read_hist():
        """{SM id: blocks that ran on it so far}"""
        call("cuMemcpyDtoH_v2", counts, ctypes.c_uint64(hist.value), ctypes.c_size_t(4 * HIST_SLOTS))
        return {i: n for i, n in enumerate(counts) if n}

    launches, first, t0 = 0, {}, time.time()
    while time.time() - t0 < DURATION:
        call("cuLaunchKernel", fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, args, None)
        call("cuCtxSynchronize")
        launches += 1
        if launches == 1:
            first = read_hist()                                        # SMs of one launch
    ms = 1000 * (time.time() - t0) / max(launches, 1)

    print(json.dumps({"pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"),
                      "mem": mem_mib, "sms": sms.value, "launches": launches, "ms": round(ms, 2),
                      "first": sorted(first), "smids": read_hist()}))


def ranges(ids):
    """[0, 1, 2, 5] -> '0-2,5'"""
    out, ids = [], sorted(ids)
    while ids:
        end = 0
        while end + 1 < len(ids) and ids[end + 1] == ids[end] + 1:
            end += 1
        out.append(str(ids[0]) if end == 0 else f"{ids[0]}-{ids[end]}")
        ids = ids[end + 1:]
    return ",".join(out) or "-"


def launch(specs, show_ids=False):
    """Start one client per 'pct' or 'pct:MiB' spec, all at once, and print their results."""
    procs, used = [], []
    for spec in specs:
        pct, _, mem = spec.partition(":")
        env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=pct, BURN_MEM_MIB=mem or "0")
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker"],
                                      env=env, stdout=subprocess.PIPE, text=True))

    # SMs = ceiling reported by the driver; 1st = distinct SMs hit by the first
    # launch; used = distinct SMs hit over the whole run.
    print(" pct  mem MiB  SMs  1st  used  launches  ms/launch")
    for p in procs:
        lines = p.communicate()[0].strip().splitlines()
        if not lines:
            print("client failed")
            continue
        r = json.loads(lines[-1])
        if "error" in r:
            print(f"{r['pct']:>3}%  {r['mem']:>7}  FAILED: {r['error']}")
        else:
            ids = {int(i): n for i, n in r["smids"].items()}
            used.append((r["pct"], ids))
            print(f"{r['pct']:>3}%  {r['mem']:>7}  {r['sms']:>3}  {len(r['first']):>3}  {len(ids):>4}"
                  f"  {r['launches']:>8}  {r['ms']:>9}")

    if show_ids:
        print("\nSM ids used by each client (blocks per SM: min / max)")
        for n, (pct, ids) in enumerate(used):
            print(f"  #{n} {pct:>3}%  {ranges(ids)}  ({min(ids.values())} / {max(ids.values())})")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        specs = [a for a in sys.argv[1:] if a != "--smids"]
        launch(specs or ["100"], show_ids="--smids" in sys.argv)
