#!/usr/bin/env python3
"""Dummy GPU workload for MPS testing (no torch needed).

Usage:  python mps_burn.py 25 50 25         # 3 concurrent clients at 25%, 50%, 25% SMs
        python mps_burn.py 25:300 50:600    # same, each client also holds 300 / 600 MiB

Every driver call is checked. A client that gets an error (e.g. out of memory at
its CUDA_MPS_PINNED_DEVICE_MEM_LIMIT) prints it to stderr, exits 1, and is shown
in the table with the error instead of timings.
"""
import ctypes, json, os, subprocess, sys, time

# GPU kernel: each thread loops ITERS times doing x = x*1 + 1 (dummy load)
PTX = rb"""
.version 7.0
.target sm_52
.address_size 64
.visible .entry burn(.param .u64 out, .param .u32 iters)
{
  .reg .f32 %f<3>; .reg .u32 %r<4>; .reg .u64 %rd<2>; .reg .pred %p;
  ld.param.u64 %rd1, [out]; ld.param.u32 %r1, [iters];
  mov.u32 %r2, %tid.x; cvt.rn.f32.u32 %f1, %r2;
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
    sms, ptr, held = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_uint64()
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

    out, iters = ctypes.c_uint64(ptr.value), ctypes.c_uint32(ITERS)
    args = (ctypes.c_void_p * 2)(ctypes.addressof(out), ctypes.addressof(iters))

    launches, t0 = 0, time.time()
    while time.time() - t0 < DURATION:
        call("cuLaunchKernel", fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, args, None)
        call("cuCtxSynchronize")
        launches += 1
    ms = 1000 * (time.time() - t0) / max(launches, 1)

    print(json.dumps({"pct": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100"),
                      "mem": mem_mib, "sms": sms.value, "launches": launches, "ms": round(ms, 2)}))


def launch(specs):
    """Start one client per 'pct' or 'pct:MiB' spec, all at once, and print their results."""
    procs = []
    for spec in specs:
        pct, _, mem = spec.partition(":")
        env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=pct, BURN_MEM_MIB=mem or "0")
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker"],
                                      env=env, stdout=subprocess.PIPE, text=True))

    print(" pct  mem MiB  SMs  launches  ms/launch")
    for p in procs:
        lines = p.communicate()[0].strip().splitlines()
        if not lines:
            print("client failed")
            continue
        r = json.loads(lines[-1])
        if "error" in r:
            print(f"{r['pct']:>3}%  {r['mem']:>7}  FAILED: {r['error']}")
        else:
            print(f"{r['pct']:>3}%  {r['mem']:>7}  {r['sms']:>3}  {r['launches']:>8}  {r['ms']:>9}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        launch(sys.argv[1:] or ["100"])
