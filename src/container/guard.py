#!/usr/bin/env python3
"""guard.py - start the inference server, watch it for GPU faults, record why it exited.

    python3 guard.py -- <server command ...>

supervisord starts guard.py and guard.py starts the server as its child, so it
sees everything the server prints and its real exit status or signal.

  - Server stdout/stderr is passed through and scanned for CUDA / MPS errors
    (by number, enum name or driver message; see the tables below).
  - Signals sent to guard.py (SIGTERM from `docker stop`, ...) are forwarded.
  - A fatal GPU error leaves the CUDA context unusable while the process may
    keep running, so the server is stopped FATAL_GRACE_S seconds after one.
  - On exit the cause is written to VITRO_EXIT_FILE, the handler for its class
    is called (placeholders for now, see HANDLERS), and supervisord is stopped
    so the container exits and the IWMR decides on recovery.

Culprit vs victim is decided by the first fatal error seen; the exact error
text per side must be confirmed with acceptance test T9 (tests/fault_inject.py).

Environment
  VITRO_EXIT_FILE    exit record, JSON (default /run/vitro/exit.json)
  FATAL_GRACE_S      wait after a fatal GPU error before stopping the server (3)
  STOP_TIMEOUT_S     wait after SIGTERM before SIGKILL (15; < supervisord stopwaitsecs)
  VITRO_GUARD_STOP_SUPERVISOR  1 (default): stop supervisord when the server is gone

guard.py exits with the server's status (128 + signal number after a signal).
"""
import collections, ctypes, json, os, re, signal, subprocess, sys, threading, time

EXIT_FILE = os.environ.get("VITRO_EXIT_FILE", "/run/vitro/exit.json")
FATAL_GRACE_S = float(os.environ.get("FATAL_GRACE_S", "3"))
STOP_TIMEOUT_S = float(os.environ.get("STOP_TIMEOUT_S", "15"))
STOP_SUPERVISOR = os.environ.get("VITRO_GUARD_STOP_SUPERVISOR", "1") == "1"
TAIL_LINES = 40
MAX_EVIDENCE = 20

# --------------------------------------------------------------------------- #
# Error tables. Driver (CUresult) and runtime (cudaError_t) share the numbers.
# kind: what the error means for this process
#   oom      device memory exhausted; not fatal, the server answers 503
#   culprit  this process faulted on the GPU
#   victim   MPS terminated this client because of a fault elsewhere
#   device   the GPU itself is in trouble
#   mps      could not reach / join the MPS server
# Every kind except oom is fatal: the CUDA context cannot be used any more.
# --------------------------------------------------------------------------- #

# Errors that stay with this workload: the other MPS clients of the GPU live on.
CUDA_ERRORS = [
    # code, driver enum,                            runtime enum,                        message,                                                      kind
    (2,   "CUDA_ERROR_OUT_OF_MEMORY",               "cudaErrorMemoryAllocation",         "out of memory",                                              "oom"),
    (3,   "CUDA_ERROR_NOT_INITIALIZED",             "cudaErrorInitializationError",      "initialization error",                                       "mps"),
    (4,   "CUDA_ERROR_DEINITIALIZED",               "cudaErrorCudartUnloading",          "driver shutting down",                                       "victim"),
    (46,  "CUDA_ERROR_DEVICES_UNAVAILABLE",         "cudaErrorDevicesUnavailable",       "CUDA-capable device(s) is/are busy or unavailable",          "mps"),
    (100, "CUDA_ERROR_NO_DEVICE",                   "cudaErrorNoDevice",                 "no CUDA-capable device is detected",                         "mps"),
    (709, "CUDA_ERROR_CONTEXT_IS_DESTROYED",        "cudaErrorContextIsDestroyed",       "context is destroyed",                                       "victim"),
    (805, "CUDA_ERROR_MPS_CONNECTION_FAILED",       "cudaErrorMpsConnectionFailed",      "MPS client failed to connect to the MPS control daemon or the MPS server", "mps"),
    (806, "CUDA_ERROR_MPS_RPC_FAILURE",             "cudaErrorMpsRpcFailure",            "the remote procedural call between the MPS server and the MPS client failed", "victim"),
    (807, "CUDA_ERROR_MPS_SERVER_NOT_READY",        "cudaErrorMpsServerNotReady",        "MPS server is not ready to accept new MPS client requests",  "mps"),
    (808, "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED",     "cudaErrorMpsMaxClientsReached",     "the hardware resources required to create MPS client have been exhausted", "mps"),
    (809, "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED", "cudaErrorMpsMaxConnectionsReached", "the hardware resources required to support device connections have been exhausted", "mps"),
    (810, "CUDA_ERROR_MPS_CLIENT_TERMINATED",       "cudaErrorMpsClientTerminated",      "the MPS client has been terminated by the server",           "victim"),
    (999, "CUDA_ERROR_UNKNOWN",                     "cudaErrorUnknown",                  "unknown error",                                              "device"),
]

# Errors that make the MPS server drop *every* workload on the GPU. Under MPS
# (Volta and newer) a fatal exception is contained to the clients sharing the
# GPU with the faulting one; those see 810 / 806 above. Confirm with T9.
DESTRUCTIVE_ERRORS = [
    (214, "CUDA_ERROR_ECC_UNCORRECTABLE",           "cudaErrorECCUncorrectable",         "uncorrectable ECC error encountered",                        "device"),
    (700, "CUDA_ERROR_ILLEGAL_ADDRESS",             "cudaErrorIllegalAddress",           "an illegal memory access was encountered",                   "culprit"),
    (702, "CUDA_ERROR_LAUNCH_TIMEOUT",              "cudaErrorLaunchTimeout",            "the launch timed out and was terminated",                    "device"),
    (714, "CUDA_ERROR_HARDWARE_STACK_ERROR",        "cudaErrorHardwareStackError",       "hardware stack error",                                       "culprit"),
    (715, "CUDA_ERROR_ILLEGAL_INSTRUCTION",         "cudaErrorIllegalInstruction",       "an illegal instruction was encountered",                     "culprit"),
    (716, "CUDA_ERROR_MISALIGNED_ADDRESS",          "cudaErrorMisalignedAddress",        "misaligned address",                                         "culprit"),
    (717, "CUDA_ERROR_INVALID_ADDRESS_SPACE",       "cudaErrorInvalidAddressSpace",      "operation not supported on global/shared address space",     "culprit"),
    (718, "CUDA_ERROR_INVALID_PC",                  "cudaErrorInvalidPc",                "invalid program counter",                                    "culprit"),
    (719, "CUDA_ERROR_LAUNCH_FAILED",               "cudaErrorLaunchFailure",            "unspecified launch failure",                                 "culprit"),
]
DESTRUCTIVE_CODES = {e[0] for e in DESTRUCTIVE_ERRORS}
ALL_ERRORS = CUDA_ERRORS + DESTRUCTIVE_ERRORS
BY_CODE = {e[0]: e for e in ALL_ERRORS}
BY_NAME = {n: e for e in ALL_ERRORS for n in e[1:3]}

# "CUDA failure 700", "CUDA error code = 806", "CUresult 810", "cudaError_t 2"
CODE_RE = re.compile(r"\b(?:CUDA(?:\s+(?:driver|runtime|API))*\s*(?:failure|error)(?:\s*code)?|"
                     r"CUresult|cudaError(?:_t)?)\s*[=:#(]?\s*(\d{1,3})\b", re.I)
NAME_RE = re.compile(r"\b(" + "|".join(BY_NAME) + r")\b")
# Driver messages count only on lines that look like GPU errors; "out of memory"
# and "unknown error" are too common and are left to the numbered forms and OOM_RE.
GPU_LINE_RE = re.compile(r"cuda|cublas|cudnn|\bgpu\b|\bmps\b|onnxruntime|\bcu[A-Z]\w+", re.I)
MSG_RES = [(re.compile(re.escape(e[3]), re.I), e) for e in ALL_ERRORS if e[0] not in (2, 999)]
# Allocation failures reported by libraries without a CUDA number.
OOM_RE = re.compile(r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|BFCArena.*Failed to allocate|"
                    r"Failed to allocate memory for requested buffer|(?:cuda|gpu)\w*[^\n]{0,80}out of memory", re.I)

KIND_CLASS = {"oom": "oom", "culprit": "gpu_fault_culprit", "victim": "gpu_fault_victim",
              "device": "device_error", "mps": "mps_unavailable"}
CRASH_SIGNALS = {signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGILL, signal.SIGFPE}
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def log(msg):
    print(f"[guard] {msg}", file=sys.stderr, flush=True)


def hit(e):
    return {"code": e[0], "name": e[1], "kind": e[4], "message": e[3], "destructive": e[0] in DESTRUCTIVE_CODES}


def match_line(line):
    """GPU errors in one line of server output, as a list of hit dicts."""
    found = {}
    for m in NAME_RE.finditer(line):
        found[BY_NAME[m.group(1)][0]] = BY_NAME[m.group(1)]
    for m in CODE_RE.finditer(line):
        if int(m.group(1)) in BY_CODE:
            found[int(m.group(1))] = BY_CODE[int(m.group(1))]
    if GPU_LINE_RE.search(line):
        found.update((e[0], e) for rx, e in MSG_RES if rx.search(line))
    if not found and OOM_RE.search(line):
        found[2] = BY_CODE[2]
    return [hit(e) for e in found.values()]


# --------------------------------------------------------------------------- #
# Handlers - one per exit class, called once after the exit record is written.
# Placeholders: they log only. The recovery itself belongs to the IWMR (README,
# "Faults and recovery"); each docstring says what is planned there.
# --------------------------------------------------------------------------- #
def handle_destructive(h):
    """Live, the moment a DESTRUCTIVE_ERRORS entry is seen (server still running).
    TODO: tell the IWMR at once, so it opens an incident for this GPU before the
    other workloads' exits arrive and attributes them as victims of this one."""
    log(f"TODO handle_destructive: {h['name']}")


def handle_ok(r):
    """Clean exit. TODO: nothing to recover; the IWMR releases the slice."""


def handle_stopped(r):
    """Stopped by a signal (docker stop / DELETE /workloads). TODO: nothing to recover."""


def handle_oom(r):
    """GPU out of memory, then exit. TODO (IWMR): re-place with a 25 % larger
    memory cap, bounded by GPU memory."""
    log("TODO handle_oom")


def handle_host_oom(r):
    """Killed by the kernel OOM killer (host RAM). TODO (IWMR): raise the
    container's memory limit; the GPU cap is not the problem."""
    log("TODO handle_host_oom")


def handle_gpu_fault_culprit(r):
    """This workload faulted on the GPU (DESTRUCTIVE_ERRORS). TODO (IWMR): drop it
    (no restart), revoke its image approval, answer its pending requests with 503."""
    log("TODO handle_gpu_fault_culprit")


def handle_gpu_fault_victim(r):
    """MPS terminated this workload because of a fault elsewhere. TODO (IWMR):
    re-place immediately and replay its pending requests from the journal."""
    log("TODO handle_gpu_fault_victim")


def handle_device_error(r):
    """The GPU is in trouble (ECC, launch timeout, unknown error). TODO (IWMR):
    drain the GPU and re-place all its workloads elsewhere."""
    log("TODO handle_device_error")


def handle_mps_unavailable(r):
    """Could not join the MPS server (not running, recovering, client limit).
    TODO (IWMR): retry later on the same GPU, or place on another GPU."""
    log("TODO handle_mps_unavailable")


def handle_app_error(r):
    """Non-GPU failure (exception, SIGSEGV, ...). TODO (IWMR): treat as a victim if
    the GPU had a culprit in the last 15 s, else restart with back-off (max 5)."""
    log("TODO handle_app_error")


def handle_unknown(r):
    """SIGKILL from outside, cause unknown. TODO (IWMR): same rule as app_error."""
    log("TODO handle_unknown")


HANDLERS = {
    "ok": handle_ok, "stopped": handle_stopped, "oom": handle_oom, "host_oom": handle_host_oom,
    "gpu_fault_culprit": handle_gpu_fault_culprit, "gpu_fault_victim": handle_gpu_fault_victim,
    "device_error": handle_device_error, "mps_unavailable": handle_mps_unavailable,
    "app_error": handle_app_error, "unknown": handle_unknown,
}


# --------------------------------------------------------------------------- #
def oom_kills():
    """Kernel OOM kills in this container's cgroup (v2), or None."""
    try:
        with open("/sys/fs/cgroup/memory.events") as f:
            return int(dict(row.split() for row in f).get("oom_kill", 0))
    except (OSError, ValueError):
        return None


class Guard:
    def __init__(self, cmd):
        self.cmd = cmd
        self.mu = threading.Lock()
        self.tail = collections.deque(maxlen=TAIL_LINES)
        self.evidence = []
        self.counts = collections.Counter()   # error name -> times seen
        self.first_fatal = None               # decides culprit / victim
        self.destructive = None
        self.stop_signal = None               # signal received by guard.py
        self.stopped_by_guard = False
        self.oom_kills_start = oom_kills()
        self.started = time.time()
        self.child = None

    def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        self.child = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      env=env, preexec_fn=die_with_parent)
        log(f"started pid {self.child.pid}: {' '.join(self.cmd)}")
        self.readers = [threading.Thread(target=self.read, args=(pipe, out), daemon=True)
                        for pipe, out in ((self.child.stdout, sys.stdout.buffer),
                                          (self.child.stderr, sys.stderr.buffer))]
        for t in self.readers:
            t.start()

    def read(self, pipe, out):
        for raw in iter(pipe.readline, b""):
            out.write(raw)
            out.flush()
            line = raw.decode("utf-8", "replace").rstrip()
            hits = match_line(line)
            with self.mu:
                self.tail.append(line)
                if hits:
                    self.record(line, hits)

    def record(self, line, hits):                       # under self.mu
        for h in hits:
            self.counts[h["name"]] += 1
            if h["kind"] != "oom" and not self.first_fatal:
                self.first_fatal = dict(h, line=line, t=time.time())
                log(f"fatal GPU error {h['name']} ({h['kind']}): {line[:200]}")
            if h["destructive"] and not self.destructive:
                self.destructive = dict(h, line=line)
                handle_destructive(h)
        if len(self.evidence) < MAX_EVIDENCE:
            self.evidence.append({"t": round(time.time(), 3), "errors": [h["name"] for h in hits],
                                  "line": line[:500]})

    def on_signal(self, signum, _frame):
        self.stop_signal = self.stop_signal or signum
        log(f"received {signal.Signals(signum).name}, forwarding to the server")
        self.send(signum)

    def send(self, signum):
        if self.child and self.child.poll() is None:
            self.child.send_signal(signum)

    def run(self):
        for s in FORWARDED:
            signal.signal(s, self.on_signal)
        self.start()
        kill_at = None
        while (rc := self.child.poll()) is None:
            time.sleep(0.2)
            now = time.time()
            if kill_at is None and self.stop_signal:
                kill_at = now + STOP_TIMEOUT_S
            elif kill_at is None and self.first_fatal and now - self.first_fatal["t"] >= FATAL_GRACE_S:
                log(f"server still running {FATAL_GRACE_S:g}s after a fatal GPU error: stopping it")
                self.stopped_by_guard = True
                self.send(signal.SIGTERM)
                kill_at = now + min(STOP_TIMEOUT_S, 5)
            elif kill_at and now >= kill_at:
                log("server did not stop in time: SIGKILL")
                self.send(signal.SIGKILL)
                kill_at = float("inf")
        for t in self.readers:                          # last lines of output
            t.join(timeout=2)
        r = self.classify(rc)
        write_json(EXIT_FILE, r)
        log(f"server exited ({r['how']}): class={r['class']} - {r['reason']}")
        HANDLERS[r["class"]](r)
        if STOP_SUPERVISOR:
            stop_supervisor()
        return 128 + r["signal"] if r["signal"] else rc

    def classify(self, rc):
        sig = signal.Signals(-rc) if rc < 0 else None
        kills = oom_kills()
        host_oom = kills is not None and self.oom_kills_start is not None and kills > self.oom_kills_start
        ff = self.first_fatal
        victim_seen = any(self.counts[e[1]] for e in ALL_ERRORS if e[4] == "victim")
        device = next((e for e in ALL_ERRORS if e[4] == "device" and self.counts[e[1]]), None)

        if device:
            cls, why = "device_error", f"{device[1]}: {device[3]}"
        elif ff:
            # 719 is also what some drivers report to a victim of an MPS teardown
            cls = "gpu_fault_victim" if ff["code"] == 719 and victim_seen else KIND_CLASS[ff["kind"]]
            why = f"{ff['name']}: {ff['message']}"
        elif self.stop_signal:
            cls, why = "stopped", f"stopped by {signal.Signals(self.stop_signal).name}"
        elif sig == signal.SIGKILL and host_oom:
            cls, why = "host_oom", "killed by the kernel OOM killer"
        elif rc == 0:
            cls, why = "ok", "clean exit"
        elif self.counts["CUDA_ERROR_OUT_OF_MEMORY"]:
            cls, why = "oom", f"GPU out of memory ({self.counts['CUDA_ERROR_OUT_OF_MEMORY']} times), then exit"
        elif sig == signal.SIGKILL:
            cls, why = "unknown", "SIGKILL from outside (not the OOM killer)"
        elif sig in CRASH_SIGNALS:
            cls, why = "app_error", f"crashed with {sig.name}"
        else:
            cls, why = "app_error", f"exit code {rc}" + (f" ({sig.name})" if sig else "")

        ended = time.time()
        return {
            "class": cls, "reason": why,
            "how": f"signal {sig.name}" if sig else f"exit code {rc}",
            "exit_code": rc if rc >= 0 else None, "signal": int(sig) if sig else None,
            "signal_name": sig.name if sig else None,
            "stop_signal": signal.Signals(self.stop_signal).name if self.stop_signal else None,
            "stopped_by_guard": self.stopped_by_guard,
            "destructive": self.destructive is not None,    # likely cause of an MPS teardown of the GPU
            "destructive_error": self.destructive,
            "first_fatal_gpu_error": ff,
            "gpu_error_counts": dict(self.counts),
            "started_at": round(self.started, 3), "ended_at": round(ended, 3),
            "evidence": self.evidence, "tail": list(self.tail),
        }


def write_json(path, data):
    """Atomic write; falls back to stderr when the path is not writable."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(data, f, indent=2)
        os.replace(path + ".tmp", path)
    except OSError as e:
        log(f"cannot write {path}: {e}")
        print(json.dumps(data), file=sys.stderr, flush=True)


def die_with_parent():
    """In the child: SIGKILL when guard.py dies, so the server never outlives it."""
    ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))     # PR_SET_PDEATHSIG


def stop_supervisor():
    """Stop supervisord (our parent) so that the container exits with the server."""
    ppid = os.getppid()
    try:
        with open(f"/proc/{ppid}/cmdline", "rb") as f:
            if b"supervisord" in f.read():
                os.kill(ppid, signal.SIGTERM)
    except OSError:
        pass


if __name__ == "__main__":
    cmd = sys.argv[1:]
    cmd = cmd[1:] if cmd[:1] == ["--"] else cmd
    if not cmd:
        sys.exit("usage: guard.py -- <server command ...>")
    sys.exit(Guard(cmd).run())
