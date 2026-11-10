#!/usr/bin/env python3
"""guard.py - error catcher that wraps the inference server inside the container.

Runs the real server as a child process, mirrors its stderr, and on exit writes
/run/gpushare/exit.json with the exit code and a classification of the cause.
The IWMR runtime reads that file when Docker reports the container as dead.

Usage (container ENTRYPOINT):  python3 /opt/gpushare/guard.py -- <server command...>
"""
import collections, json, os, re, signal, subprocess, sys, threading, time

STATUS_DIR = os.environ.get("GPUSHARE_STATUS_DIR", "/run/gpushare")

# Ordered: the first matching class wins.
PATTERNS = [
    ("gpu_fault_victim", r"MPS client terminated|MPS server.*(terminated|unavailable)|CUDA_ERROR_MPS"),
    ("gpu_fault_culprit", r"illegal memory access|illegal address|misaligned address|"
                          r"unspecified launch failure|illegal instruction|"
                          r"Memory access fault by GPU|hipErrorIllegalAddress"),
    ("device_error", r"GPU reset|launch timed out|hipErrorLaunchTimeOut|uncorrectable ECC|"
                     r"GPU is lost|fallen off the bus"),
    ("oom", r"out of memory|OutOfMemoryError|hipErrorOutOfMemory|\[hipmemcap\] denied"),
]


def classify(exit_code, tail):
    text = "\n".join(tail)
    for name, pat in PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return name
    if exit_code == 0:
        return "ok"
    if exit_code in (-signal.SIGTERM, 128 + signal.SIGTERM):
        return "stopped"
    return "app_error"


def main():
    cmd = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if not cmd:
        sys.exit("usage: guard.py -- <command>")

    os.makedirs(STATUS_DIR, exist_ok=True)
    tail = collections.deque(maxlen=200)          # last stderr lines for classification
    started = time.time()
    child = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, bufsize=1)

    def forward(sig, _frame):                     # pass docker stop through to the server
        child.send_signal(sig)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    def pump():
        for line in child.stderr:
            sys.stderr.write(line)
            tail.append(line.rstrip())
    t = threading.Thread(target=pump, daemon=True)
    t.start()

    code = child.wait()
    t.join(timeout=2)
    report = {
        "workload_id": os.environ.get("GPUSHARE_WORKLOAD_ID"),
        "exit_code": code,
        "class": classify(code, tail),
        "started": started,
        "ended": time.time(),
        "stderr_tail": list(tail)[-20:],
    }
    tmp = os.path.join(STATUS_DIR, "exit.json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f)
    os.replace(tmp, os.path.join(STATUS_DIR, "exit.json"))
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
