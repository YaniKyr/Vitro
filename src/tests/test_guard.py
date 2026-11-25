#!/usr/bin/env python3
"""test_guard.py - exit classification of container/guard.py, without a GPU.

guard.py is run around small fake servers that print the error text a real
server would and then exit, crash, hang or wait for a signal.

  python3 tests/test_guard.py          (unittest; no extra packages)
"""
import json, os, signal, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "..", "container", "guard.py")
sys.path.insert(0, os.path.join(HERE, "..", "container"))
import guard  # noqa: E402


def run_guard(child_src, timeout=30, after_start=None, env=None):
    """Run guard.py around `python3 -c child_src`; return (guard exit code, exit record)."""
    with tempfile.TemporaryDirectory() as d:
        e = dict(os.environ, VITRO_EXIT_FILE=f"{d}/exit.json",
                 VITRO_GUARD_STOP_SUPERVISOR="0", FATAL_GRACE_S="1", STOP_TIMEOUT_S="3", **(env or {}))
        p = subprocess.Popen([sys.executable, GUARD, "--", sys.executable, "-c", child_src], env=e,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if after_start:
            time.sleep(1.0)
            after_start(p)
        rc = p.wait(timeout=timeout)
        with open(f"{d}/exit.json") as f:
            return rc, json.load(f)


PRINT = "import sys,time; print({!r}, file=sys.stderr, flush=True); "


class Classify(unittest.TestCase):
    def test_clean_exit(self):
        rc, r = run_guard("print('hello')")
        self.assertEqual((rc, r["class"]), (0, "ok"))

    def test_illegal_access_hangs_then_guard_stops_it(self):
        # ONNX Runtime's wording; the server keeps running after the sticky fault
        rc, r = run_guard(PRINT.format("CUDA failure 700: an illegal memory access was encountered ; "
                                       "GPU=0 ; expr=cudaStreamSynchronize(stream);") + "time.sleep(60)")
        self.assertEqual(r["class"], "gpu_fault_culprit")
        self.assertTrue(r["stopped_by_guard"])
        self.assertEqual(r["first_fatal_gpu_error"]["code"], 700)
        self.assertEqual(rc, 128 + signal.SIGTERM)

    def test_message_only(self):
        # tests/gpu_lowlevel.py wording: driver message, no number
        rc, r = run_guard(PRINT.format("RuntimeError: cuCtxSynchronize: misaligned address") + "sys.exit(1)")
        self.assertEqual(r["class"], "gpu_fault_culprit")
        self.assertEqual(r["first_fatal_gpu_error"]["code"], 716)

    def test_mps_victim(self):
        rc, r = run_guard(PRINT.format("CUDA error: the MPS client has been terminated by the server")
                          + "sys.exit(1)")
        self.assertEqual(r["class"], "gpu_fault_victim")

    def test_victim_launch_failure_after_termination(self):
        src = (PRINT.format("CUDA failure 719: unspecified launch failure")
               + PRINT.format("cudaErrorMpsClientTerminated").split("; ", 1)[1] + "sys.exit(1)")
        rc, r = run_guard(src)
        self.assertEqual(r["class"], "gpu_fault_victim")

    def test_mps_connect(self):
        rc, r = run_guard(PRINT.format("CUDA failure 805: MPS client failed to connect to the MPS control "
                                       "daemon or the MPS server") + "sys.exit(1)")
        self.assertEqual(r["class"], "mps_unavailable")

    def test_ecc(self):
        rc, r = run_guard(PRINT.format("CUDA_ERROR_ECC_UNCORRECTABLE") + "sys.exit(1)")
        self.assertEqual(r["class"], "device_error")

    def test_oom_is_not_fatal(self):
        src = (PRINT.format("inference failed: CUDA failure 2: out of memory")
               + PRINT.format("BFCArena::AllocateRawInternal Failed to allocate memory").split("; ", 1)[1]
               + "time.sleep(2); sys.exit(0)")
        rc, r = run_guard(src)
        self.assertEqual(r["class"], "ok")               # kept running, exited cleanly
        self.assertFalse(r["stopped_by_guard"])
        self.assertEqual(r["gpu_error_counts"]["CUDA_ERROR_OUT_OF_MEMORY"], 2)

    def test_oom_then_crash(self):
        rc, r = run_guard(PRINT.format("CUBLAS failure 3: CUBLAS_STATUS_ALLOC_FAILED") + "sys.exit(1)")
        self.assertEqual(r["class"], "oom")

    def test_segfault(self):
        rc, r = run_guard("import os,signal; os.kill(os.getpid(), signal.SIGSEGV)")
        self.assertEqual((r["class"], r["signal_name"]), ("app_error", "SIGSEGV"))
        self.assertEqual(rc, 128 + signal.SIGSEGV)

    def test_abort(self):
        rc, r = run_guard("import os; os.abort()")
        self.assertEqual((r["class"], r["signal_name"]), ("app_error", "SIGABRT"))

    def test_outside_sigkill(self):
        rc, r = run_guard("import os,signal; os.kill(os.getpid(), signal.SIGKILL)")
        self.assertEqual((r["class"], r["signal_name"]), ("unknown", "SIGKILL"))

    def test_sigterm_forwarded(self):
        rc, r = run_guard("import time; time.sleep(60)", after_start=lambda p: p.send_signal(signal.SIGTERM))
        self.assertEqual((r["class"], r["stop_signal"]), ("stopped", "SIGTERM"))

    def test_stop_escalates_to_sigkill(self):
        src = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        rc, r = run_guard(src, after_start=lambda p: p.send_signal(signal.SIGTERM))
        self.assertEqual((r["class"], r["signal_name"]), ("stopped", "SIGKILL"))


class Destructive(unittest.TestCase):
    def test_every_class_has_a_handler(self):
        classes = set(guard.KIND_CLASS.values()) | {"ok", "stopped", "host_oom", "app_error", "unknown"}
        self.assertEqual(classes, set(guard.HANDLERS))

    def test_tables_disjoint(self):
        self.assertFalse({e[0] for e in guard.CUDA_ERRORS} & guard.DESTRUCTIVE_CODES)

    def test_culprit_is_destructive(self):
        rc, r = run_guard(PRINT.format("CUDA failure 700: an illegal memory access was encountered") + "sys.exit(1)")
        self.assertTrue(r["destructive"])
        self.assertEqual(r["destructive_error"]["code"], 700)

    def test_victim_and_oom_are_not(self):
        for text in ("the MPS client has been terminated by the server", "CUDA failure 2: out of memory"):
            rc, r = run_guard(PRINT.format(text) + "sys.exit(1)")
            self.assertFalse(r["destructive"], text)


class Match(unittest.TestCase):
    def codes(self, line):
        return [h["code"] for h in guard.match_line(line)]

    def test_no_false_positives(self):
        for line in ["INFO: Uvicorn running on http://0.0.0.0:8000",
                     "CUDA version 12.4, 2 devices",
                     "providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']",
                     "unknown error in request parser",
                     "GET /health 200 OK"]:
            self.assertEqual(guard.match_line(line), [], line)

    def test_forms(self):
        self.assertEqual(self.codes("CUDA failure 700: an illegal memory access was encountered"), [700])
        self.assertEqual(self.codes("cudaErrorIllegalAddress"), [700])
        self.assertEqual(self.codes("CUresult 810"), [810])
        self.assertEqual(self.codes("CUDA error: an illegal instruction was encountered"), [715])
        self.assertEqual(self.codes("CUDA driver error code = 806"), [806])


if __name__ == "__main__":
    unittest.main(verbosity=2)
