"""validator.py - image validation inside the runtime.

Runs an image's /app/smoke_test.py under a GPU memory checker on a dedicated
validation GPU (VALIDATION_GPU, excluded from sharing and without MPS). A clean
run approves the image digest; only approved digests may share a GPU.
  NVIDIA: compute-sanitizer from the host CUDA toolkit, mounted into the image.
  AMD:    the submitted image must be an AddressSanitizer build (xnack+).
Host paths are used for volumes because the runtime drives the host's Docker.
"""
import os, re, threading, time, uuid
import docker

APPROVED = "/var/lib/gpushare/approved_digests.txt"
SANITIZER_HOST_DIR = os.environ.get("SANITIZER_DIR", "/usr/local/cuda/compute-sanitizer")
TIMEOUT_S = int(os.environ.get("VALIDATION_TIMEOUT_S", "3600"))
FAILED = re.compile(r"ERROR SUMMARY: [1-9]|AddressSanitizer|Memory access fault", re.I)


class Validator:
    def __init__(self, vendor, gpu, amd_render_minor=None):
        self.vendor, self.gpu, self.minor = vendor, gpu, amd_render_minor
        self.dock = docker.from_env()
        self.jobs = {}
        self.lock = threading.Lock()

    def approved(self, digest):
        try:
            return digest in {l.strip() for l in open(APPROVED)}
        except OSError:
            return False

    def revoke(self, digest):
        with self.lock:
            try:
                keep = [l for l in open(APPROVED) if l.strip() != digest]
                open(APPROVED, "w").writelines(keep)
            except OSError:
                pass

    def submit(self, image):
        if self.gpu is None:
            raise RuntimeError("no VALIDATION_GPU configured on this node")
        job = str(uuid.uuid4())[:8]
        self.jobs[job] = {"image": image, "status": "running", "started": time.time()}
        threading.Thread(target=self._run, args=(job, image), daemon=True).start()
        return job

    def _run(self, job, image):
        j = self.jobs[job]
        try:
            img = self.dock.images.pull(image)
            digest = (img.attrs.get("RepoDigests") or [img.id])[0]
            j["digest"] = digest
            if self.vendor == "nvidia":
                kw = dict(device_requests=[docker.types.DeviceRequest(
                              device_ids=[str(self.gpu)], capabilities=[["gpu"]])],
                          volumes={SANITIZER_HOST_DIR: {"bind": "/opt/cs", "mode": "ro"}},
                          entrypoint="/opt/cs/compute-sanitizer",
                          command=["--tool", "memcheck", "--error-exitcode", "99",
                                   "--target-processes", "all", "python3", "/app/smoke_test.py"])
            else:
                kw = dict(devices=["/dev/kfd:/dev/kfd",
                                   f"/dev/dri/renderD{self.minor}:/dev/dri/renderD{self.minor}"],
                          environment={"HSA_XNACK": "1", "ASAN_OPTIONS": "abort_on_error=1"},
                          entrypoint="python3", command=["/app/smoke_test.py"])
            c = self.dock.containers.run(image, detach=True, **kw)
            rc = c.wait(timeout=TIMEOUT_S)["StatusCode"]
            log = c.logs().decode(errors="replace")
            c.remove(force=True)
            ok = rc == 0 and not FAILED.search(log)
            j.update(status="approved" if ok else "rejected", exit_code=rc, log_tail=log[-4000:])
            if ok:
                with self.lock, open(APPROVED, "a") as f:
                    f.write(digest + "\n")
        except Exception as e:
            j.update(status="error", error=str(e))
        j["ended"] = time.time()
