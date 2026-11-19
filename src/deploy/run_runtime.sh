#!/usr/bin/env bash
# run_runtime.sh - start the Inference Workload Management Runtime on this node.
#   sudo VENDOR=nvidia VALIDATION_GPU=3 ./deploy/run_runtime.sh
#   sudo VENDOR=amd ./deploy/run_runtime.sh
# With CALIB_MODEL=<onnx file> the per-container GPU metrics are verified on
# every shared GPU first (iwmr/calibrate.py), while the node is still idle.
# Every path mounted below is identical on host and in the runtime, because the
# runtime starts sibling containers through the host's Docker daemon.
set -euo pipefail

VENDOR=${VENDOR:-nvidia}
IMAGE=${IMAGE:-iwmr:latest}
PORT=${PORT:-8080}

HERE="$(cd "$(dirname "$0")" && pwd)"
docker network inspect gpushare >/dev/null 2>&1 || docker network create gpushare
mkdir -p /var/lib/gpushare

# MPS version of every shared GPU. The runtime is told which protocol the node
# speaks: 3 means limits are set through MPS namespaces, 2 through the
# container's environment. A node with daemons of both kinds counts as 2.
MPS_PROTOCOL=
if [ "$VENDOR" = nvidia ]; then
  . "$HERE/mps_version.sh"
  MPS_PROTOCOL=3
  found=0
  for d in /var/run/nvidia-mps/gpu*; do
    [ -d "$d" ] || continue
    p=$(mps_daemon_protocol "$d")
    echo "MPS daemon $d: protocol $p"
    [ "$p" != none ] || { echo "ERROR: no MPS daemon answers in $d (run deploy/setup_nvidia_node.sh)"; exit 1; }
    found=1
    [ "$p" = 3 ] || MPS_PROTOCOL=2
  done
  [ "$found" = 1 ] || { echo "ERROR: no MPS daemon found under /var/run/nvidia-mps"; exit 1; }
  if [ "$MPS_PROTOCOL" = 3 ] && ! mps_supports_namespaces; then MPS_PROTOCOL=2; fi
  [ "$MPS_PROTOCOL" = 3 ] || echo "WARN: MPS protocol 2: no namespaces, limits are set per container" \
                                  "(driver $(mps_driver_version); see deploy/mps_version.sh)"
  [ "$MPS_PROTOCOL" = 3 ] || [ "${MPS_REQUIRE_V3:-0}" != 1 ] || { echo "ERROR: MPS_REQUIRE_V3=1"; exit 1; }
fi

# Metrics check before the runtime starts. A failure does not stop the runtime:
# limits are enforced by MPS; the metrics of that GPU are only left unverified.
if [ "$VENDOR" = nvidia ] && [ -n "${CALIB_MODEL:-}" ]; then
  VALIDATION_GPU="${VALIDATION_GPU:-}" python3 "$HERE/../iwmr/calibrate.py" --all \
    --model "$CALIB_MODEL" --image "${CALIB_IMAGE:-onnx-infer:latest}" \
    || echo "WARN: GPU metrics not verified on every GPU (see /var/lib/gpushare/calibration)"
fi

COMMON=(
  -d --name iwmr --restart unless-stopped
  --network gpushare -p "$PORT:8080"
  -v /var/run/docker.sock:/var/run/docker.sock      # launch/stop inference containers
  -v /var/lib/gpushare:/var/lib/gpushare            # exit.json files, approved digests
  --pid host                                        # map faulting pids to containers
  --device /dev/kmsg --cap-add SYSLOG               # kernel log (Xid / amdgpu faults)
  -e GPUSHARE_VENDOR="$VENDOR"
  -e VALIDATION_GPU="${VALIDATION_GPU:-}"
  -e GPUSHARE_MPS_PROTOCOL="$MPS_PROTOCOL"           # 3: namespaces, 2: per-container limits
)

if [ "$VENDOR" = nvidia ]; then
  docker run "${COMMON[@]}" \
    --gpus all --ipc host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v /var/run/nvidia-mps:/var/run/nvidia-mps \
    -v "$(command -v nvidia-cuda-mps-control)":/usr/bin/nvidia-cuda-mps-control:ro \
    -e SANITIZER_DIR="${SANITIZER_DIR:-/usr/local/cuda/compute-sanitizer}" \
    "$IMAGE"
else
  docker run "${COMMON[@]}" \
    -v /opt/gpushare/lib:/opt/gpushare/lib:ro \
    "$IMAGE"
fi

sleep 3
curl -fsS "http://localhost:$PORT/health" && echo
