#!/usr/bin/env bash
# run_runtime.sh - start the Inference Workload Management Runtime on this node.
#   sudo VENDOR=nvidia VALIDATION_GPU=3 ./deploy/run_runtime.sh
#   sudo VENDOR=amd ./deploy/run_runtime.sh
# Every path mounted below is identical on host and in the runtime, because the
# runtime starts sibling containers through the host's Docker daemon.
set -euo pipefail

VENDOR=${VENDOR:-nvidia}
IMAGE=${IMAGE:-iwmr:latest}
PORT=${PORT:-8080}

docker network inspect gpushare >/dev/null 2>&1 || docker network create gpushare
mkdir -p /var/lib/gpushare

COMMON=(
  -d --name iwmr --restart unless-stopped
  --network gpushare -p "$PORT:8080"
  -v /var/run/docker.sock:/var/run/docker.sock      # launch/stop inference containers
  -v /var/lib/gpushare:/var/lib/gpushare            # exit.json files, approved digests
  --pid host                                        # map faulting pids to containers
  --device /dev/kmsg --cap-add SYSLOG               # kernel log (Xid / amdgpu faults)
  -e GPUSHARE_VENDOR="$VENDOR"
  -e VALIDATION_GPU="${VALIDATION_GPU:-}"
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
