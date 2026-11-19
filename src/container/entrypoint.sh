#!/bin/sh
# entrypoint.sh - connect the container to its MPS namespace, then run the command.
#
# Under MPS v3 the limits of a container (SMs, GPU memory) belong to a namespace
# of the MPS server on the host, not to the container's environment. A client
# joins a namespace through the pipe directory it connects to. Two ways to give
# the container its namespace:
#
#   1. Mount only the namespace's directory (preferred: the container cannot
#      reach any other namespace):
#        -v /var/run/nvidia-mps/gpu0/<server>/<namespace>:/run/nvidia-mps
#        -e CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps
#   2. Mount the daemon's whole directory and name the namespace:
#        -v /var/run/nvidia-mps/gpu0:/run/nvidia-mps
#        -e CUDA_MPS_PIPE_DIRECTORY=/run/nvidia-mps
#        -e VITRO_MPS_NAMESPACE=<server>/<namespace>
#
# VITRO_MPS_NAMESPACE may also be set with (1); it is then only a label that
# /health reports. VITRO_MEM_CAP (e.g. 200M, 1G) tells the observer library the
# memory limit of the namespace: the driver does not report it to a client.
#
# The container stops here, with exit status 78, if it was given a pipe
# directory or a namespace that does not exist. Without this check the workload
# would run in the server's default namespace, i.e. without its limits.
set -eu

PIPE="${CUDA_MPS_PIPE_DIRECTORY:-}"
NS="${VITRO_MPS_NAMESPACE:-}"
die() { echo "[vitro] $*" >&2; exit 78; }

if [ -z "$PIPE" ]; then
  [ -z "$NS" ] || die "VITRO_MPS_NAMESPACE=$NS is set but CUDA_MPS_PIPE_DIRECTORY is not"
  echo "[vitro] no MPS pipe directory: running without MPS limits" >&2
else
  if [ -n "$NS" ]; then
    if [ -S "$PIPE/$NS/control" ]; then
      PIPE="$PIPE/$NS"
    elif [ -e "$PIPE/nvidia-cuda-mps-control.pid" ] || [ -d "$PIPE/default" ]; then
      # daemon or server directory, and the namespace is not in it
      die "MPS namespace '$NS' not found under $PIPE"
    fi                                   # else: $PIPE is the namespace directory itself
  fi
  [ -S "$PIPE/control" ] || die "no MPS control socket in $PIPE (is the pipe directory mounted?)"
  export CUDA_MPS_PIPE_DIRECTORY="$PIPE"
  echo "[vitro] MPS pipe directory $PIPE, namespace ${NS:-<not named>}, memory cap ${VITRO_MEM_CAP:-<not given>}" >&2
fi

exec "$@"
