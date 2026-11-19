#!/usr/bin/env bash
# mps_version.sh - which MPS does this node have, and which one is running?
#
# MPS ships with the GPU driver. Protocol 3 (namespaces, SM partitions) needs a
# control binary that offers it; assigning SM partitions to namespaces and
# routing by cgroup need driver 620 or newer. A daemon speaks the protocol it
# was started with, whatever the binary could do.
#
#   ./mps_version.sh [pipe dir ...]     report; default: every /var/run/nvidia-mps/gpu*
#   exit status: 0  protocol 3 available, and spoken by every daemon found
#                1  protocol 2 only (binary, driver or a running daemon)
#                2  no MPS control binary or no driver
#
# Can also be sourced for the functions (it then prints nothing):
#   mps_driver_version     e.g. 620.70.06
#   mps_binary_protocol    3 or 2: the newest protocol the installed binary offers
#   mps_daemon_protocol D  3, 2 or none: what the daemon of pipe directory D speaks
#   mps_supports_namespaces  true if the binary offers 3 and the driver is >= 620

MPS_MIN_DRIVER_NAMESPACES=620
MPS_CTL=${MPS_CTL:-nvidia-cuda-mps-control}

mps_driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1
}

mps_binary_protocol() {
  if "$MPS_CTL" --help 2>&1 | grep -q -- '--protocol'; then echo 3; else echo 2; fi
}

mps_daemon_protocol() {
  local dir=$1
  [ -S "$dir/control" ] || { echo none; return; }
  if CUDA_MPS_PIPE_DIRECTORY="$dir" CUDA_MPS_PROTOCOL_VERSION=3 timeout 10 "$MPS_CTL" device list >/dev/null 2>&1; then
    echo 3
  elif echo get_default_active_thread_percentage |
       CUDA_MPS_PIPE_DIRECTORY="$dir" CUDA_MPS_PROTOCOL_VERSION=2 timeout 10 "$MPS_CTL" >/dev/null 2>&1; then
    echo 2
  else
    echo none
  fi
}

mps_supports_namespaces() {
  local major
  major=$(mps_driver_version | cut -d. -f1)
  [ "$(mps_binary_protocol)" = 3 ] && [ "${major:-0}" -ge "$MPS_MIN_DRIVER_NAMESPACES" ] 2>/dev/null
}

# Sourced: stop here.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi

command -v "$MPS_CTL" >/dev/null || { echo "MPS control binary not found"; exit 2; }
driver=$(mps_driver_version)
[ -n "$driver" ] || { echo "no NVIDIA driver (nvidia-smi failed)"; exit 2; }
binary=$(mps_binary_protocol)
echo "driver          $driver"
echo "control binary  $("$MPS_CTL" -v 2>/dev/null | sed -n 's/.*Binary version: *//p' | head -n1), newest protocol $binary"
status=0
if mps_supports_namespaces; then
  echo "namespaces      available (protocol 3, driver >= $MPS_MIN_DRIVER_NAMESPACES)"
else
  echo "namespaces      NOT available: needs protocol 3 and driver >= $MPS_MIN_DRIVER_NAMESPACES"
  status=1
fi
dirs=("$@")
[ ${#dirs[@]} -gt 0 ] || dirs=(/var/run/nvidia-mps/gpu*)
for d in "${dirs[@]}"; do
  [ -d "$d" ] || continue
  p=$(mps_daemon_protocol "$d")
  echo "daemon $d  protocol $p"
  [ "$p" = 3 ] || status=1
done
exit $status
