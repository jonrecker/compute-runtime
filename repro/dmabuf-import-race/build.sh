#!/usr/bin/env bash
# Build the reproducer. Needs: g++, libze-dev (or any Level Zero headers + loader),
# ocloc (intel-ocloc), and the kernel UAPI header drm/xe_drm.h (libdrm-dev or linux-libc-dev).
set -euo pipefail
cd "$(dirname "$0")"
ocloc compile -file kernel.cl -spv_only -out_dir . >/dev/null
mv kernel_.spv kernel.spv
g++ -O1 -std=c++17 -pthread repro.cpp -o repro -lze_loader
echo "built ./repro and kernel.spv"
