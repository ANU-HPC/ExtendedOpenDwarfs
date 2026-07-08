#!/usr/bin/env bash
# Extended OpenDwarfs / SCALE paper -- system info collector for Table 2 (Section 4.1/4.2).
# Run this on each benchmark machine (trill, alpha, epsilon, beta, troi, andoria, zenith, ...)
# and paste the full output back. It is safe to run on NVIDIA-only, AMD-only, or mixed boxes.

echo "=================================================="
echo "HOST: $(hostname)"
echo "DATE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=================================================="

echo
echo "--- CPU ---"
lscpu | grep -E "Model name|Socket|Core|Thread" 2>/dev/null || cat /proc/cpuinfo | grep "model name" | uniq

echo
echo "--- OS / Kernel ---"
cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="
uname -r

echo
echo "--- Host compiler ---"
gcc --version 2>/dev/null | head -1
clang --version 2>/dev/null | head -1

echo
echo "--- NVIDIA (if present) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo "CUDA (driver-reported, top-right of nvidia-smi):"
    nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+"
else
    echo "nvidia-smi not found"
fi
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc (active toolchain -- run this once with plain PATH, and again after 'source \${SCALE_ROOT}/scaleenv' to capture SCALE's version banner):"
    nvcc --version
else
    echo "nvcc not found on PATH"
fi

echo
echo "--- AMD (if present) ---"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | grep -E "Marketing Name|Name:" | head -20
fi
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>/dev/null
    rocm-smi --showdriverversion 2>/dev/null
    rocm-smi --showmeminfo vram 2>/dev/null
fi
if [ -f /opt/rocm/.info/version ]; then
    echo "ROCm version file:"
    cat /opt/rocm/.info/version
fi
if command -v hipcc >/dev/null 2>&1; then
    echo "hipcc:"
    hipcc --version
else
    echo "hipcc not found on PATH"
fi
if command -v apt >/dev/null 2>&1; then
    dpkg -l 2>/dev/null | grep -i rocm | head -5
fi

echo
echo "--- SCALE (if scaleenv already sourced) ---"
echo "SCALE_ROOT=${SCALE_ROOT:-<not set>}"
if command -v scale-nvcc >/dev/null 2>&1; then
    scale-nvcc --version
fi

echo
echo "=================================================="
echo "DONE -- paste everything above back for $(hostname)"
echo "=================================================="
