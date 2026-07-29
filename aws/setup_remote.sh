#!/usr/bin/env bash
#
# setup_remote.sh — Run ON the GPU instance after SSH-ing in.
#
# This script:
#   1. Verifies GPU access (nvcc, nvidia-smi, ncu)
#   2. Installs Python deps for reference generation
#   3. Builds the CUDA kernels
#   4. Runs correctness tests
#   5. Runs benchmarks
#   6. Runs Nsight Compute profiling
#   7. Prints results ready to paste into README
#
# Usage (on the remote instance):
#   chmod +x ~/setup_remote.sh && ~/setup_remote.sh
#
set -euo pipefail

PROJECT_DIR="$HOME/fused-attention"

echo "=== Fused Attention — Remote GPU Setup ==="
echo ""

# ── Step 0: Verify GPU access ────────────────────────────────────────────────
echo "──── Step 0: GPU Verification ────"

echo -n "nvidia-smi: "
if nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null; then
    :
else
    echo "FAILED — no GPU detected. Wrong instance type?"
    exit 1
fi

echo -n "nvcc: "
if nvcc --version 2>/dev/null | grep -o "release.*" | head -1; then
    :
else
    # Deep Learning AMI sometimes needs PATH setup
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    if nvcc --version 2>/dev/null | grep -o "release.*" | head -1; then
        echo "  (added /usr/local/cuda/bin to PATH)"
    else
        echo "FAILED — nvcc not found. Install CUDA toolkit."
        exit 1
    fi
fi

echo -n "ncu: "
if ncu --version 2>/dev/null | head -1; then
    NCU_OK=true
else
    # ncu is sometimes in a different path on DLAMIs
    for p in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
        if [ -x "$p" ]; then
            export PATH="$(dirname "$p"):$PATH"
            echo "  found at $p"
            NCU_OK=true
            break
        fi
    done
    if [ "${NCU_OK:-}" != "true" ]; then
        echo "WARNING — ncu not found. Profiling step will be skipped."
        echo "  You can still get correctness + benchmark numbers."
        NCU_OK=false
    fi
fi

# Test ncu can actually profile (the permission check that kills Colab)
if [ "$NCU_OK" = "true" ]; then
    echo -n "ncu permissions: "
    # Compile a trivial kernel to test profiling
    cat > /tmp/ncu_test.cu << 'NVCC_EOF'
__global__ void noop() {}
int main() { noop<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
NVCC_EOF
    nvcc -o /tmp/ncu_test /tmp/ncu_test.cu 2>/dev/null
    if ncu --target-processes all /tmp/ncu_test 2>&1 | grep -q "ERR_NVGPUCTRPERM\|permission"; then
        echo "DENIED — need to set NVreg_RestrictProfilingToAdminUsers=0"
        echo "  Trying to fix..."
        # This usually works on EC2 instances where you have sudo
        if sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; then
            sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
            sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset
            echo "  Retrying..."
            if ncu --target-processes all /tmp/ncu_test 2>&1 | grep -q "ERR_NVGPUCTRPERM\|permission"; then
                echo "  Still denied. Profiling will be skipped."
                NCU_OK=false
            else
                echo "  Fixed!"
            fi
        else
            # Alternative: write to /etc/modprobe.d (requires reboot)
            echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu.conf
            echo "  Wrote modprobe config. You may need to reboot and re-run this script."
            NCU_OK=false
        fi
    else
        echo "OK"
    fi
    rm -f /tmp/ncu_test /tmp/ncu_test.cu
fi

echo ""

# ── Detect GPU arch ──────────────────────────────────────────────────────────
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${COMPUTE_CAP}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME (arch: $ARCH)"

# ── Step 1: Install Python deps ──────────────────────────────────────────────
echo ""
echo "──── Step 1: Python Dependencies ────"
pip install torch numpy --quiet --break-system-packages 2>/dev/null || \
pip install torch numpy --quiet 2>/dev/null || \
pip3 install torch numpy --quiet --break-system-packages 2>/dev/null || \
pip3 install torch numpy --quiet
echo "  torch + numpy installed"

# ── Step 2: Generate reference data ──────────────────────────────────────────
echo ""
echo "──── Step 2: Generate Reference Data ────"
cd "$PROJECT_DIR"
python3 generate_reference.py
echo ""

# ── Step 3: Build ─────────────────────────────────────────────────────────────
echo "──── Step 3: Build CUDA Kernels ────"
make clean
make ARCH="$ARCH"
echo "  Build successful"
echo ""

# ── Step 4: Correctness tests ────────────────────────────────────────────────
echo "──── Step 4: Correctness Tests ────"
./attention test test_data
echo ""

# ── Step 5: Benchmark ────────────────────────────────────────────────────────
echo "──── Step 5: Benchmark ────"
./attention bench 2>&1 | tee benchmark_results.txt
echo ""
echo "  Results saved to benchmark_results.txt"
echo ""

# ── Step 6: Nsight Compute profiling ─────────────────────────────────────────
if [ "$NCU_OK" = "true" ]; then
    echo "──── Step 6: Nsight Compute Profiling ────"
    # Profile just the fused kernel at N=1024 for the occupancy number
    ncu --set full \
        --kernel-name fused_attention_kernel \
        --launch-skip 10 --launch-count 1 \
        -o profile_fused \
        ./attention bench 2>&1 | tail -20
    echo ""
    echo "  Profile saved to profile_fused.ncu-rep"
    echo "  Download it and open in Nsight Compute GUI for detailed analysis."
    echo ""

    # Extract key metrics
    echo "Key metrics from ncu:"
    ncu --import profile_fused.ncu-rep \
        --csv \
        --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
        2>/dev/null | tail -2 || echo "  (manual extraction needed — open the .ncu-rep file)"
else
    echo "──── Step 6: Nsight Compute Profiling — SKIPPED (no ncu permissions) ────"
    echo "  Drop the occupancy clause from the resume bullet."
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All done!"
echo ""
echo "  GPU:    $GPU_NAME"
echo "  CUDA:   $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo "  Arch:   $ARCH"
echo ""
echo "  Benchmark results: benchmark_results.txt"
if [ "$NCU_OK" = "true" ]; then
echo "  Profile:           profile_fused.ncu-rep"
fi
echo ""
echo "  Fill these into your README.md, then:"
echo "    git add -A && git commit -m 'Add benchmark results'"
echo ""
echo "  When done, exit SSH and run:  ./aws/teardown.sh"
echo "════════════════════════════════════════════════════════════════"
