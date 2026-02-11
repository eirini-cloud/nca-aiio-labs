#!/usr/bin/env bash
# nvidia_smi_report.sh — Generate a quick GPU health report
set -euo pipefail

echo "=== NVIDIA GPU Report — $(date) ==="
echo ""

echo "--- Driver & CUDA ---"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader | head -1
echo ""

echo "--- GPU(s) ---"
nvidia-smi --query-gpu=index,name,pci.bus_id,persistence_mode,compute_mode --format=csv
echo ""

echo "--- Thermals & Power ---"
nvidia-smi --query-gpu=index,temperature.gpu,fan.speed,power.draw,power.limit --format=csv
echo ""

echo "--- Memory ---"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
echo ""

echo "--- Utilisation ---"
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv
echo ""

echo "--- Clocks ---"
nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv
echo ""

echo "--- Running Processes ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No compute processes running."
echo ""

echo "--- Topology ---"
nvidia-smi topo -m 2>/dev/null || echo "Topology not available (single GPU or unsupported)."
echo ""
echo "=== End of Report ==="
