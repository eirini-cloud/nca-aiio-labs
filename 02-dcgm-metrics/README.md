# Lab 02 — DCGM Container & Metrics

Deploy the NVIDIA Data Center GPU Manager (DCGM) as a container and collect GPU metrics for monitoring.

## Objectives

- Run the DCGM container using the NVIDIA Container Toolkit
- Use `dcgmi` to inspect GPU health, diagnostics, and telemetry
- Expose DCGM metrics in Prometheus format with `dcgm-exporter`
- Understand field groups and watched fields

## Prerequisites

- NVIDIA GPU with driver 535+
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NGC access for pulling the DCGM image

## 1 — Run DCGM Container

```bash
# Pull and run the DCGM container
docker run --gpus all -d --name dcgm \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/cloud-native/dcgm:3.3.5-1-ubuntu22.04

# Verify it's running
docker ps | grep dcgm
```

## 2 — Using dcgmi Inside the Container

```bash
# Open a shell in the running container
docker exec -it dcgm bash

# --- Inside the container ---

# Discover GPUs
dcgmi discovery -l

# Show detailed GPU info
dcgmi discovery -i 0 -v

# Check system health (quick)
dcgmi health -c -g 0

# Check health status
dcgmi health -f -g 0

# Run diagnostics — level 1 (quick), 2 (medium), 3 (long)
dcgmi diag -r 1

# Run level 3 diagnostics (comprehensive — includes memory, PCIe, compute stress)
dcgmi diag -r 3
```

## 3 — Field Groups and Watches

DCGM organises GPU telemetry into **fields** grouped by category. You can watch fields to collect time-series data.

```bash
# List available field groups
dcgmi fieldgroup -l

# List all introspectable fields
dcgmi introspect -s

# Watch a field group (e.g., default group) — sample every 1000ms, keep for 30s
dcgmi dmon -e 150,155,203,204,1001,1002,1003,1004,1005 -d 1000

# Field IDs reference:
# 150 = GPU temperature
# 155 = Power usage
# 203 = GPU utilisation
# 204 = Memory utilisation
# 1001 = Total memory
# 1002 = Free memory
# 1003 = Used memory
# 1004 = SM clock
# 1005 = Memory clock
```

## 4 — DCGM Exporter (Prometheus Metrics)

`dcgm-exporter` exposes GPU metrics on an HTTP endpoint for Prometheus to scrape.

```bash
# Run dcgm-exporter
docker run -d --gpus all --name dcgm-exporter \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04

# Test the metrics endpoint
curl -s localhost:9400/metrics | head -50

# Filter for specific metrics
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_TEMP
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_FB_USED
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_POWER_USAGE
```

### Key Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (C) |
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilisation (%) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory utilisation (%) |
| `DCGM_FI_DEV_FB_FREE` | Framebuffer free (MiB) |
| `DCGM_FI_DEV_FB_USED` | Framebuffer used (MiB) |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) |
| `DCGM_FI_DEV_SM_CLOCK` | SM clock (MHz) |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock (MHz) |
| `DCGM_FI_DEV_ENC_UTIL` | Encoder utilisation (%) |
| `DCGM_FI_DEV_DEC_UTIL` | Decoder utilisation (%) |
| `DCGM_FI_DEV_PCIE_TX_THROUGHPUT` | PCIe TX (KB/s) |
| `DCGM_FI_DEV_PCIE_RX_THROUGHPUT` | PCIe RX (KB/s) |
| `DCGM_FI_DEV_XID_ERRORS` | XID error count |

## 5 — Custom Metrics CSV

You can specify which metrics to export by mounting a custom CSV file:

```bash
# Create a custom metrics file
cat > custom-dcgm-metrics.csv << 'EOF'
# Custom DCGM metrics for NCA-AIIO lab
DCGM_FI_DEV_GPU_TEMP,       gauge, Temperature of the GPU (C).
DCGM_FI_DEV_GPU_UTIL,       gauge, GPU utilisation (%).
DCGM_FI_DEV_FB_USED,        gauge, Framebuffer memory used (MiB).
DCGM_FI_DEV_FB_FREE,        gauge, Framebuffer memory free (MiB).
DCGM_FI_DEV_POWER_USAGE,    gauge, Power draw (W).
DCGM_FI_DEV_PCIE_TX_THROUGHPUT, gauge, PCIe TX throughput (KB/s).
DCGM_FI_DEV_PCIE_RX_THROUGHPUT, gauge, PCIe RX throughput (KB/s).
EOF

# Run dcgm-exporter with the custom metrics
docker run -d --gpus all --name dcgm-exporter-custom \
  -p 9401:9400 \
  -v "$(pwd)/custom-dcgm-metrics.csv:/etc/dcgm-exporter/customized.csv" \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04 \
  -f /etc/dcgm-exporter/customized.csv

curl -s localhost:9401/metrics | grep DCGM
```

## 6 — DCGM Policy and Alerts

```bash
docker exec -it dcgm bash

# Set a policy — alert if GPU temperature exceeds 85C
dcgmi policy --set 0,0 -t 85

# View current policies
dcgmi policy --get

# Register for policy violations (runs in foreground)
dcgmi policy --reg
```

## Cleanup

```bash
docker stop dcgm dcgm-exporter dcgm-exporter-custom 2>/dev/null
docker rm dcgm dcgm-exporter dcgm-exporter-custom 2>/dev/null
```
