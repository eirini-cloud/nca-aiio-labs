# Lab 01 - nvidia-smi

GPU inspection, monitoring, and process management using `nvidia-smi`.

## Objectives

- Query GPU hardware and driver information
- Monitor GPU utilisation, memory, temperature, and power in real time
- Understand output formats (human-readable, CSV, XML)
- Manage GPU compute modes and persistence mode

## Evidence (Windows Host - GPU Ready)

<img width="1255" height="911" alt="image" src="https://github.com/user-attachments/assets/f0ed10a2-dea5-44a7-915b-b50513eba0aa" />


Ran `nvidia-smi` on **Wed Feb 11 2026** to confirm the driver stack is functioning:

| Item | Value |
|------|-------|
| GPU detected | NVIDIA GeForce RTX 4070 |
| Driver | 572.16 |
| CUDA runtime (reported by driver) | 12.8 |
| Display mode | WDDM (Windows Display Driver Model) |
| VRAM at capture | ~1330 MiB / 12282 MiB in use |
| GPU utilisation at capture | ~6 % |
| Process evidence | Windows desktop apps (e.g. Chrome) visible in the `nvidia-smi` process list — confirms the OS + driver stack is functioning end-to-end |
| Monitoring proof | Continuous `dmon`-style sample view captured (P-states, clocks, power, memory) |

## Validation Commands

Copy-paste to reproduce:

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi dmon -s pucm          # optional — continuous monitoring
```

> **WDDM note:** On Windows (WDDM), `nvidia-smi` will show desktop and video-decode workloads (Chrome, etc.) under the process list. For compute and container labs, ensure Docker / WSL2 is configured to use the NVIDIA runtime.

## Prerequisites

- NVIDIA GPU with driver 535+
- `nvidia-smi` available on `$PATH` (installed with the driver)

## 1 — Basic GPU Information

```bash
# Full default output — driver version, CUDA version, GPU name, temp, power, memory, utilisation
nvidia-smi
```

Example output (RTX Ada Lovelace):

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x                 |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name        Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada   Off  | 00000000:01:00.0 Off |                  Off |
| 30%   35C    P8    20W / 300W   |    512MiB / 49140MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

## 2 - Querying Specific Fields

`nvidia-smi` supports `--query-gpu` with a wide range of fields:

```bash
# List all available query fields
nvidia-smi --help-query-gpu

# Query specific fields as CSV
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,power.limit --format=csv

# Same query without header and units (useful for scripting)
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

## 3 - Real-Time Monitoring

```bash
# Continuous monitoring — refreshes every 1 second (like top for GPUs)
nvidia-smi dmon -s pucvmet -d 1

# Loop mode — reruns nvidia-smi every 2 seconds
nvidia-smi -l 2

# Or use watch
watch -n 1 nvidia-smi
```

### `dmon` field groups

| Flag | Fields |
|------|--------|
| `p` | Power |
| `u` | Utilisation |
| `c` | Clocks |
| `v` | Violations (thermal, power, etc.) |
| `m` | Memory |
| `e` | ECC errors |
| `t` | Temperature |

## 4 - Process Information

```bash
# Show all GPU processes
nvidia-smi pmon -s m -d 1

# Query compute processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## 5 - Persistence Mode

Persistence mode keeps the driver loaded even when no GPU application is running. This avoids driver initialisation latency for subsequent GPU tasks.

```bash
# Check current persistence mode
nvidia-smi -q | grep "Persistence Mode"

# Enable persistence mode (requires root)
sudo nvidia-smi -pm 1

# Disable persistence mode
sudo nvidia-smi -pm 0
```

## 6 - Compute Mode

Controls which processes can use the GPU.

| Mode | Description |
|------|-------------|
| `0` — Default | Multiple processes can use the GPU |
| `1` — Exclusive Thread | Deprecated |
| `2` — Prohibited | No compute processes allowed |
| `3` — Exclusive Process | Only one context allowed at a time |

```bash
# Set exclusive process mode
sudo nvidia-smi -c 3

# Reset to default
sudo nvidia-smi -c 0
```

## 7 - Clock and Power Management

```bash
# Query current clocks
nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv

# List supported clock speeds
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock clocks to max (useful for benchmarking)
sudo nvidia-smi -lgc <max_graphics_clock>

# Reset clocks
sudo nvidia-smi -rgc

# Set power limit (watts)
sudo nvidia-smi -pl 250
```

## 8 - XML Output

```bash
# Full XML output (all GPU details)
nvidia-smi -q -x > gpu_info.xml

# Useful for automation / parsing with xmllint or Python
nvidia-smi -q -x | python3 -c "
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.stdin).getroot()
for gpu in root.findall('gpu'):
    name = gpu.find('product_name').text
    temp = gpu.find('temperature/gpu_temp').text
    print(f'{name}: {temp}')
"
```

## 9 - Topology

```bash
# Show GPU topology (NVLink, PCIe relationships)
nvidia-smi topo -m
```

## Useful One-Liners

```bash
# Quick GPU memory check
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Is the GPU busy?
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# Log GPU stats to file every 5 seconds
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used --format=csv -l 5 > gpu_log.csv
```

## Script

See [`nvidia_smi_report.sh`](nvidia_smi_report.sh) for an automated report script.
