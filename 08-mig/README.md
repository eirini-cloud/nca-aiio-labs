# Lab 08 — MIG (Multi-Instance GPU)

Partition a GPU into multiple isolated instances using NVIDIA MIG.

## Objectives

- Understand what MIG is and which GPUs support it
- Enable MIG mode on a GPU
- Create MIG GPU instances and compute instances
- Run workloads on individual MIG partitions
- Manage MIG lifecycle (create, list, destroy)

## Background

MIG allows a single GPU to be partitioned into up to **7 isolated GPU instances**, each with dedicated:
- **SM (Streaming Multiprocessor) compute resources**
- **GPU memory** (dedicated framebuffer partition)
- **Memory bandwidth** (isolated L2 cache)

Each partition is hardware-isolated — a fault or overload in one partition does not affect others.

### MIG-Capable GPUs

| GPU | MIG Support | Max Instances |
|-----|-------------|---------------|
| A100 (40/80GB) | Yes | 7 |
| A30 | Yes | 4 |
| H100 | Yes | 7 |
| RTX Ada Lovelace (some models) | Limited | Varies |

> **Note:** The RTX Ada Lovelace family has limited MIG support compared to data centre GPUs. This lab was initially attempted on AWS spot instances (A100) but can be adapted for local GPUs where supported. Check your GPU's MIG capability with `nvidia-smi -q | grep MIG`.

## Prerequisites

- MIG-capable NVIDIA GPU
- NVIDIA driver 535+
- Root/sudo access (MIG operations require elevated privileges)
- No running GPU processes (MIG mode changes require the GPU to be idle)

## 1 — Check MIG Capability

```bash
# Check if GPU supports MIG
nvidia-smi -q | grep -i "MIG Mode"

# Expected output for a MIG-capable GPU:
#   MIG Mode
#     Current: Disabled
#     Pending: Disabled

# Detailed MIG capability info
nvidia-smi -q | grep -A 3 "MIG"
```

## 2 — Enable MIG Mode

```bash
# Enable MIG mode (requires root, GPU must be idle)
sudo nvidia-smi -i 0 -mig 1

# You may need to reset the GPU or reboot
# Option 1: GPU reset (if supported)
sudo nvidia-smi -i 0 -r

# Option 2: Reboot
# sudo reboot

# Verify MIG is enabled
nvidia-smi -q | grep -i "MIG Mode"
# Expected:
#   Current: Enabled
#   Pending: Enabled
```

## 3 — List Available MIG Profiles

MIG profiles define how the GPU is partitioned. Each profile specifies a **GPU Instance (GI)** size.

```bash
# List available GPU Instance profiles
nvidia-smi mig -lgip

# Example output (A100 80GB):
# +-----------------------------------------------------------------------------+
# | GPU Instance Profiles:                                                       |
# |   GPU   Name          ID    Instances   Memory     P2P    SM    DEC   ENC   |
# |                              Free/Total   GiB             CE    JPEG  OFA   |
# |   0     MIG 1g.10gb    19     7/7        9.50       No     14    0     0    |
# |   0     MIG 2g.20gb    14     3/3        19.50      No     28    1     0    |
# |   0     MIG 3g.40gb     9     2/2        39.25      No     42    2     0    |
# |   0     MIG 4g.40gb     5     1/1        39.25      No     56    2     0    |
# |   0     MIG 7g.80gb     0     1/1        79.00      No     98    5     0    |
# +-----------------------------------------------------------------------------+

# Profile naming convention: <SMs>g.<memory>gb
# 1g.10gb = 1/7 of SMs, ~10GB memory
# 7g.80gb = all SMs, all memory (full GPU as one instance)
```

## 4 — Create GPU Instances

```bash
# Create a GPU Instance using a profile
# Example: Create two 3g.40gb instances on GPU 0
sudo nvidia-smi mig -cgi 9,9 -i 0

# Or create instances one at a time
sudo nvidia-smi mig -cgi 19 -i 0    # Create 1g.10gb
sudo nvidia-smi mig -cgi 19 -i 0    # Create another 1g.10gb

# Mixed partitioning example (A100 80GB):
# 1 x 3g.40gb + 2 x 2g.20gb
sudo nvidia-smi mig -cgi 9 -i 0     # 3g.40gb
sudo nvidia-smi mig -cgi 14,14 -i 0 # 2 x 2g.20gb

# List created GPU Instances
nvidia-smi mig -lgi
```

## 5 — Create Compute Instances

Each GPU Instance needs at least one **Compute Instance (CI)** before it can run workloads.

```bash
# List available Compute Instance profiles for existing GPU Instances
nvidia-smi mig -lcip

# Create a Compute Instance for each GPU Instance
# The -gi flag specifies which GPU Instance to add the CI to
sudo nvidia-smi mig -cci -gi 0    # CI for first GI
sudo nvidia-smi mig -cci -gi 1    # CI for second GI

# Or create CIs for all GPU Instances at once
sudo nvidia-smi mig -cci

# List all instances
nvidia-smi mig -lgi
nvidia-smi mig -lci
```

## 6 — View MIG Devices

```bash
# nvidia-smi now shows MIG devices
nvidia-smi

# Detailed MIG device listing
nvidia-smi -L

# Example output:
# GPU 0: NVIDIA A100 80GB (UUID: GPU-xxxx)
#   MIG 3g.40gb  Device 0: (UUID: MIG-xxxx)
#   MIG 2g.20gb  Device 1: (UUID: MIG-xxxx)
#   MIG 2g.20gb  Device 2: (UUID: MIG-xxxx)
```

## 7 — Run Workloads on MIG Instances

Use `CUDA_VISIBLE_DEVICES` to target a specific MIG instance:

```bash
# List MIG UUIDs
nvidia-smi -L

# Run on a specific MIG device using UUID
CUDA_VISIBLE_DEVICES=MIG-<uuid> nvidia-smi

# Run on a specific MIG device using index notation
# Format: MIG-GPU-<gpu-uuid>/<gi-index>/<ci-index>
CUDA_VISIBLE_DEVICES=MIG-GPU-<gpu-uuid>/0/0 nvidia-smi
```

### Docker with MIG

```bash
# Run a container on a specific MIG instance
docker run --gpus '"device=0:0"' -it --rm \
  nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04 \
  nvidia-smi

# Using the MIG UUID
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=MIG-<uuid> -it --rm \
  nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04 \
  nvidia-smi

# Run different workloads on different MIG instances simultaneously
docker run --gpus '"device=0:0"' -d --name mig-job-1 \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  python3 -c "import torch; x=torch.randn(1000,1000,device='cuda'); print('Job 1 done on', torch.cuda.get_device_name(0))"

docker run --gpus '"device=0:1"' -d --name mig-job-2 \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  python3 -c "import torch; x=torch.randn(1000,1000,device='cuda'); print('Job 2 done on', torch.cuda.get_device_name(0))"
```

## 8 — MIG with Kubernetes

With the NVIDIA Device Plugin or GPU Operator, MIG instances are exposed as separate resources:

### Single Strategy (each MIG device = 1 `nvidia.com/gpu`)

```bash
# Configure the device plugin for MIG single strategy
# In GPU Operator Helm values:
# devicePlugin.config.name: mig-single
```

```yaml
# Pod requesting a MIG device
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

### Mixed Strategy (resource names reflect MIG profiles)

```yaml
# Pod requesting a specific MIG profile
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod-specific
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/mig-3g.40gb: 1
```

## 9 — Reconfigure MIG Partitions

```bash
# Destroy all Compute Instances first
sudo nvidia-smi mig -dci -i 0

# Destroy all GPU Instances
sudo nvidia-smi mig -dgi -i 0

# Now create a new partitioning scheme
# Example: 7 x 1g.10gb instances (maximum partitioning on A100)
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -i 0
sudo nvidia-smi mig -cci

# Verify
nvidia-smi mig -lgi
nvidia-smi -L
```

## 10 — Disable MIG Mode

```bash
# First destroy all instances
sudo nvidia-smi mig -dci -i 0
sudo nvidia-smi mig -dgi -i 0

# Disable MIG mode
sudo nvidia-smi -i 0 -mig 0

# Reset GPU (or reboot)
sudo nvidia-smi -i 0 -r

# Verify
nvidia-smi -q | grep -i "MIG Mode"
# Expected:
#   Current: Disabled
#   Pending: Disabled
```

## AWS Spot Instance Notes

This lab was initially attempted on AWS EC2 spot instances with A100 GPUs:

- **Instance type:** `p4d.24xlarge` (8x A100 40GB) — expensive, spot pricing helps
- **AMI:** Deep Learning AMI (Ubuntu) — comes with NVIDIA drivers pre-installed
- **Gotcha:** Spot instances can be terminated at any time — save your work frequently
- **Tip:** Use `p3.2xlarge` (V100) for cheaper practice, but note V100 does **not** support MIG
- The lab was ultimately run locally on an RTX Ada Lovelace for more stable access

## Quick Reference

| Command | Description |
|---------|-------------|
| `nvidia-smi -i 0 -mig 1` | Enable MIG mode |
| `nvidia-smi -i 0 -mig 0` | Disable MIG mode |
| `nvidia-smi mig -lgip` | List GPU Instance profiles |
| `nvidia-smi mig -lcip` | List Compute Instance profiles |
| `nvidia-smi mig -cgi <id>` | Create GPU Instance |
| `nvidia-smi mig -cci` | Create Compute Instance |
| `nvidia-smi mig -lgi` | List GPU Instances |
| `nvidia-smi mig -lci` | List Compute Instances |
| `nvidia-smi mig -dci` | Destroy Compute Instances |
| `nvidia-smi mig -dgi` | Destroy GPU Instances |
| `nvidia-smi -L` | List GPUs and MIG devices |

## Cleanup

```bash
# Destroy all MIG instances
sudo nvidia-smi mig -dci -i 0
sudo nvidia-smi mig -dgi -i 0

# Disable MIG mode
sudo nvidia-smi -i 0 -mig 0
sudo nvidia-smi -i 0 -r

# Docker cleanup
docker rm -f mig-job-1 mig-job-2 2>/dev/null
```
