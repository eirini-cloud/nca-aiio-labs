# Lab 02 - DCGM Container & Metrics (Windows + Docker Desktop)

Deploy NVIDIA Data Center GPU Manager (DCGM) as a container and validate GPU telemetry/health tooling. Optionally expose metrics in Prometheus format via `dcgm-exporter`.

## Objectives

* Run the DCGM host engine container using Docker + NVIDIA Container Toolkit
* Use `dcgmi` to discover GPUs and run basic diagnostics
* (Optional) Expose DCGM metrics in Prometheus format with `dcgm-exporter`
* Understand the difference between **GPU discovery**, **health watches**, and **diagnostic deployment checks**

<img width="798" height="235" alt="docker-desktop-dcgm-running" src="https://github.com/user-attachments/assets/87229754-5b9e-43cb-a3de-dc30de5b3597" />

<img width="652" height="966" alt="dcgmi-discovery-and-diag" src="https://github.com/user-attachments/assets/49d60f1b-1e20-455c-b378-8f7433c2f4c2" />

## Prerequisites

* NVIDIA GPU + working driver install (validated via `nvidia-smi` on host)
* **Docker Desktop running** with **Linux containers** (WSL2 backend)
* NVIDIA Container Toolkit working (`docker run --gpus all …` functions)
* NGC access / ability to pull images from `nvcr.io`

---

## 0 - Troubleshooting checkpoint (Docker daemon)

If you see:
`failed to connect to the docker API at npipe:...dockerDesktopLinuxEngine`
→ Docker Desktop isn't running (or Linux engine not enabled). Start Docker Desktop and ensure **Linux containers**.

---

## 1 - Pull DCGM Image

PowerShell:

```powershell
docker pull nvcr.io/nvidia/cloud-native/dcgm:3.3.5-1-ubuntu22.04
```

---

## 2 - Run DCGM Container (PowerShell)

PowerShell (single line — PowerShell does **not** use `\` for line continuation):

```powershell
docker rm -f dcgm 2>$null
docker run --gpus all -d --name dcgm --privileged --pid=host nvcr.io/nvidia/cloud-native/dcgm:3.3.5-1-ubuntu22.04
```

Verify:

```powershell
docker ps --filter "name=dcgm"
docker logs dcgm --tail 50
```

**Expected**

* Container shows `Up`
* Logs include:

  * `Started host engine version 3.3.5 using port number: 5555`

---

## 3 - Use `dcgmi` inside the container

Enter the container:

```powershell
docker exec -it dcgm bash
```

Inside container:

```bash
# Confirm the GPU is visible
nvidia-smi

# Discover GPUs via DCGM
dcgmi discovery -l

# Detailed info for GPU 0
dcgmi discovery -i 0 -v
```

**Expected**

* `dcgmi discovery -l` reports **1 GPU found** and lists RTX 4070 details.

---

## 4 - Health Watches (enable + report)

Health report showing all "Off" usually means watches aren't enabled yet.

Inside container:

```bash
# Enable all health watches for GPU 0
dcgmi health -s -g 0 -a

# Show health status
dcgmi health -f -g 0
```

**Expected**

* Health modules show as enabled/active.
* Any warnings/errors will be listed per module.

---

## 5 - Diagnostics (Quick)

Inside container:

```bash
dcgmi diag -r 1
```

**Expected**

* Diagnostics run and return a table of results.

**Known limitation (Windows + Docker Desktop/WSL2 + consumer GPU)**
You may see:

* `Deployment -> Permissions and OS Blocks: Fail`
* Error: *"The number of devices NVML returns is different than the number of devices in /dev …"*

This is a common container/WSL2 enumeration quirk. Treat **successful discovery + working metrics** as the primary validation signals for this lab.

---

## 6 - DCGM Exporter (Prometheus metrics) Optional

Run exporter on the host (PowerShell):

```powershell
docker rm -f dcgm-exporter 2>$null
docker run -d --gpus all --name dcgm-exporter -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04
```

Test endpoint (PowerShell):

```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:9400/metrics).Content.Split("`n") | Select-Object -First 30
```

Filter examples:

```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:9400/metrics).Content | Select-String "DCGM_FI_DEV_GPU_TEMP"
(Invoke-WebRequest -UseBasicParsing http://localhost:9400/metrics).Content | Select-String "DCGM_FI_DEV_GPU_UTIL"
(Invoke-WebRequest -UseBasicParsing http://localhost:9400/metrics).Content | Select-String "DCGM_FI_DEV_FB_USED"
(Invoke-WebRequest -UseBasicParsing http://localhost:9400/metrics).Content | Select-String "DCGM_FI_DEV_POWER_USAGE"
```

**Expected**

* `/metrics` returns text including multiple `DCGM_FI_*` metrics.

---


## Cleanup

PowerShell:

```powershell
docker rm -f dcgm dcgm-exporter 2>$null
```
