# Lab 07 — NVIDIA GPU Operator

Deploy the NVIDIA GPU Operator on Kubernetes to automate the management of all GPU software components.

## Objectives

- Understand what the GPU Operator provides and why it exists
- Install the GPU Operator via Helm
- Verify all GPU Operator components are running
- Deploy GPU workloads on a GPU Operator-managed cluster

## Background

The NVIDIA GPU Operator automates the provisioning of all software components needed to run GPU workloads on Kubernetes:

| Component | What It Does |
|-----------|--------------|
| **NVIDIA Driver** | GPU kernel driver (can install on nodes automatically) |
| **NVIDIA Container Toolkit** | Runtime hooks for GPU containers |
| **NVIDIA Device Plugin** | Exposes GPUs to the K8s scheduler |
| **DCGM / DCGM Exporter** | GPU monitoring and Prometheus metrics |
| **MIG Manager** | Manages MIG partitions |
| **GPU Feature Discovery** | Labels nodes with GPU properties |
| **NVIDIA MPS** | Multi-Process Service for GPU sharing |

Without the GPU Operator, you would need to install and manage each of these components individually on every GPU node.

## Prerequisites

- Kubernetes cluster (1.25+)
- GPU nodes with NVIDIA GPUs
- Helm 3 installed
- `kubectl` configured to talk to your cluster

## 1 — Install Helm (if not already installed)

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

## 2 — Add the NVIDIA Helm Repository

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

## 3 — Install the GPU Operator

### Option A: Fresh Cluster (GPU Operator installs the driver)

If the nodes do **not** have NVIDIA drivers installed:

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --wait
```

### Option B: Nodes Already Have Drivers

If the host already has NVIDIA drivers installed (common for local dev):

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --wait
```

### Option C: With Custom Configuration

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgm.enabled=true \
  --set dcgmExporter.enabled=true \
  --set migManager.enabled=false \
  --set mps.enabled=false \
  --set gfd.enabled=true \
  --wait
```

## 4 — Verify Installation

```bash
# Check all GPU Operator pods are running
kubectl get pods -n gpu-operator

# Expected pods (names will vary):
# gpu-operator-xxxx                          Running
# nvidia-container-toolkit-daemonset-xxxx    Running
# nvidia-device-plugin-daemonset-xxxx        Running
# nvidia-dcgm-exporter-xxxx                  Running
# nvidia-dcgm-xxxx                           Running
# gpu-feature-discovery-xxxx                 Running
# nvidia-operator-validator-xxxx             Running

# Wait for all pods to be ready
kubectl wait --for=condition=Ready pods --all -n gpu-operator --timeout=300s

# Check node GPU resources
kubectl describe nodes | grep -A 5 "nvidia.com"
```

## 5 — GPU Feature Discovery Labels

The GPU Operator's GFD component adds labels to nodes describing GPU properties:

```bash
# View GPU-related node labels
kubectl get nodes -o json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for node in data['items']:
    name = node['metadata']['name']
    labels = {k: v for k, v in node['metadata']['labels'].items() if 'nvidia' in k.lower()}
    print(f'\n=== {name} ===')
    for k, v in sorted(labels.items()):
        print(f'  {k}: {v}')
"
```

Common GFD labels:

| Label | Example Value | Description |
|-------|--------------|-------------|
| `nvidia.com/gpu.product` | `NVIDIA-RTX-6000-Ada-Generation` | GPU model |
| `nvidia.com/gpu.memory` | `49140` | GPU memory (MiB) |
| `nvidia.com/gpu.count` | `1` | Number of GPUs |
| `nvidia.com/cuda.driver.major` | `535` | Driver major version |
| `nvidia.com/cuda.runtime.major` | `12` | CUDA major version |
| `nvidia.com/gpu.compute.major` | `8` | Compute capability major |
| `nvidia.com/mig.capable` | `true` | MIG support |

## 6 — DCGM Exporter Metrics

The GPU Operator deploys DCGM exporter automatically:

```bash
# Find the dcgm-exporter pod
DCGM_POD=$(kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter -o jsonpath='{.items[0].metadata.name}')

# Port-forward to access metrics
kubectl port-forward -n gpu-operator $DCGM_POD 9400:9400 &

# Query metrics
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_TEMP
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL

# Stop port-forward
kill %1
```

## 7 — Run a GPU Workload

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-operator-test
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda-test
    image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
    command: ["bash", "-c"]
    args:
    - |
      echo "=== nvidia-smi ==="
      nvidia-smi
      echo ""
      echo "=== NVIDIA Driver Version ==="
      cat /proc/driver/nvidia/version 2>/dev/null || echo "Not available"
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

sleep 20
kubectl logs gpu-operator-test
kubectl delete pod gpu-operator-test
```

## 8 — GPU Operator Validation

The operator includes a validator that runs automated checks:

```bash
# Check the validator pod
kubectl get pods -n gpu-operator | grep validator

# View validation results
VALIDATOR_POD=$(kubectl get pods -n gpu-operator -l app=nvidia-operator-validator -o jsonpath='{.items[0].metadata.name}')
kubectl logs -n gpu-operator $VALIDATOR_POD

# The validator checks:
# - Driver is loaded
# - Container toolkit is working
# - Device plugin is advertising GPUs
# - CUDA workloads can run
```

## 9 — Upgrading the GPU Operator

```bash
# Update Helm repos
helm repo update

# Check available versions
helm search repo nvidia/gpu-operator --versions | head -10

# Upgrade
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false \
  --wait
```

## 10 — Customising the GPU Operator

### View Current Configuration

```bash
helm get values gpu-operator -n gpu-operator
```

### Override Configuration

```bash
# Example: Enable MIG Manager
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false \
  --set migManager.enabled=true \
  --wait

# Example: Set custom DCGM exporter metrics
helm upgrade gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --set driver.enabled=false \
  --set dcgmExporter.config.name=custom-dcgm-metrics \
  --wait
```

## Troubleshooting

```bash
# Check operator logs
kubectl logs -n gpu-operator deployment/gpu-operator

# Check individual component logs
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
kubectl logs -n gpu-operator -l app=nvidia-container-toolkit-daemonset
kubectl logs -n gpu-operator -l app=nvidia-dcgm-exporter

# Describe a failing pod
kubectl describe pod -n gpu-operator <pod-name>

# Check events
kubectl get events -n gpu-operator --sort-by='.lastTimestamp'
```

| Issue | Fix |
|-------|-----|
| Driver pod in CrashLoopBackOff | If host has drivers, set `driver.enabled=false` |
| Device plugin not finding GPUs | Check host driver with `nvidia-smi` on the node |
| Toolkit pods failing | Ensure container runtime (Docker/containerd) is properly configured |
| Validator failing CUDA test | Check all preceding components are healthy first |

## Cleanup

```bash
# Uninstall the GPU Operator
helm uninstall gpu-operator -n gpu-operator

# Remove the namespace
kubectl delete namespace gpu-operator

# Remove CRDs (if you want a complete removal)
kubectl get crd | grep nvidia | awk '{print $1}' | xargs kubectl delete crd
```
