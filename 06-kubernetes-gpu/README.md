# Lab 06 — Kubernetes for GPU Workloads

Set up a Kubernetes cluster and schedule GPU-accelerated workloads.

## Objectives

- Set up a K8s environment capable of scheduling GPU pods
- Understand GPU resource requests and limits in Kubernetes
- Deploy GPU workloads as pods and jobs
- Monitor GPU usage in a K8s cluster

## Prerequisites

- NVIDIA GPU with driver 535+
- Docker with NVIDIA Container Toolkit
- `kubectl` installed
- A Kubernetes cluster (this lab covers minikube and kubeadm approaches)

## 1 — Cluster Setup Options

### Option A: minikube (Single Node — Easiest for Learning)

```bash
# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# Start minikube with GPU support
minikube start --driver=docker --gpus all

# Verify
minikube status
kubectl get nodes
```

### Option B: kubeadm (Multi-Node — Closer to Production)

```bash
# Install kubeadm, kubelet, kubectl (on all nodes)
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | \
  sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg

echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | \
  sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Initialise the control plane (on master node)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Set up kubectl for the current user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install a CNI (e.g., Flannel)
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# If single-node, allow scheduling on the control plane
kubectl taint nodes --all node-role.kubernetes.io/control-plane-

# Verify
kubectl get nodes
```

## 2 — Install the NVIDIA Device Plugin

The NVIDIA device plugin makes GPUs visible to the Kubernetes scheduler.

```bash
# Deploy the NVIDIA device plugin as a DaemonSet
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# Verify the device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Check that GPUs are advertised as allocatable resources
kubectl describe nodes | grep -A 5 "Capacity:" | grep nvidia
kubectl describe nodes | grep -A 5 "Allocatable:" | grep nvidia
```

Expected output:

```
nvidia.com/gpu:     1
```

## 3 — Run a Simple GPU Pod

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda-test
    image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Wait for the pod to complete
kubectl wait --for=condition=Ready pod/gpu-test --timeout=120s 2>/dev/null || true
sleep 5

# Check the output
kubectl logs gpu-test

# Clean up
kubectl delete pod gpu-test
```

## 4 — Run a GPU Job (Batch Processing)

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-vector-add
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: vector-add
        image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
        command: ["/bin/bash", "-c"]
        args:
        - |
          echo "=== GPU Info ==="
          nvidia-smi
          echo ""
          echo "=== CUDA Device Query ==="
          nvidia-smi -q | head -30
        resources:
          limits:
            nvidia.com/gpu: 1
  backoffLimit: 4
EOF

# Watch the job
kubectl get jobs -w

# Get the logs
kubectl logs job/gpu-vector-add

# Clean up
kubectl delete job gpu-vector-add
```

## 5 — Run a PyTorch Training Pod

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-gpu
spec:
  restartPolicy: OnFailure
  containers:
  - name: pytorch
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command: ["python3", "-c"]
    args:
    - |
      import torch
      print(f"PyTorch version: {torch.__version__}")
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"GPU count: {torch.cuda.device_count()}")
      if torch.cuda.is_available():
          print(f"GPU name: {torch.cuda.get_device_name(0)}")
          # Quick CUDA operation
          x = torch.randn(1000, 1000, device='cuda')
          y = torch.randn(1000, 1000, device='cuda')
          z = torch.matmul(x, y)
          print(f"Matrix multiply result shape: {z.shape}")
          print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")
      print("Done!")
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Wait and check logs
sleep 30
kubectl logs pytorch-gpu

# Clean up
kubectl delete pod pytorch-gpu
```

## 6 — GPU Resource Management

### Understanding GPU Resource Requests and Limits

```yaml
resources:
  limits:
    nvidia.com/gpu: 1    # Request exactly 1 GPU
```

Key points:
- GPUs are **non-shareable** by default — a pod gets exclusive access to a whole GPU
- You can only set `limits`, not `requests`, for `nvidia.com/gpu`
- A pod requesting 1 GPU will have exclusive access to that GPU
- If no GPUs are available, the pod stays in `Pending` state
- MIG (Lab 08) allows GPU partitioning for sharing

### Check GPU Allocation

```bash
# See which pods are using GPUs
kubectl describe nodes | grep -A 10 "Allocated resources"

# See pending pods waiting for GPUs
kubectl get pods --field-selector=status.phase=Pending
```

## 7 — Node Labels and GPU Scheduling

```bash
# Label a GPU node
kubectl label nodes <node-name> gpu-type=rtx-ada-lovelace

# Schedule a pod on a specific GPU node
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-specific-node
spec:
  restartPolicy: OnFailure
  nodeSelector:
    gpu-type: rtx-ada-lovelace
  containers:
  - name: cuda-test
    image: nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF
```

## 8 — Monitoring GPU Pods

```bash
# Watch GPU pod status
kubectl get pods -w

# Describe a pod to see GPU allocation events
kubectl describe pod <pod-name>

# View resource usage (requires metrics-server)
kubectl top pods
kubectl top nodes

# Check events for GPU scheduling issues
kubectl get events --sort-by='.lastTimestamp' | grep -i gpu
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Pod stuck in `Pending` | No GPU available | Check `kubectl describe pod` for events |
| `nvidia.com/gpu` not in node resources | Device plugin not running | Deploy/restart the NVIDIA device plugin |
| `Failed to initialize NVML` | Container can't access GPU | Ensure NVIDIA Container Toolkit is installed |
| Image pull errors for `nvcr.io` | Not authenticated to NGC | Run `docker login nvcr.io` on the node |

## Cleanup

```bash
kubectl delete pod gpu-test pytorch-gpu gpu-specific-node 2>/dev/null
kubectl delete job gpu-vector-add 2>/dev/null

# If using minikube
minikube stop
```
