# NCA AIIO Labs

Hands-on lab exercises covering the NVIDIA Certified Associate — AI Infrastructure and Operations (NCA-AIIO) exam topics.

These labs were developed and tested during exam preparation, primarily on an **NVIDIA RTX Ada Lovelace** GPU (with some initial attempts on AWS spot instances).

## Labs

| # | Lab | Topics |
|---|-----|--------|
| 01 | [nvidia-smi](01-nvidia-smi/) | GPU inspection, monitoring, querying, process management |
| 02 | [DCGM Metrics](02-dcgm-metrics/) | DCGM container, `dcgmi`, metrics collection with Prometheus |
| 03 | [NGC Auth & Containers](03-ngc-auth-containers/) | NGC CLI setup, authentication, pulling containers |
| 04 | [NCCL all_reduce_perf](04-nccl-all-reduce/) | NCCL collective benchmarks, bandwidth & latency testing |
| 05 | [Triton Dynamic Batching](05-triton-dynamic-batching/) | Triton Inference Server, model repository, dynamic batching |
| 06 | [Kubernetes for GPU Workloads](06-kubernetes-gpu/) | K8s cluster setup, scheduling GPU pods, resource limits |
| 07 | [GPU Operator](07-gpu-operator/) | NVIDIA GPU Operator on Kubernetes via Helm |
| 08 | [MIG — Multi-Instance GPU](08-mig/) | MIG partitioning, profiles, compute instances |

## Prerequisites

- Linux host (Ubuntu 22.04+ recommended)
- NVIDIA GPU with recent drivers (535+)
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For K8s labs: a running Kubernetes cluster (minikube, kubeadm, or managed)
- NGC account (free) at <https://ngc.nvidia.com>

## Environment Used

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA RTX Ada Lovelace (local), various instances on AWS (spot) |
| OS | Ubuntu 22.04 LTS |
| Driver | 535+ |
| CUDA | 12.x |
| Container Runtime | Docker + NVIDIA Container Toolkit |
| Kubernetes | kubeadm / minikube |

## Exam Resources

- [NCA-AIIO Exam Page](https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)

## Disclaimer

These labs are for educational and reference purposes. Commands and outputs may vary depending on your GPU model, driver version, and environment. Always verify against official NVIDIA documentation.
