# Lab 04 — NCCL all_reduce_perf

Benchmark GPU collective communication using NCCL (NVIDIA Collective Communications Library) `all_reduce_perf` tests.

## Objectives

- Understand what NCCL is and why it matters for multi-GPU / multi-node training
- Run `all_reduce_perf` to measure bandwidth and latency
- Interpret benchmark results
- Explore other NCCL collective operations

## Background

NCCL provides optimised primitives for collective communication across GPUs:
- **AllReduce** — combine values from all GPUs and distribute result to all
- **Broadcast** — send data from one GPU to all others
- **Reduce** — combine values from all GPUs, result on one GPU
- **AllGather** — gather data from all GPUs to all GPUs
- **ReduceScatter** — reduce then scatter across GPUs

These are critical for distributed deep learning (gradient synchronisation in data-parallel training).

## Prerequisites

- NVIDIA GPU(s) with driver 535+
- Docker with the NVIDIA Container Toolkit
- For meaningful multi-GPU tests: 2+ GPUs (single-GPU still works for syntax and baseline)

## 1 — Run the NCCL Tests Container

The easiest way to run `nccl-tests` is via the PyTorch NGC container (which includes NCCL) or by building from source.

### Option A: Using the PyTorch NGC Container

```bash
docker run --gpus all -it --rm \
  --shm-size=1g --ulimit memlock=-1 \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash
```

Inside the container, build nccl-tests:

```bash
cd /opt
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
```

### Option B: Build Directly on Host

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
```

## 2 — Run all_reduce_perf

```bash
# Basic all_reduce benchmark
# -b: starting message size
# -e: ending message size
# -f: factor (multiply size by this each step)
# -g: number of GPUs per thread
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### Example Output

```
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                                (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1     8.12    0.00    0.00      0     7.98    0.00    0.00      0
          16             4     float     sum      -1     8.05    0.00    0.00      0     7.92    0.00    0.00      0
          32             8     float     sum      -1     8.10    0.00    0.00      0     7.95    0.00    0.00      0
         ...
   134217728      33554432     float     sum      -1   2145.3   62.56   62.56      0   2138.1   62.78   62.78      0
```

### Understanding the Output

| Column | Meaning |
|--------|---------|
| `size` | Message size in bytes |
| `count` | Number of elements |
| `type` | Data type (float, half, etc.) |
| `redop` | Reduction operation (sum, prod, min, max) |
| `time` | Time in microseconds |
| `algbw` | Algorithm bandwidth (GB/s) = size / time |
| `busbw` | Bus bandwidth (GB/s) = algbw * correction factor for the collective |
| `#wrong` | Number of incorrect results (should be 0) |

**Bus bandwidth** is the more meaningful metric — it accounts for the communication pattern and tells you how efficiently you're using the interconnect.

## 3 — Vary Parameters

```bash
# Test with different data types
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -d float
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -d half
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -d int8

# Test with different reduction operations
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -o sum
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -o prod
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -o min
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -o max

# Test with larger message sizes (up to 1GB)
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 1

# Fixed number of iterations for stable measurements
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -n 100 -w 50
# -n: number of iterations
# -w: warmup iterations
```

## 4 — Multi-GPU Tests

```bash
# Use all available GPUs (e.g., 2 GPUs)
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2

# Specify which GPUs with CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
```

## 5 — Other NCCL Collectives

```bash
# AllGather
./build/all_gather_perf -b 8 -e 128M -f 2 -g 1

# Broadcast
./build/broadcast_perf -b 8 -e 128M -f 2 -g 1

# Reduce
./build/reduce_perf -b 8 -e 128M -f 2 -g 1

# ReduceScatter
./build/reduce_scatter_perf -b 8 -e 128M -f 2 -g 1

# SendRecv (point-to-point)
./build/sendrecv_perf -b 8 -e 128M -f 2 -g 1
```

## 6 — NCCL Environment Variables

These control NCCL behaviour and are useful for debugging and tuning:

```bash
# Show NCCL debug info
export NCCL_DEBUG=INFO

# Show detailed NCCL topology and algorithm selection
export NCCL_DEBUG=TRACE

# Force a specific transport
export NCCL_P2P_DISABLE=0       # Enable P2P (NVLink/PCIe)
export NCCL_SHM_DISABLE=0       # Enable shared memory

# Force specific algorithm
export NCCL_ALGO=Ring            # Ring, Tree, CollnetDirect, CollnetChain

# Force specific protocol
export NCCL_PROTO=Simple         # Simple, LL, LL128

# Example: run with debug info
NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## 7 — Interpreting Results for Single GPU

On a single GPU, `all_reduce_perf` is essentially a loopback test. The bandwidth numbers reflect GPU memory bandwidth rather than interconnect bandwidth. This is still useful for:
- Verifying NCCL is installed and working
- Establishing a memory bandwidth baseline
- Practising the commands before running on multi-GPU systems

## Expected Bandwidth Ranges

| Interconnect | Expected busbw |
|-------------|----------------|
| PCIe Gen4 x16 | ~25 GB/s |
| PCIe Gen5 x16 | ~50 GB/s |
| NVLink 3.0 (A100) | ~300 GB/s per GPU |
| NVLink 4.0 (H100) | ~450 GB/s per GPU |
| Single GPU loopback | Close to GPU memory bandwidth |

## Cleanup

```bash
# If you built nccl-tests on the host
rm -rf nccl-tests

# If using Docker, the container is already removed (--rm flag)
```
