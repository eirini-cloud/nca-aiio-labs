# Lab 05 — Triton Inference Server & Dynamic Batching

Deploy NVIDIA Triton Inference Server, serve a model, and configure dynamic batching for improved throughput.

## Objectives

- Set up a Triton model repository
- Deploy Triton Inference Server as a container
- Serve a simple ONNX model
- Configure and test dynamic batching
- Use `perf_analyzer` to measure performance
- Understand Triton's batching strategies

## Prerequisites

- NVIDIA GPU with driver 535+
- Docker with NVIDIA Container Toolkit
- ~10 GB disk space for the Triton container

## 1 — Create a Model Repository

Triton requires models in a specific directory structure:

```
model_repository/
└── simple_model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### Generate a Simple ONNX Model

```bash
mkdir -p model_repository/simple_model/1

# Create a simple ONNX model using Python
docker run --gpus all --rm \
  -v "$(pwd)/model_repository:/workspace/model_repository" \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  python3 -c "
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleModel()
model.eval()
dummy_input = torch.randn(1, 16)
torch.onnx.export(
    model,
    dummy_input,
    '/workspace/model_repository/simple_model/1/model.onnx',
    input_names=['INPUT'],
    output_names=['OUTPUT'],
    dynamic_axes={'INPUT': {0: 'batch_size'}, 'OUTPUT': {0: 'batch_size'}}
)
print('Model exported successfully')
"
```

### Create the Model Configuration (No Dynamic Batching)

```bash
cat > model_repository/simple_model/config.pbtxt << 'EOF'
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 8 ]
  }
]
EOF
```

## 2 — Start Triton Inference Server

```bash
docker run --gpus all -d --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Wait for the server to be ready
sleep 10

# Check server health
curl -s localhost:8000/v2/health/ready
# Expected: 200 OK

# Check loaded models
curl -s localhost:8000/v2/models | python3 -m json.tool
```

### Triton Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | REST API |
| 8001 | gRPC | gRPC API |
| 8002 | HTTP | Prometheus metrics |

## 3 — Send Inference Requests

```bash
# Send a single inference request via HTTP
curl -s -X POST localhost:8000/v2/models/simple_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "INPUT",
      "shape": [1, 16],
      "datatype": "FP32",
      "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
               9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    }]
  }' | python3 -m json.tool
```

## 4 — Enable Dynamic Batching

Dynamic batching combines multiple inference requests into a single batch to improve GPU utilisation and throughput.

```bash
cat > model_repository/simple_model/config.pbtxt << 'EOF'
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "INPUT"
    data_type: TYPE_FP32
    dims: [ 16 ]
  }
]

output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 8 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
EOF
```

### Reload the Model

```bash
# Option 1: Restart Triton
docker restart triton
sleep 10

# Option 2: Use the model control API (if --model-control-mode=explicit was set)
# curl -X POST localhost:8000/v2/repository/models/simple_model/load
```

## 5 — Performance Testing with perf_analyzer

`perf_analyzer` is Triton's built-in benchmarking tool.

```bash
# Run perf_analyzer from the Triton SDK container
docker run --rm --net host \
  nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
  perf_analyzer -m simple_model -u localhost:8000 \
    --concurrency-range 1:16:2 \
    --shape INPUT:16 \
    -i http

# Key output columns:
# Concurrency — number of simultaneous requests
# Inferences/Second — throughput
# p50/p90/p99 latency — latency percentiles
```

### Compare: Without vs With Dynamic Batching

```bash
# Test without dynamic batching (single request at a time)
docker run --rm --net host \
  nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
  perf_analyzer -m simple_model -u localhost:8000 \
    --concurrency-range 1:1 \
    --shape INPUT:16 \
    -i http

# Test with dynamic batching under load (multiple concurrent requests)
docker run --rm --net host \
  nvcr.io/nvidia/tritonserver:24.01-py3-sdk \
  perf_analyzer -m simple_model -u localhost:8000 \
    --concurrency-range 1:32:4 \
    --shape INPUT:16 \
    -i http
```

## 6 — Understanding Dynamic Batching Configuration

| Parameter | Description |
|-----------|-------------|
| `max_batch_size` | Maximum batch size Triton will form |
| `preferred_batch_size` | Batch sizes Triton tries to form before executing |
| `max_queue_delay_microseconds` | Max time (us) Triton waits to form a preferred batch |

### Trade-offs

- **Lower `max_queue_delay_microseconds`** → lower latency, smaller batches, lower throughput
- **Higher `max_queue_delay_microseconds`** → higher latency, larger batches, higher throughput
- **Larger `preferred_batch_size`** → more GPU utilisation, but requires more concurrent requests

### Experiment with Different Delays

```bash
# Low delay (latency-optimised)
cat > model_repository/simple_model/config.pbtxt << 'EOF'
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [ { name: "INPUT", data_type: TYPE_FP32, dims: [ 16 ] } ]
output [ { name: "OUTPUT", data_type: TYPE_FP32, dims: [ 8 ] } ]
dynamic_batching { preferred_batch_size: [ 4, 8 ], max_queue_delay_microseconds: 50 }
EOF

docker restart triton && sleep 10

# High delay (throughput-optimised)
cat > model_repository/simple_model/config.pbtxt << 'EOF'
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [ { name: "INPUT", data_type: TYPE_FP32, dims: [ 16 ] } ]
output [ { name: "OUTPUT", data_type: TYPE_FP32, dims: [ 8 ] } ]
dynamic_batching { preferred_batch_size: [ 16, 32 ], max_queue_delay_microseconds: 5000 }
EOF

docker restart triton && sleep 10
```

## 7 — Triton Metrics (Prometheus)

```bash
# Scrape Triton's metrics endpoint
curl -s localhost:8002/metrics | head -60

# Key metrics to watch:
curl -s localhost:8002/metrics | grep nv_inference_request_success
curl -s localhost:8002/metrics | grep nv_inference_queue_duration_us
curl -s localhost:8002/metrics | grep nv_inference_compute_infer_duration_us
curl -s localhost:8002/metrics | grep nv_inference_request_duration_us
curl -s localhost:8002/metrics | grep nv_gpu_utilization
```

### Key Triton Metrics

| Metric | Description |
|--------|-------------|
| `nv_inference_request_success` | Total successful inference requests |
| `nv_inference_request_failure` | Total failed requests |
| `nv_inference_count` | Total inferences (requests × batch size) |
| `nv_inference_queue_duration_us` | Time spent in the queue (batching delay) |
| `nv_inference_compute_infer_duration_us` | Actual compute time |
| `nv_gpu_utilization` | GPU utilisation during inference |

## 8 — Model Instance Groups

Control how many model instances run on each GPU:

```bash
cat > model_repository/simple_model/config.pbtxt << 'EOF'
name: "simple_model"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [ { name: "INPUT", data_type: TYPE_FP32, dims: [ 16 ] } ]
output [ { name: "OUTPUT", data_type: TYPE_FP32, dims: [ 8 ] } ]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
EOF
```

## Cleanup

```bash
docker stop triton && docker rm triton
rm -rf model_repository
```
