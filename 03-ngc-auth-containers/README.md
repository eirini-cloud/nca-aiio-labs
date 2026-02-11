# Lab 03 — NGC Authentication & Pulling Containers

Set up the NGC CLI, authenticate, and pull GPU-optimised containers from the NVIDIA GPU Cloud registry.

## Objectives

- Install and configure the NGC CLI
- Authenticate with an NGC API key
- Browse and pull containers from `nvcr.io`
- Use Docker to pull NGC containers directly

## Prerequisites

- Free NGC account at <https://ngc.nvidia.com>
- Docker with the NVIDIA Container Toolkit
- Internet access

## 1 — Create an NGC API Key

1. Log in to <https://ngc.nvidia.com>
2. Click your profile icon (top right) → **Setup**
3. Click **Generate API Key**
4. Copy and save the key — you'll need it for CLI and Docker auth

## 2 — Install the NGC CLI

```bash
# Download the NGC CLI (AMD64 Linux)
wget -q https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.2/files/ngccli_linux.zip -O ngccli.zip

# Unzip and install
unzip -o ngccli.zip
chmod +x ngc-cli/ngc

# Move to a directory on your PATH
sudo mv ngc-cli/ngc /usr/local/bin/

# Verify installation
ngc --version

# Clean up
rm -rf ngccli.zip ngc-cli
```

## 3 — Configure NGC CLI

```bash
# Interactive configuration — enter your API key when prompted
ngc config set

# You'll be asked for:
#   API key:         <paste your key>
#   CLI output format: ascii (default)
#   org:             <your org, or leave blank for personal>
#   team:            <your team, or leave blank>
```

Alternatively, set via environment variable:

```bash
export NGC_API_KEY="your-api-key-here"
```

## 4 — Browse the NGC Catalog

```bash
# List available container collections
ngc registry image list

# Search for specific containers
ngc registry image list --format_type csv | grep -i pytorch
ngc registry image list --format_type csv | grep -i triton
ngc registry image list --format_type csv | grep -i tensorflow

# Get details on a specific image
ngc registry image info nvidia/pytorch:24.01-py3
```

## 5 — Pull Containers via NGC CLI

```bash
# Pull a container using the NGC CLI
ngc registry image pull nvcr.io/nvidia/pytorch:24.01-py3
```

## 6 — Pull Containers via Docker

The more common approach is to use Docker directly with NGC registry credentials.

```bash
# Log in to the NGC container registry
# Username is always "$oauthtoken", password is your NGC API key
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

# Pull containers
docker pull nvcr.io/nvidia/pytorch:24.01-py3
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
docker pull nvcr.io/nvidia/tensorflow:24.01-tf2-py3
docker pull nvcr.io/nvidia/cloud-native/dcgm:3.3.5-1-ubuntu22.04

# Verify pulled images
docker images | grep nvcr.io
```

## 7 — Run an NGC Container

```bash
# Run PyTorch container with GPU access
docker run --gpus all -it --rm \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run TensorFlow container with GPU access
docker run --gpus all -it --rm \
  nvcr.io/nvidia/tensorflow:24.01-tf2-py3 \
  python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(tf.config.list_physical_devices('GPU'))"
```

## 8 — NGC Container Environment Variables

NGC containers come with useful pre-set environment variables:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.01-py3 bash -c '
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"
echo "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
echo "NVIDIA_PRODUCT_NAME=$NVIDIA_PRODUCT_NAME"
echo "NVIDIA_PYTORCH_VERSION=$NVIDIA_PYTORCH_VERSION"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
'
```

## 9 — Pulling Models from NGC

NGC also hosts pre-trained models:

```bash
# List model collections
ngc registry model list

# Search for models
ngc registry model list --format_type csv | grep -i bert

# Download a model
ngc registry model download-version nvidia/nemo/megatron_gpt_345m:1
```

## Key Concepts

| Concept | Detail |
|---------|--------|
| Registry URL | `nvcr.io` |
| Username | Always `$oauthtoken` |
| Password | Your NGC API key |
| Naming convention | `nvcr.io/<org>/<container>:<tag>` |
| Free tier | Most NVIDIA containers are freely accessible with an NGC account |
| Container optimisation | NGC containers are optimised for NVIDIA GPUs with pre-tuned libraries (cuDNN, NCCL, TensorRT) |

## Cleanup

```bash
# Remove pulled images (optional)
docker rmi nvcr.io/nvidia/pytorch:24.01-py3
docker rmi nvcr.io/nvidia/tritonserver:24.01-py3
docker rmi nvcr.io/nvidia/tensorflow:24.01-tf2-py3

# Remove NGC CLI config
rm -rf ~/.ngc
```
