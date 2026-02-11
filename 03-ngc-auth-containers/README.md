# Lab 03 - NGC Authentication & Pulling Containers (Windows + Docker Desktop)

Authenticate with the NVIDIA GPU Cloud registry, pull GPU-optimised containers via Docker, and validate GPU access inside the container. Optionally install the NGC CLI.

## Objectives

* Authenticate Docker with `nvcr.io` using a secure login method
* Pull and run an NGC container with GPU access
* Validate CUDA + GPU detection inside the container
* (Optional) Install the NGC CLI and explore registry commands

## Prerequisites

* Free NGC account at <https://ngc.nvidia.com>
* **Docker Desktop running** with **Linux containers** (WSL2 backend)
* NVIDIA Container Toolkit working (`docker run --gpus all …` functions)
* Internet access

---

## 1 - Create an NGC API Key

1. Log in to <https://ngc.nvidia.com>
2. Click your profile icon (top right) → **Setup**
3. Click **Generate API Key**
4. Copy and save the key — you'll need it for Docker auth and optional CLI config

---

## 2 - Set API Key & Docker Login (secure)

PowerShell:

```powershell
$env:NGC_API_KEY = "<PASTE_YOUR_KEY>"
$env:NGC_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

**Expected**

* `Login Succeeded`

> Avoid `-p` because it can expose the key in CLI history.

---

## 3 - Pull an NGC Container

PowerShell:

```powershell
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

Verify:

```powershell
docker images | Select-String "nvcr.io/nvidia/pytorch"
```

**Expected**

* `Status: Image is up to date…` (or download progress on first pull)
* Image appears in `docker images` output

---

## 4 - Run Container + GPU Proof

PowerShell (single line):

```powershell
docker run --gpus all --rm nvcr.io/nvidia/pytorch:24.01-py3 python3 -c "import torch; print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**Expected output:**

```
cuda True
gpu NVIDIA GeForce RTX 4070
```

### Performance run (optional — avoids SHMEM warnings)

```powershell
docker run --gpus all --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:24.01-py3 python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 5 - NGC CLI (Optional)

### Windows

Install NGC CLI for Windows from the [NGC CLI download page](https://ngc.nvidia.com/setup/installers/cli) (or use WSL for the Linux instructions below).

Verify:

```powershell
ngc --version
ngc --help
```

### Linux/WSL

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

### Configure NGC CLI

```bash
# Interactive configuration — enter your API key when prompted
ngc config set

# You'll be asked for:
#   API key:         <paste your key>
#   CLI output format: ascii (default)
#   org:             <your org, or leave blank for personal>
#   team:            <your team, or leave blank>
```

---

## 6 - Browse the NGC Registry (CLI)

Check available commands:

```bash
ngc registry --help
```

After running `ngc config set`:

```bash
# List available container images
ngc registry image list

# Search for specific containers
ngc registry image list --format_type csv | grep -i pytorch
ngc registry image list --format_type csv | grep -i triton

# Get details on a specific image
ngc registry image info nvidia/pytorch:24.01-py3
```

> **Optional:** Use `ngc registry image pull nvcr.io/nvidia/pytorch:24.01-py3` after running `ngc config set`. In practice, most workflows use `docker pull` directly.

---

## 7 - NGC Container Environment Variables

NGC containers come with useful pre-set environment variables:

```powershell
docker run --gpus all --rm nvcr.io/nvidia/pytorch:24.01-py3 bash -c "echo NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES; echo CUDA_VERSION=$CUDA_VERSION; echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; echo NVIDIA_PRODUCT_NAME=$NVIDIA_PRODUCT_NAME; echo NVIDIA_PYTORCH_VERSION=$NVIDIA_PYTORCH_VERSION"
```

---

## Key Concepts

| Concept | Detail |
|---------|--------|
| Registry URL | `nvcr.io` |
| Username | Always `$oauthtoken` |
| Password | Your NGC API key |
| Naming convention | `nvcr.io/<org>/<container>:<tag>` |
| Free tier | Most NVIDIA containers are freely accessible with an NGC account |
| Container optimisation | NGC containers are optimised for NVIDIA GPUs with pre-tuned libraries (cuDNN, NCCL, TensorRT) |

---

## Evidence (add screenshots here)

<img width="962" height="547" alt="ngc-auth-help" src="https://github.com/user-attachments/assets/f45f7bf2-ffd9-446c-94cc-f4b8a5d45fa5" />



  * Shows `docker login … --password-stdin` → `Login Succeeded`
  * Shows `ngc --help` (CLI installed and command groups available)

<img width="960" height="786" alt="ngc-pull-run-gpu-proof" src="https://github.com/user-attachments/assets/6c51a75f-33f8-4848-9a12-a5fb41ca1abc" />

  * Shows `docker pull` success
  * Shows container run with `cuda True` and `RTX 4070` detected

---

## Cleanup

PowerShell:

```powershell
docker rmi nvcr.io/nvidia/pytorch:24.01-py3 2>$null

# Remove NGC CLI config (optional)
Remove-Item -Recurse -Force ~\.ngc 2>$null
```
