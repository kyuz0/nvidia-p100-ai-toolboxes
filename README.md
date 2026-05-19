# NVIDIA P100 (sm_60) AI Toolboxes

This repository provides automatically built Toolbox (`containertoolbx.org`) images with `llama.cpp` natively compiled and optimized for NVIDIA Tesla P100 GPUs (Pascal architecture, Compute Capability 6.0).

> [!NOTE]
> The P100 is an older architecture. To ensure optimal compatibility and avoid missing device code errors during execution, these builds explicitly compile `llama.cpp` using the `-DCMAKE_CUDA_ARCHITECTURES=60` flag.

## Getting Started

### 1. Prerequisites
You need a system with `nvidia-container-toolkit` installed and configured so that your container runtime (Docker or Podman) can access the GPUs. You also need `toolbox` installed.

If you don't have `nvidia-container-toolkit` installed (e.g., you get a `command not found` error), please follow the [official installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your distribution.

For Podman, once installed, make sure the Container Device Interface (CDI) is generated:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### 2. Create the Toolbox
Create a new toolbox utilizing the published image. Since Toolbox automatically handles GPU passthrough when `nvidia-container-toolkit` is present, no extra device flags are needed:

For the standard CUDA backend:
```bash
toolbox create -c llama-p100-cuda \
  --image docker.io/kyuz0/nvidia-p100-ai-toolboxes:latest
```

Alternatively, if you want to use the Vulkan backend (built on Fedora 43):
```bash
toolbox create -c llama-p100-vulkan \
  --image docker.io/kyuz0/nvidia-p100-ai-toolboxes:vulkan
```

### 3. Enter the Toolbox
```bash
toolbox enter llama-p100-cuda
```
*Note: The toolboxes resolve common UID 1000 conflicts, meaning your host user ID and home directories will map seamlessly into the container.*

### 4. Run Llama.cpp
You can run `llama-server` or `llama-cli` directly since the binaries are located in `/usr/local/bin/`.

Example command using the first GPU:
```bash
llama-server -m ~/models/Llama-3-8B-Instruct.Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

## Performance & Optimization Tips

*   **VRAM Limits:** Each P100 has 16GB of HBM2 VRAM. Ensure your model and KV cache fit entirely within VRAM to avoid severe performance degradation. Q4_K_M and Q5_K_M quantizations generally work best.
*   **CUDA Graphs:** By default, `llama.cpp` utilizes CUDA graphs for batch size 1. This significantly improves inference speeds on older hardware like the P100.
*   **Unsupported Kernels:** Because `sm_60` lacks newer dot-product instructions (`__dp4a`), `llama.cpp` will fall back to software-based paths for certain advanced quantizations. Stick to standard k-quants for the most reliable performance.

## Build Automation
These toolboxes are automatically rebuilt every 4 hours via GitHub Actions if upstream `llama.cpp` has a new commit on its master branch.
