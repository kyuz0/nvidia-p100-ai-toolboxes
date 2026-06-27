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

For the vLLM backend (powered by the community pascal-pkgs-ci fork):
```bash
toolbox create -c vllm-p100 \
  --image docker.io/kyuz0/nvidia-p100-ai-toolboxes:vllm
```

### 3. Enter the Toolbox
```bash
toolbox enter llama-p100-cuda
```
*Note: The toolboxes resolve common UID 1000 conflicts, meaning your host user ID and home directories will map seamlessly into the container.*

### 4. Run Inference

**With Llama.cpp:**
You can run `llama-server` or `llama-cli` directly since the binaries are located in `/usr/local/bin/`.

Example command using the first GPU:
```bash
llama-server -m ~/models/Llama-3-8B-Instruct.Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
```

**With vLLM:**
If you entered the `vllm-p100` toolbox, you can start the OpenAI-compatible vLLM server:

```bash
vllm serve ~/models/Llama-3-8B-Instruct-GPTQ --max-model-len 4096
```

## Performance & Optimization Tips

*   **VRAM Limits:** Each P100 has 16GB of HBM2 VRAM. Ensure your model and KV cache fit entirely within VRAM to avoid severe performance degradation. Q4_K_M and Q5_K_M quantizations generally work best.
*   **CUDA Graphs:** By default, `llama.cpp` utilizes CUDA graphs for batch size 1. This significantly improves inference speeds on older hardware like the P100.
*   **Unsupported Kernels:** Because `sm_60` lacks newer dot-product instructions (`__dp4a`), `llama.cpp` will fall back to software-based paths for certain advanced quantizations. Stick to standard k-quants for the most reliable performance.

## Build Automation
These toolboxes are automatically rebuilt every 4 hours via GitHub Actions if upstream `llama.cpp` has a new commit on its master branch.

## Benchmarks

### Llama.cpp Benchmarks (NGL=99, FA=1)
All tests run with `NGL=99` and `FA=1`. (32k = PP2048 @ d32768, TG32 @ d32768)

#### Prompt Processing (PP) Throughput
| Model | Size | GPUs | Backend | PP512 | PP(32k) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 4 | p100 | 168.44 ± 0.90 | 149.85 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 2 | p100 | 373.76 ± 1.07 | 333.21 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 4 | p100 | 412.61 ± 0.83 | 348.07 ± 0.00 |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 2 | p100 | 139.87 ± 0.31 | 144.20 ± 0.00 |
| Qwen3.6-27B-UD-Q8_K_XL | 32.89 GiB | 3 | p100 | 148.78 ± 0.15 | - |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 2 | p100 | 373.21 ± 0.55 | 341.89 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | 35.80 GiB | 3 | p100 | 358.40 ± 1.33 | 342.73 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 2 | p100 | 517.55 ± 3.13 | 406.61 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 2 | p100 | 552.72 ± 2.80 | 415.69 ± 0.00 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 1 | p100 | 950.46 ± 0.55 | 601.24 ± 0.00 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 1 | p100 | 748.22 ± 4.45 | 446.10 ± 0.00 |

#### Text Generation (TG) Throughput
| Model | Size | GPUs | Backend | TG128 | TG(32k) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B-Q3_K_M | 52.54 GiB | 4 | p100 | 21.61 ± 0.26 | 20.21 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q4_K_XL | 20.70 GiB | 2 | p100 | 53.78 ± 0.04 | 49.33 ± 0.00 |
| Qwen3.5-35B-A3B-UD-Q8_K_XL | 45.33 GiB | 4 | p100 | 36.19 ± 1.05 | 35.47 ± 0.00 |
| Qwen3.6-27B-UD-Q4_K_XL | 16.39 GiB | 2 | p100 | 11.57 ± 0.00 | 10.73 ± 0.00 |
| Qwen3.6-27B-UD-Q8_K_XL | 32.89 GiB | 3 | p100 | 10.08 ± 0.00 | - |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 20.81 GiB | 2 | p100 | 54.06 ± 0.04 | 49.44 ± 0.00 |
| Qwen3.6-35B-A3B-UD-Q8_K_XL | 35.80 GiB | 3 | p100 | 54.16 ± 0.03 | 49.54 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q4_K_XL | 15.90 GiB | 2 | p100 | 52.58 ± 0.01 | 46.74 ± 0.00 |
| gemma-4-26B-A4B-it-UD-Q8_K_XL | 25.94 GiB | 2 | p100 | 51.48 ± 0.01 | 45.64 ± 0.00 |
| gemma-4-E4B-it-UD-Q8_K_XL | 8.05 GiB | 1 | p100 | 45.71 ± 0.01 | 41.62 ± 0.00 |
| gpt-oss-20b-mxfp4 | 11.27 GiB | 1 | p100 | 68.64 ± 0.02 | 60.67 ± 0.00 |

### vLLM Throughput
| Model | TP | Requests | Total Tokens | Tokens/sec | Requests/sec | Elapsed (sec) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen_Qwen2.5-14B-Instruct | 4 | 500 | 363903 | 133.45 | 0.1834 | 2726.92 |
| Qwen_Qwen2.5-7B-Instruct | 2 | 500 | 363903 | 148.37 | 0.2039 | 2452.74 |
| Qwen_Qwen2.5-7B-Instruct | 4 | 500 | 363903 | 269.30 | 0.3700 | 1351.31 |
| meta-llama_Meta-Llama-3.1-8B-Instruct | 2 | 500 | 361361 | 143.79 | 0.1990 | 2513.13 |
| meta-llama_Meta-Llama-3.1-8B-Instruct | 4 | 500 | 361361 | 196.26 | 0.2716 | 1841.25 |

### GPU Performance Comparison
An interactive comparison dashboard of the Tesla P100 vs Radeon Instinct MI25 is available: [compare.html](file:///home/kyuz0/Documents/Projects/nvidia-p100-ai-toolboxes/benchmark/compare.html)
