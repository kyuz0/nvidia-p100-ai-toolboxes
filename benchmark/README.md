# Benchmarks (NVIDIA Tesla P100)

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
