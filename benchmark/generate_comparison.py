#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path

# Helper parsers
def parse_llama_log(filepath):
    results = {}
    size = "-"
    gpu_count = "-"
    
    if not filepath.exists():
        return size, gpu_count, results
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Find GPU count
    gpu_match = re.search(r"found (\d+) (?:CUDA|ROCm) devices", content)
    if gpu_match:
        gpu_count = gpu_match.group(1)
        
    lines = content.splitlines()
    header = []
    for line in lines:
        if "| model" in line:
            header = [p.strip() for p in line.split("|")][1:-1]
        elif line.startswith("|") and "---" not in line and header:
            parts = [p.strip() for p in line.split("|")][1:-1]
            if len(parts) == len(header):
                row = dict(zip(header, parts))
                test_name = row.get('test', '')
                tps = row.get('t/s', '')
                if test_name and tps:
                    results[test_name] = tps
                if 'size' in row and row['size']:
                    size = row['size']
                    
    return size, gpu_count, results

def get_gpu_count_from_suffix(suffix):
    mapping = {
        "single": "1",
        "dual": "2",
        "triple": "3",
        "quad": "4"
    }
    return mapping.get(suffix, "-")

def clean_model_name(name):
    name = re.sub(r"-0000\d-of-0000\d", "", name)
    return name

def get_llama_data(results_dir, filter_rocm=None):
    data = {}
    if not results_dir.exists():
        return data
        
    for filepath in results_dir.glob("*.log"):
        fname = filepath.name
        parts = fname.replace(".log", "").split("__")
        if len(parts) < 3:
            continue
            
        model = clean_model_name(parts[0])
        backend = parts[1]
        
        # If we want to filter for a specific ROCm version, check if backend matches
        if filter_rocm and filter_rocm not in backend:
            continue
            
        gpu_suffix = parts[-1]
        size, parsed_gpu, results = parse_llama_log(filepath)
        gpu_count = parsed_gpu if parsed_gpu != "-" else get_gpu_count_from_suffix(gpu_suffix)
        
        key = model
        if key not in data:
            data[key] = {
                "size": "-",
                "gpu_count": gpu_count,
                "pp512": "-",
                "tg128": "-",
                "pp32k": "-",
                "tg32k": "-"
            }
            
        if size != "-":
            data[key]["size"] = size
        if gpu_count != "-":
            data[key]["gpu_count"] = gpu_count
            
        for test, ts in results.items():
            # Extract float value from string like "373.76 ± 1.07"
            try:
                val = float(ts.split("±")[0].strip())
            except:
                val = "-"
                
            if "pp512" in test:
                data[key]["pp512"] = val
            elif "tg128" in test:
                data[key]["tg128"] = val
            elif "pp2048" in test:
                data[key]["pp32k"] = val
            elif "tg32" in test:
                data[key]["tg32k"] = val
                
    return data

def get_vllm_data(results_dir):
    data = {}
    if not results_dir.exists():
        return data
        
    for filepath in results_dir.glob("*.json"):
        # e.g., Qwen_Qwen2.5-7B-Instruct_tp2_throughput.json
        name_parts = filepath.stem.split("_tp")
        model = name_parts[0]
        # Remove cyankiwi or similar prefix to match
        model_clean = model.split("/")[-1].split("_")[-1]
        
        tp = name_parts[1].split("_")[0] if len(name_parts) > 1 else "1"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        tps = content.get("tokens_per_second", 0)
        rps = content.get("requests_per_second", 0)
        
        key = (model, tp)
        data[key] = {
            "tokens_per_second": tps,
            "requests_per_second": rps,
            "elapsed_time": content.get("elapsed_time", 0),
            "num_requests": content.get("num_requests", 0),
            "total_num_tokens": content.get("total_num_tokens", 0)
        }
    return data

def build_comparison():
    p100_bench_dir = Path("/home/kyuz0/Documents/Projects/nvidia-p100-ai-toolboxes/benchmark")
    mi25_bench_dir = Path("/home/kyuz0/Documents/Projects/ML-gfx900")
    
    # Load llama.cpp
    p100_llama = get_llama_data(p100_bench_dir / "results_llamacpp")
    mi25_llama = get_llama_data(mi25_bench_dir / "llama.cpp" / "benchmark" / "results", filter_rocm="rocm7_2_1")
    
    # Load vllm
    p100_vllm = get_vllm_data(p100_bench_dir / "results_vllm")
    mi25_vllm = get_vllm_data(mi25_bench_dir / "vllm" / "benchmark_results")
    
    # Merge Llama.cpp Data
    llama_comparison = []
    all_llama_models = sorted(list(set(p100_llama.keys()).union(set(mi25_llama.keys()))))
    
    for model in all_llama_models:
        p100_info = p100_llama.get(model, {})
        mi25_info = mi25_llama.get(model, {})
        
        size = p100_info.get("size", mi25_info.get("size", "-"))
        gpus = p100_info.get("gpu_count", mi25_info.get("gpu_count", "-"))
        
        llama_comparison.append({
            "model": model,
            "size": size,
            "gpus": gpus,
            "p100": {
                "pp512": p100_info.get("pp512", "-"),
                "tg128": p100_info.get("tg128", "-"),
                "pp32k": p100_info.get("pp32k", "-"),
                "tg32k": p100_info.get("tg32k", "-")
            },
            "mi25": {
                "pp512": mi25_info.get("pp512", "-"),
                "tg128": mi25_info.get("tg128", "-"),
                "pp32k": mi25_info.get("pp32k", "-"),
                "tg32k": mi25_info.get("tg32k", "-")
            }
        })
        
    # Merge vLLM Data
    # For vllm, let's match by comparing Llama-3.1-8B-Instruct or any overlap models
    vllm_comparison = []
    
    # Let's list all keys
    all_p100_vllm_keys = list(p100_vllm.keys())
    all_mi25_vllm_keys = list(mi25_vllm.keys())
    
    # Let's create an comparison list
    # We can match them based on model tags (e.g. "Meta-Llama-3.1-8B-Instruct" vs "Meta-Llama-3.1-8B-Instruct")
    for (p_model, p_tp), p_data in p100_vllm.items():
        # Find matching in mi25
        p_model_clean = p_model.split("/")[-1].split("_")[-1]
        
        matching_mi25_key = None
        for (m_model, m_tp) in mi25_vllm.keys():
            m_model_clean = m_model.split("/")[-1].split("_")[-1]
            if p_model_clean.lower() in m_model_clean.lower() and p_tp == m_tp:
                matching_mi25_key = (m_model, m_tp)
                break
                
        m_data = mi25_vllm.get(matching_mi25_key) if matching_mi25_key else None
        
        vllm_comparison.append({
            "model_p100": p_model,
            "model_mi25": matching_mi25_key[0] if matching_mi25_key else "-",
            "model_display": p_model_clean,
            "tp": p_tp,
            "p100": {
                "tokens_per_second": p_data["tokens_per_second"],
                "requests_per_second": p_data["requests_per_second"]
            },
            "mi25": {
                "tokens_per_second": m_data["tokens_per_second"] if m_data else "-",
                "requests_per_second": m_data["requests_per_second"] if m_data else "-"
            }
        })
        
    return llama_comparison, vllm_comparison

def main():
    p100_bench_dir = Path(__file__).parent
    llama_comp, vllm_comp = build_comparison()
    
    # Let's render the HTML page
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Comparison Dashboard: Tesla P100 vs Radeon Instinct MI25</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-p100: #10b981; /* Emerald/Green for NVIDIA */
            --accent-mi25: #ef4444; /* Red for AMD */
            --gradient-accent: linear-gradient(135deg, #3b82f6, #8b5cf6);
            --speedup-color: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            padding: 2rem 1.5rem;
            min-height: 100vh;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 2.5rem;
            text-align: center;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.15);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Tabs styling */
        .tabs {
            display: flex;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.3rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            font-weight: 500;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .tab-btn.active {
            background: var(--gradient-accent);
            color: #fff;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }

        /* Chart container */
        .chart-container {
            position: relative;
            height: 350px;
            width: 100%;
            margin-bottom: 2rem;
        }

        /* Table styling */
        .table-responsive {
            width: 100%;
            overflow-x: auto;
            border-radius: 8px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 1rem;
            border-bottom: 2px solid var(--border-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        td {
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.95rem;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.01);
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-align: center;
        }

        .badge-p100 {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-p100);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-mi25 {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-mi25);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .speedup-pos {
            color: #10b981;
            font-weight: bold;
        }

        .speedup-neg {
            color: #ef4444;
            font-weight: bold;
        }

        .speedup-neutral {
            color: var(--text-secondary);
        }

        .meta-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
        }

        .meta-tag {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 0.4rem 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .meta-tag strong {
            color: var(--text-primary);
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 2rem;
            }
            .card-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 1rem;
            }
            .tabs {
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Tesla P100 vs Radeon Instinct MI25</h1>
            <p>Interactive Performance Comparison Benchmark Dashboard</p>
        </header>

        <div class="meta-tags">
            <div class="meta-tag">⚡ NVIDIA GPU: <strong>Tesla P100 (16GB VRAM, sm_60, Pascal)</strong></div>
            <div class="meta-tag">🔴 AMD GPU: <strong>Radeon Instinct MI25 (16GB VRAM, gfx900, Vega)</strong></div>
            <div class="meta-tag">🛠️ Software: <strong>llama.cpp CUDA vs ROCm (rocm7.2.1) + vLLM</strong></div>
        </div>

        <div class="dashboard-grid">
            <!-- Llama.cpp Section -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🦙 Llama.cpp Inference Throughput</div>
                    <div class="tabs">
                        <button class="tab-btn active" onclick="switchLlamaTab('pp512')">PP 512</button>
                        <button class="tab-btn" onclick="switchLlamaTab('tg128')">TG 128</button>
                        <button class="tab-btn" onclick="switchLlamaTab('pp32k')">PP (32k)</button>
                        <button class="tab-btn" onclick="switchLlamaTab('tg32k')">TG (32k)</button>
                    </div>
                </div>

                <div class="chart-container">
                    <canvas id="llamaChart"></canvas>
                </div>

                <div class="table-responsive">
                    <table id="llamaTable">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Size</th>
                                <th>GPUs</th>
                                <th>Tesla P100 (t/s)</th>
                                <th>MI25 (t/s)</th>
                                <th>Ratio (MI25/P100)</th>
                            </tr>
                        </thead>
                        <tbody id="llamaTableBody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- vLLM Section -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🚀 vLLM Serving Throughput</div>
                </div>

                <div class="chart-container">
                    <canvas id="vllmChart"></canvas>
                </div>

                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>TP Size</th>
                                <th>Tesla P100 (Tokens/s)</th>
                                <th>MI25 (Tokens/s)</th>
                                <th>Ratio (MI25/P100)</th>
                            </tr>
                        </thead>
                        <tbody id="vllmTableBody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data injected from Python
        const llamaData = _LLAMA_DATA_INJECTION_;
        const vllmData = _VLLM_DATA_INJECTION_;

        // Active state
        let activeLlamaTab = 'pp512';
        let llamaChartInst = null;
        let vllmChartInst = null;

        // Render helper for Ratio
        function getRatioCell(p100Val, mi25Val) {
            if (p100Val === '-' || mi25Val === '-' || p100Val === 0) {
                return `<span class="speedup-neutral">-</span>`;
            }
            const ratio = mi25Val / p100Val;
            const percentage = ((ratio - 1) * 100).toFixed(1);
            if (ratio > 1) {
                return `<span class="speedup-pos">AMD +${percentage}% (${ratio.toFixed(2)}x)</span>`;
            } else if (ratio < 1) {
                const pMinus = (100 - ratio*100).toFixed(1);
                return `<span class="speedup-neg">NVIDIA +${pMinus}% (${(1/ratio).toFixed(2)}x)</span>`;
            } else {
                return `<span class="speedup-neutral">1.00x</span>`;
            }
        }

        function switchLlamaTab(tabId) {
            activeLlamaTab = tabId;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent.toLowerCase().includes(tabId.replace('32k', '(32k').replace('pp', 'pp ').replace('tg', 'tg ')));
            });
            // Update active styling correctly
            const btns = document.querySelectorAll('.tab-btn');
            btns[0].classList.toggle('active', tabId === 'pp512');
            btns[1].classList.toggle('active', tabId === 'tg128');
            btns[2].classList.toggle('active', tabId === 'pp32k');
            btns[3].classList.toggle('active', tabId === 'tg32k');

            updateLlamaDashboard();
        }

        function updateLlamaDashboard() {
            // Update Table
            const tbody = document.getElementById('llamaTableBody');
            tbody.innerHTML = '';

            const chartLabels = [];
            const p100Values = [];
            const mi25Values = [];

            llamaData.forEach(row => {
                const p100Val = row.p100[activeLlamaTab];
                const mi25Val = row.mi25[activeLlamaTab];

                // Append to table
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${row.model}</strong></td>
                    <td><span class="badge badge-p100">${row.size}</span></td>
                    <td>${row.gpus}</td>
                    <td>${p100Val}</td>
                    <td>${mi25Val}</td>
                    <td>${getRatioCell(p100Val, mi25Val)}</td>
                `;
                tbody.appendChild(tr);

                // Add to chart arrays if at least one value is valid
                if (p100Val !== '-' || mi25Val !== '-') {
                    chartLabels.push(row.model);
                    p100Values.push(p100Val === '-' ? 0 : p100Val);
                    mi25Values.push(mi25Val === '-' ? 0 : mi25Val);
                }
            });

            // Update Chart
            if (llamaChartInst) {
                llamaChartInst.destroy();
            }

            const ctx = document.getElementById('llamaChart').getContext('2d');
            llamaChartInst = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartLabels,
                    datasets: [
                        {
                            label: 'Tesla P100 (NVIDIA)',
                            data: p100Values,
                            backgroundColor: 'rgba(16, 185, 129, 0.75)',
                            borderColor: 'rgba(16, 185, 129, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        },
                        {
                            label: 'Radeon Instinct MI25 (AMD)',
                            data: mi25Values,
                            backgroundColor: 'rgba(239, 68, 68, 0.75)',
                            borderColor: 'rgba(239, 68, 68, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#9ca3af' },
                            title: { display: true, text: 'Tokens / Sec', color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af', font: { size: 10 } }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f3f4f6' } }
                    }
                }
            });
        }

        function buildVllmDashboard() {
            const tbody = document.getElementById('vllmTableBody');
            tbody.innerHTML = '';

            const chartLabels = [];
            const p100Values = [];
            const mi25Values = [];

            vllmData.forEach(row => {
                const p100Val = row.p100.tokens_per_second;
                const mi25Val = row.mi25.tokens_per_second;

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${row.model_display}</strong></td>
                    <td>${row.tp}</td>
                    <td>${p100Val.toFixed(2)}</td>
                    <td>${mi25Val !== '-' ? mi25Val.toFixed(2) : '-'}</td>
                    <td>${getRatioCell(p100Val, mi25Val)}</td>
                `;
                tbody.appendChild(tr);

                chartLabels.push(`${row.model_display} (TP ${row.tp})`);
                p100Values.push(p100Val);
                mi25Values.push(mi25Val === '-' ? 0 : mi25Val);
            });

            // Render Chart
            const ctx = document.getElementById('vllmChart').getContext('2d');
            vllmChartInst = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartLabels,
                    datasets: [
                        {
                            label: 'Tesla P100 (NVIDIA)',
                            data: p100Values,
                            backgroundColor: 'rgba(16, 185, 129, 0.75)',
                            borderColor: 'rgba(16, 185, 129, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        },
                        {
                            label: 'Radeon Instinct MI25 (AMD)',
                            data: mi25Values,
                            backgroundColor: 'rgba(239, 68, 68, 0.75)',
                            borderColor: 'rgba(239, 68, 68, 1)',
                            borderWidth: 1,
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#9ca3af' },
                            title: { display: true, text: 'Tokens / Sec', color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af', font: { size: 10 } }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f3f4f6' } }
                    }
                }
            });
        }

        // Init
        updateLlamaDashboard();
        buildVllmDashboard();
    </script>
</body>
</html>
"""
    
    # Inject data as JSON
    html_content = html_template.replace("_LLAMA_DATA_INJECTION_", json.dumps(llama_comp, indent=4))
    html_content = html_content.replace("_VLLM_DATA_INJECTION_", json.dumps(vllm_comp, indent=4))
    
    # Save output
    output_path = p100_bench_dir / "compare.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated comparison dashboard at {output_path}")

if __name__ == "__main__":
    main()
