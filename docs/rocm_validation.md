# ROCm Profiling and Validation Playbook

This document defines a reproducible AMD Radeon workflow for Vidur profiling, predictor training, and accuracy validation against real runs.

## 1) Reproducible environment

Pin these versions in the PR description and keep the same versions across profiling and validation runs.

| Component | Version (pin) | How to record |
| --- | --- | --- |
| ROCm | `6.2.x` | `rocminfo | grep -m1 ROCm` |
| PyTorch ROCm | `2.4.x+rocm6.2` | `python -c "import torch; print(torch.__version__, torch.version.hip)"` |
| vLLM ROCm | commit SHA | `python -c "import vllm, inspect; print(getattr(vllm, '__version__', 'unknown'))"` and `git rev-parse HEAD` in vLLM repo |
| `amd-smi` | package version | `amd-smi --version` |
| Vidur | commit SHA | `git rev-parse HEAD` |

Recommended install notes:

- Use Linux with the official ROCm userspace and matching kernel driver.
- Ensure `amd-smi` is on `PATH` for telemetry auto-detection.
- Use the same Python environment for Vidur profiling and simulator predictor training.

## 2) Profiling on Radeon

Run all profiling pipelines with AMD telemetry enabled:

```bash
python vidur/profiling/mlp/main.py --models <model> --num_gpus 1 --gpu_vendor amd
python vidur/profiling/attention/main.py --models <model> --num_gpus 1 --gpu_vendor amd --attention_backend triton
python vidur/profiling/collectives/main.py --collective all_reduce --num_workers_per_node_combinations 1 --gpu_vendor amd
python vidur/profiling/collectives/main.py --collective send_recv --num_workers_per_node_combinations 1 --gpu_vendor amd
```

Notes:

- Vidur writes `gpu_telemetry.jsonl` beside each profiler CSV output.
- If `flashinfer` is selected on AMD, Vidur falls back to a ROCm-compatible backend when one is present.
- If a telemetry metric is unsupported on the SKU, Vidur emits `null` and logs a warning instead of failing.

## 3) Predictor training and simulator wiring

Copy profiling outputs into Vidur data layout:

- `data/profiling/compute/radeon_pro_w7900/<model>/mlp.csv`
- `data/profiling/compute/radeon_pro_w7900/<model>/attention.csv`
- `data/profiling/network/radeon_pro_w7900_single/all_reduce.csv`
- `data/profiling/network/radeon_pro_w7900_single/send_recv.csv`

Run simulator with the AMD device config:

```bash
python -m vidur.main \
  --replica_config_device radeon_pro_w7900 \
  --replica_config_network_device radeon_pro_w7900_single \
  --replica_config_model_name <model>
```

No schema changes are required for predictor inputs beyond the telemetry `--gpu_vendor` selection during profiling.

Radeon device config shipped in Vidur:

- device SKU: `radeon_pro_w7900`
- network SKU: `radeon_pro_w7900_single`
- fp16 compute: `123 TFLOPS`
- VRAM: `48 GB`
- memory bandwidth: `864 GB/s`

## 4) Ground-truth collection with vLLM-ROCm

Collect real serving results on the same Radeon host and same model for at least two traces/QPS points.

Minimum metrics to store per run:

- TTFT mean and P95
- TPOT/TBT mean and P95
- throughput (req/s or tok/s; use one definition consistently)

Store raw outputs under `docs/results/amd/` (CSV or JSON) and include command lines used.

NVIDIA baseline outputs should be stored under `docs/results/nvidia/` using the same parser and metric definitions.

## 5) Accuracy comparison methodology

Use the same workload traces, model, context limits, and SLO method as NVIDIA reporting.

Report these error metrics:

- Mean absolute percent error (MAPE) for TTFT, TPOT/TBT, throughput
- P95 absolute percent error for TTFT, TPOT/TBT, throughput

Suggested table format:

| Vendor | Device | Model | Trace | TTFT mean err | TTFT P95 err | TPOT mean err | TPOT P95 err | Throughput mean err | Throughput P95 err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA | A100/H100 | ... | ... | ... | ... | ... | ... | ... | ... |
| AMD | Radeon PRO W7900 | ... | ... | ... | ... | ... | ... | ... | ... |

Use `docs/results/accuracy_comparison.md` to publish final tables for both vendors in PR-ready form.

Target threshold guidance:

- Aim for single-digit percent latency error when possible, aligned with paper-era NVIDIA methodology.
- If AMD exceeds the bound, ship with a documented gap plus a follow-up issue containing root-cause hypotheses.

## 6) PR checklist

- Include environment versions from section 1.
- Include telemetry and CUDA/ROCm audit list (see `docs/gpu_telemetry.md`).
- Attach profiling completion logs for MLP, attention, and collectives on Radeon.
- Attach at least two trace-based real-run comparisons and Vidur error tables.
- Note known operator coverage gaps for the selected attention backend.
