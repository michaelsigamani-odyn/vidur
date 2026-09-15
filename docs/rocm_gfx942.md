# ROCm gfx942 Validation Log

## Phase 2 condition 1: overhead breakdown

Condition goal: break out profiling overhead into measurable buckets so we can decide whether run-time inflation is profiler/tooling cost or model-kernel cost.

Data source:

- `profiling_outputs_amd_baseline/mlp/2026-09-13_07-03-41/iteration_metrics.jsonl`
- `profiling_outputs_amd_baseline/mlp/2026-09-13_07-03-41/overhead_breakdown_summary.json`

Steady-state means (excluding iteration `0`, compile-heavy warmup):

- `wall_clock_ms_mean`: `322.849`
- `warmup_wall_ms_mean`: `120.207`
- `measured_kernel_wall_ms_mean`: `199.727`
- `profiler_enter_wall_ms_mean`: `0.403`
- `profiler_exit_wall_ms_mean`: `180.440`
- `profiler_start_stop_wall_ms_mean`: `180.843` (`56.01%` of wall clock)
- `telemetry_snapshot_wall_ms_mean`: `1.369` (`0.42%` of wall clock)
- `cache_size_scan_wall_ms_mean`: `0.021` (`0.0066%` of wall clock)
- `jsonl_write_wall_ms_mean`: `0.116` (from summary file)
- residual (`wall - warmup - measured_kernel`): `2.915 ms`

Interpretation:

- Inline telemetry and cache scanning are negligible at this scale.
- The dominant non-kernel cost is `record_function` profiler enter/exit overhead.
- Breakdown condition is met: overhead is isolated and quantified, with a clear dominant bucket.

## Phase 2 condition 2: A100 ratio check

Condition goal: ensure MI300X MLP timing shape is in a plausible range relative to committed A100 compute profile data, before moving to Phase 3.

Data source:

- MI300X run: `profiling_outputs/mlp/2026-09-13_06-44-40/microsoft/phi-2/mlp.csv`
- A100 reference: `data/profiling/compute/a100/microsoft/phi-2/mlp.csv`

Method:

- Align on shared `num_tokens` points.
- Compute a kernel proxy per row as the sum of all `time_stats.*.mean` columns.
- Compute ratio: `mi300x_kernel_proxy / a100_kernel_proxy`.

Results (`131` shared points):

- `mi300x_kernel_proxy_mean_ms`: `0.4489`
- `a100_kernel_proxy_mean_ms`: `0.4821`
- `ratio_mean`: `1.0097`
- `ratio_median`: `0.9962`
- `ratio_min`: `0.7416`
- `ratio_max`: `1.4504`

Interpretation:

- Mean and median are both ~`1.0x`, indicating near-parity on this reduced sweep.
- Extremes stay within the same order of magnitude and do not suggest broken timing scale.
- Ratio check condition is met; Phase 3 can proceed.

## Phase 3 status (attention) in this workspace

I attempted to continue directly with the reduced 200-point attention sweep, but this workspace does not currently have a usable `vllm` install, so the AMD attention path cannot initialize.

Attempted command:

```bash
PYTHONPATH=. .venv-rocm/bin/python vidur/profiling/attention/main.py --models microsoft/phi-2 --num_gpus 1 --gpu_vendor amd --max_points 200 --output_dir profiling_outputs --disable_ray
```

Observed blocker:

```text
ModuleNotFoundError: No module named 'vllm'
```

Additional attempt (torch backend) also does not currently unblock Phase 3 attention profiling in this tree:

```text
NotImplementedError
```

for `TorchRocmBackend.create_attention_wrapper(...)`.

Next runnable path for Phase 3 remains the ROCm vLLM container workflow where `vllm` is present, then rerun the command above (or the same command in-container) and capture artifacts under `profiling_outputs/attention/<timestamp>/`.
