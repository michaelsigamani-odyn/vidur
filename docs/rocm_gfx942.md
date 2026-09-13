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
