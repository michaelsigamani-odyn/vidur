# GPU Telemetry Backends

Vidur profiling entrypoints now support a pluggable GPU telemetry backend with vendor auto-detection.

## CLI

The following profiling commands accept `--gpu_vendor {auto,nvidia,amd}`:

- `python -m vidur.profiling.attention.main`
- `python -m vidur.profiling.mlp.main`
- `python -m vidur.profiling.collectives.main`

`auto` is the default and selects the backend from `PATH`:

1. If both `nvidia-smi` and `amd-smi` exist, Vidur uses `nvidia-smi` for compatibility.
2. If only one tool exists, that backend is used.
3. If neither exists, telemetry capture is skipped and profiling continues.

Telemetry snapshots are written to `gpu_telemetry.jsonl` in each profiler output directory.

## Normalized output schema

Each sample contains these normalized fields:

- `gpu_utilization_percent`
- `memory_used_mib`
- `memory_total_mib`
- `power_draw_watts`
- `temperature_celsius`
- `device_name`
- `vram_total_mib`

Missing metrics are returned as `null` (`None` in Python), and a warning is logged instead of failing the profiler.

## Backend mapping

### NVIDIA (`NvidiaSmiBackend`)

Command:

`nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits`

Field mapping:

- `gpu_utilization_percent` <- `utilization.gpu`
- `memory_used_mib` <- `memory.used`
- `memory_total_mib` <- `memory.total`
- `power_draw_watts` <- `power.draw`
- `temperature_celsius` <- `temperature.gpu`
- `device_name` <- `name`
- `vram_total_mib` <- `memory.total`

### AMD (`AmdSmiBackend`)

Commands:

- `amd-smi metric --usage --mem-usage --power --temperature --json`
- `amd-smi static --asic --vram --json`

Field mapping:

- `gpu_utilization_percent` <- `usage` (fallbacks include `gpu_usage`, `gpu_utilization`, `gfx_activity`)
- `memory_used_mib` <- `vram_used` (fallbacks include `vram_used_mb`, `vram_usage`, `used_vram`)
- `memory_total_mib` <- `vram_total` (fallbacks include `vram_total_mb`, `total_vram`)
- `power_draw_watts` <- `average_socket_power` (fallbacks include `average_power`, `power`, `current_socket_power`)
- `temperature_celsius` <- `temperature_edge` (fallbacks include `edge_temperature`, `temperature`, `temperature_junction`)
- `device_name` <- static `product_name` (fallbacks include `market_name`, `device_name`)
- `vram_total_mib` <- static `vram_total` (fallback `memory_total_mib`)

## Unit and semantic caveats

- AMD exposes multiple temperature sensors. Vidur prefers edge temperature (`temperature_edge`) because it is the closest match to NVIDIA's board-level `temperature.gpu`.
- AMD power fields can be average or instantaneous depending on SKU and firmware. Vidur prefers average socket power when available to reduce jitter.
- Any unsupported SKU-specific field is emitted as `null` and logged, so profiling remains non-fatal.

## Audit list: NVIDIA and CUDA-specific calls

This repository does not contain any `pynvml` usage.

Current telemetry/backend calls:

- `vidur/profiling/telemetry/backends.py`: `nvidia-smi --query-gpu=...` (collects utilization, memory, power, temperature, device identity)
- `vidur/profiling/telemetry/backends.py`: `amd-smi metric ... --json` (collects utilization, memory, power, temperature)
- `vidur/profiling/telemetry/backends.py`: `amd-smi static ... --json` (collects static device name and VRAM)

CUDA-specific calls and kernels outside telemetry backends:

- `vidur/profiling/utils/__init__.py`: `torch.cuda.mem_get_info()[1]` fallback through `torch.cuda.get_device_properties(...).total_memory` (total GPU memory for attention block sizing)
- `vidur/profiling/common/accelerator.py`: runtime device synchronization, memory queries, and profiler runtime selection (`torch.cuda.*`, `ProfilerActivity.CUDA`)
- `vidur/profiling/common/cuda_timer.py`: CUDA-event and Kineto profiling paths (`torch.cuda.Event`, `ProfilerActivity.CUDA`) for operator timing
- `vidur/profiling/utils/record_function_tracer.py`: Kineto GPU activity tracing with CUDA/HIP runtime category parsing from trace runtime events
- `vidur/profiling/mlp/mlp_wrapper.py`: model/input tensors allocated on accelerator device; synchronization around timed regions
- `vidur/profiling/attention/attention_wrapper.py`: attention input/cache tensors allocated on accelerator device; synchronization around measured forward loops
- `vidur/profiling/collectives/collectives_impl.py`: collective buffers and optional CUDA graph capture/replay for NCCL/RCCL collectives
- `vidur/profiling/collectives/collectives_wrapper.py`: synchronization/barrier around collective timing windows
- `vidur/profiling/collectives/benchmark_runner.py`: process-level device binding through `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`; distributed backend initialized as `nccl` (RCCL compatible on ROCm)

No hard-coded `nvidia-smi` calls exist outside `NvidiaSmiBackend`.
NVIDIA command parity is regression-tested with fixture-backed unit tests in `tests/profiling/telemetry/test_backends.py`.
