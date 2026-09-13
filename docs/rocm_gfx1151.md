# ROCm gfx1151 Validation Log

Note: existing `RadeonProW7900` SKU entries in Vidur are kept for dGPU profiling data and do not describe this machine (`AMD Radeon 8060S Graphics`, gfx1151).

## Phase 0 status

Phase 0 **passed** with the primary required image. No compatibility fallback was needed.

## Image pin and digest

Pulled image:

- `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`

Recorded digest:

- `rocm/vllm@sha256:b8a082f346d069376d35784250e38b23a043efe979408ae3a33d7c6b62ee3276`

Fallback image was **not used**:

- `rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`

## Container launch (exact required flags)

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE -v /home/michael/vidur:/workspace/vidur -w /workspace/vidur rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 bash -lc "..."
```

No `HSA_OVERRIDE_GFX_VERSION`, no `VLLM_USE_V1=0`, and no other GPU env overrides were set.

## Required validation outputs

### 1) `torch.cuda.get_arch_list()` contains `gfx1151`

Command:

```bash
python3 -c "import torch; print('arch_list=', torch.cuda.get_arch_list()); print('device_name=', torch.cuda.get_device_name(0))"
```

Output:

```text
arch_list= ['gfx1010', 'gfx1011', 'gfx1012', 'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103', 'gfx1200', 'gfx1201', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153', 'gfx908', 'gfx90a', 'gfx942', 'gfx950', 'gfx1250']
device_name= AMD Radeon 8060S Graphics
```

### 2) `torch.cuda.get_device_name()` reports Radeon 8060S

From the same output above:

- `device_name= AMD Radeon 8060S Graphics`

### 3) `import vllm` succeeds and `vllm.__version__ >= 0.23`

Command:

```bash
python3 -c "import vllm; print('vllm_version=', vllm.__version__)"
```

Output:

```text
vllm_version= 0.27.1.dev5+gf46a9dfe2.d20260827
```

### 4) `amd-smi static` reports `gfx1151`

Command:

```bash
amd-smi static
```

Relevant output:

```text
MARKET_NAME: AMD Radeon 8060S Graphics
TARGET_GRAPHICS_VERSION: gfx1151
```

## Smoke test

### Editable install from mounted Vidur repo

Command:

```bash
pip install -e .
```

Result:

```text
Successfully installed vidur-0.0.1
```

`pip install -e .` completed without pulling or building `sarathi-serve`.

### Single vLLM forward pass on ~1B Llama-class model

Model used:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Run script result:

```text
generation_text= !
```

Backend selection evidence from logs:

```text
Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER. Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.
```

This confirms the run used the Triton attention path on RDNA.

## Phase 2 status (MLP profiler)

Phase 2 code wiring is implemented, and the profiler now runs through a vLLM-backed AMD path with explicit unsupported-point logging.

### Backend abstraction changes for MLP

MLP code now resolves execution classes through `ModelExecutorBackend` and selects:

- `SarathiBackend` for NVIDIA path (unchanged behavior)
- `VllmRocmBackend` for AMD path

Key files:

- `vidur/profiling/model_executor_backend.py`
- `vidur/profiling/mlp/mlp_impl.py`
- `vidur/profiling/mlp/mlp_wrapper.py`
- `vidur/profiling/mlp/main.py`

### Layer mapping table (sarathi -> vLLM)

| Sarathi class/function | vLLM class/function | Notes |
| --- | --- | --- |
| `sarathi.model_executor.layers.activation.SiluAndMul` | `vllm.model_executor.layers.activation.SiluAndMul` | Used for gated MLP activation |
| `sarathi.model_executor.layers.layernorm.RMSNorm` | `vllm.model_executor.layers.layernorm.RMSNorm` | Uses vLLM custom op path under active vLLM config context |
| `sarathi.model_executor.layers.rotary_embedding.get_rope` | `vllm.model_executor.layers.rotary_embedding.get_rope` | Adapter maps legacy args (`base`, `rotary_dim`, `rope_scaling`) to `rope_parameters` |
| `sarathi...tensor_parallel.layers.ColumnParallelLinear` | `vllm.model_executor.layers.linear.ColumnParallelLinear` | Adapter sets `tp_size`/`tp_rank` and preserves profiler metric prefix |
| `sarathi...tensor_parallel.layers.RowParallelLinear` | `vllm.model_executor.layers.linear.RowParallelLinear` | Adapter preserves metric prefix; for `reduce_results=False`, bias is disabled to satisfy vLLM constructor constraint |
| `sarathi...tensor_parallel.layers.VocabParallelEmbedding` | `vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding` | Adapter preserves embedding layer role for timing |
| `initialize_dummy_weights` helper | backend-local parameter initialization | For vLLM path, all floating parameters are initialized with normal distribution |
| LM head linear | `vllm.model_executor.layers.linear.RowParallelLinear` | Mapping is available via backend adapter; not currently exercised by `GPTModel` in the MLP profiler |

### MLP run command used

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE -v /home/michael/vidur:/workspace/vidur -w /workspace/vidur rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 bash -lc "pip install -e . && python3 vidur/profiling/mlp/main.py --models microsoft/phi-2 --num_gpus 1 --gpu_vendor amd --output_dir profiling_outputs_amd"
```

### Output artifacts

Run directory:

- `profiling_outputs_amd/mlp/2026-09-13_03-21-10/`

Model output files:

- `profiling_outputs_amd/mlp/2026-09-13_03-21-10/microsoft/phi-2/mlp.csv`
- `profiling_outputs_amd/mlp/2026-09-13_03-21-10/microsoft/phi-2/unsupported.json`
- `profiling_outputs_amd/mlp/2026-09-13_03-21-10/gpu_telemetry.jsonl`

Observed coverage:

- `mlp.csv` rows: `196`
- token range with successful measurements: `1..2048`
- unsupported points logged: `65`

`mlp.csv` contains expected timing namespaces for this backend path, including:

- `time_stats.emb.*`
- `time_stats.input_layernorm.*`
- `time_stats.attn_pre_proj.*`
- `time_stats.attn_rope.*`
- `time_stats.attn_post_proj.*`
- `time_stats.mlp_up_proj.*`
- `time_stats.mlp_act.*`
- `time_stats.mlp_down_proj.*`
- `time_stats.add.*`

### `unsupported.json` contents summary

Unsupported entries are recorded instead of being skipped. For this run, unsupported points are concentrated at larger token counts (starting from `4096` and descending) where the worker process exits with ROCm runtime errors.

Representative error signature:

```text
HSA_STATUS_ERROR_EXCEPTION ... kernel: at::native::vectorized_gather_kernel<16, long>
```

Each unsupported record includes:

- `model`
- `num_tensor_parallel_workers`
- `num_tokens`
- `error`

## Phase 3 pause diagnostics (attention)

The long default attention sweep was paused. No profiler container is currently running (`docker ps` empty).

### What was running when paused

- Profiler: `attention`
- Command family: `python3 vidur/profiling/attention/main.py ... --gpu_vendor amd`

### Instrumentation added before resuming

Per-iteration diagnostics were added to the attention profiler:

- `wall_clock_ms` per sweep point
- `compile_warmup_wall_ms`, `warmup_wall_ms`, `measured_kernel_wall_ms`, `measured_kernel_mean_ms`
- runtime snapshot per point (`gpu_utilization_percent`, `memory_used_mb`, `memory_total_mb`, `temperature_edge_c`, `socket_power_w`, `sys_clock_mhz`, `mem_clock_mhz`)
- Triton/vLLM cache size tracking (`triton_cache_size_bytes`)
- global cap `--max_points` to run controlled subsets

Primary new log artifact:

- `profiling_outputs_amd/attention/2026-09-13_04-34-15/iteration_metrics.jsonl`

### 200-point controlled run

Run command:

```bash
python3 vidur/profiling/attention/main.py --models microsoft/phi-2 --num_gpus 1 --gpu_vendor amd --max_points 200 --output_dir profiling_outputs_amd
```

Artifacts:

- `profiling_outputs_amd/attention/2026-09-13_04-34-15/microsoft/phi-2/attention.csv`
- `profiling_outputs_amd/attention/2026-09-13_04-34-15/microsoft/phi-2/unsupported.json`
- `profiling_outputs_amd/attention/2026-09-13_04-34-15/iteration_metrics.jsonl`

Observed results for 200 points:

- points executed: `200`
- unsupported points: `0`
- wall-clock mean: `128.319 ms` (P95 `203.879 ms`)
- first-iteration compile+warmup: `2078.029 ms`
- aggregate split: compile `8.10%`, warmup `34.48%`, measured kernel `57.53%`
- Triton cache growth (`/root/.cache/vllm`): `0 bytes`

Flatness check on `s/it` trend (excluding first compile-heavy point):

- first 50 mean: `99.10 ms`
- middle 50 mean: `113.09 ms`
- last 50 mean: `113.14 ms`

Interpretation:

- compile time does **not** dominate this 200-point run; measured kernel time is the largest share.
- throughput stabilizes after startup; no immediate need to reorder sweep solely for compile amortization.

### AMD clock and memory telemetry check across 200 points

- VRAM used ranged from `20398 MB` to `20602 MB`.
- Edge temperature ranged from `42 C` to `46 C`.
- Socket power ranged from `55 W` to `78 W`.
- `gpu_utilization_percent` and clock fields (`sys_clock_mhz`, `mem_clock_mhz`) were unavailable (`null`) via `amd-smi` on this SKU/runtime command path during this run.

Because usable clock samples were unavailable, no thermal-throttling decision was inferred from clock-drop telemetry in this 200-point subset.
