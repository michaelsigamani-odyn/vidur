# Qwen3-TTS Multi-SKU Bring-Up (2026-09-15)

## Scope

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Targets: DGX1, DGX3 (NVIDIA) plus MI300X profiling carry-over status
- Branch: `epic-integration-mi300x-dgx`

## Captured Evidence

- `qwen3tts_dgx1.status.txt`: container `qwen3tts_dgx1` exited with code 1.
- `qwen3tts_dgx3.status.txt`: container `qwen3tts_dgx3` exited with code 1.
- `qwen3tts_dgx1.log` and `qwen3tts_dgx3.log`: identical startup trace and terminal failure (uploaded to GCS).
- `qwen3tts_dgx1_error_excerpt.txt` and `qwen3tts_dgx3_error_excerpt.txt`: committed failure excerpts.
- `mi300x_qwen2_status.txt`: MI300X log file exists and is being updated.
- `mi300x_qwen2_mlp_tail.log`: repeated actor creation failures in MLP profiler (uploaded to GCS).
- `mi300x_tp_size_error_excerpt.txt`: committed MI300X API-drift error excerpt.

## DGX Failure Signature

- `FileNotFoundError: Deploy config not found: vllm_omni/deploy/qwen3_tts.yaml`
- Observed after model weight download and omni engine init, on both DGX hosts.
- This is a deterministic runtime blocker for serving readiness on `/v1/models`.

## MI300X Failure Signature

- `TypeError: ColumnParallelLinear.__init__() got an unexpected keyword argument 'tp_size'`
- Error repeats across actor restarts in `MlpWrapper.__init__`.
- This indicates API drift between profiler backend expectations and current vLLM class signature.

## Current State

- NVIDIA bring-up retry:
  - Deploy-config path blocker was bypassed by launching from `/usr/local/lib/python3.12/dist-packages`.
  - DGX3 then failed at engine init with `AssertionError` in `SupportsMRoPE` path.
  - DGX1 stays `Up` with deep init logs, but readiness probe on `/v1/models` repeatedly resets connection.
- MI300X profiling: run remains blocked by `tp_size` incompatibility.
- No new successful latency/throughput metrics produced in this capture window.

## Code Patch Applied

- `vidur/profiling/model_executor_backend.py` now adapts linear-layer constructor kwargs to runtime vLLM signature via `inspect.signature`, avoiding hardcoded `tp_size`/`tp_rank`/`return_bias` assumptions.
