# DGX Spark (GB10) bring-up notes

## Phase 0 gate

- `uname -m`: `aarch64`
- `torch.__version__`: `2.14.0+cu130`
- `torch.version.cuda`: `13.0`
- `torch.cuda.get_device_capability()`: `(12, 1)`
- `nvidia-smi`:

```text
Sun Sep 13 05:22:18 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   43C    P8              4W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## Base environment decision

I used the native DGX Spark Python/CUDA stack instead of an NGC container. This host already has a working CUDA 13 toolchain (`nvcc` and `nvidia-smi`), and PyTorch CUDA wheels for `aarch64` install and run natively with device capability `(12, 1)`. Staying native avoids container runtime variables as another source of drift while validating GB10-specific build issues (ARM wheel availability and `sm_121` compilation) directly on the target machine.

## Installation log and build notes

### Vidur

```bash
python3 -m venv .venv-gb10
.venv-gb10/bin/python -m pip install --upgrade pip setuptools wheel
.venv-gb10/bin/pip install -r requirements.txt
.venv-gb10/bin/pip install -e .
```

Install reports:

- `docs/install_reports/vidur_requirements_report.json`
- `docs/install_reports/vidur_editable_report.json`

### Sarathi-serve (`vidur` branch)

```bash
git clone --branch vidur https://github.com/microsoft/sarathi-serve.git /home/michael/sarathi-serve
```

First build attempt failed before build edits with:

- `RuntimeError: Cannot find CUDA_HOME. CUDA must be available to build the package.`
- after setting `CUDA_HOME`, build failed because upstream `setup.py` hard-coded `-std=c++17` while PyTorch 2.14 headers require C++20.

Applied a minimal build-system patch (no kernel disablement, no fake capability flags):

- patch file: `docs/patches/sarathi-serve-aarch64-sm121-build.patch`
- change: `-std=c++17` -> `-std=c++20` for both host and nvcc compile flags in `setup.py`

Build/install command that succeeded:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 \
  .venv-gb10/bin/pip install --no-build-isolation --no-deps -e /home/michael/sarathi-serve
```

Install reports:

- `docs/install_reports/sarathi_editable_attempt1_report.json` (failed)
- `docs/install_reports/sarathi_editable_report_after_patch.json` (success)

### Attention backend dependency

Upstream requirement `flashinfer>=0.0.5` has no publishable wheel under that package name for this environment. The import used by Sarathi is the Python module `flashinfer`, provided by `flashinfer-python`.

```bash
.venv-gb10/bin/pip install flashinfer-python
```

Verified import:

```bash
.venv-gb10/bin/python -c "import flashinfer; print(flashinfer.__version__)"
```

Install report:

- `docs/install_reports/flashinfer_python_attempt_report.json`

### Required runtime deps for Sarathi imports

```bash
.venv-gb10/bin/pip install psutil "ray>=2.5.1" pyarrow sentencepiece \
  "transformers>=4.37.0" jupyterlab tiktoken grpcio uvicorn fastapi openai tqdm
```

Install reports:

- `docs/install_reports/sarathi_runtime_deps_report.json` (failed when trying nonexistent `flashinfer` package)
- `docs/install_reports/sarathi_runtime_deps_no_flashinfer_report.json` (success)

### Import checks

```bash
.venv-gb10/bin/python -c "import sarathi.model_executor; import sarathi.metrics; print('ok')"
.venv-gb10/bin/python -c "import flashinfer; from sarathi.model_executor.attention import AttentionBackend; print(flashinfer.__version__, AttentionBackend.FLASHINFER.value)"
```

## Telemetry smoke test (10s JSONL)

Added NVIDIA backend telemetry recorder:

- `vidur/profiling/telemetry/gpu_telemetry_recorder.py`

Smoke command:

```bash
.venv-gb10/bin/python -c "from datetime import datetime; from vidur.profiling.telemetry import GpuTelemetryRecorder; out=f'profiling_outputs/telemetry/gpu_telemetry_recorder_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl'; print(GpuTelemetryRecorder(gpu_vendor='nvidia', gpu_index=0).record(out, duration_s=10, interval_s=1.0))"
```

Output file:

- `profiling_outputs/telemetry/gpu_telemetry_recorder_2026-09-13_05-22-36.jsonl`

Each row contains utilization, memory used/total, power, and temperature.

## Phase 1: MLP profiling (completed)

Command used:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 \
  .venv-gb10/bin/python -m vidur.profiling.mlp.main \
  --disable_ray \
  --models microsoft/phi-2 meta-llama/Meta-Llama-3-8B \
  --num_gpus 1
```

Output directory:

- `profiling_outputs/mlp/2026-09-13_05-29-41`

Results summary:

- `microsoft/phi-2`: `261` rows in `mlp.csv`, `0` rows in `unsupported.json`
- `meta-llama/Meta-Llama-3-8B`: `1044` rows in `mlp.csv`, `0` rows in `unsupported.json`

Largest sweep point memory (`num_tokens=4096`):

- `microsoft/phi-2`: max `peak_memory_reserved_bytes=1105199104`, max `peak_memory_allocated_bytes=705092608`
- `meta-llama/Meta-Llama-3-8B`: max `peak_memory_reserved_bytes=4263510016`, max `peak_memory_allocated_bytes=2008104960`

No OOM points were observed in this MLP sweep.

## Phase 2: Attention profiling (backend status and sweep results)

Attempted command (default backend path):

```bash
PATH=/home/michael/vidur/.venv-gb10/bin:$PATH \
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 \
  .venv-gb10/bin/python -m vidur.profiling.attention.main \
  --disable_ray \
  --models microsoft/phi-2 meta-llama/Meta-Llama-3-8B \
  --num_gpus 1
```

Attempted output directory:

- `profiling_outputs/attention/2026-09-13_05-56-05`

Observed result:

- `attention.csv` is empty for both models
- `unsupported.json` has `43888` failures per model
- each failure is: `TypeError: append_paged_kv_cache() missing 1 required positional argument: 'kv_last_page_len'`

Backend actually used in all attention runs:

- `AttentionBackend.FLASHINFER`

Current status:

- the `append_paged_kv_cache` call-signature mismatch was resolved via a local shim change in Sarathi
- for `microsoft/phi-2`, all points still fail due to FlashInfer kernel limitation (`Unsupported head_dim: 80`)
- for `meta-llama/Meta-Llama-3-8B`, points complete successfully with the same backend

`microsoft/phi-2` unsupported reason (kept as unsupported, no workaround applied):

- `phi-2` uses attention `head_dim=80` (`n_embd=2560`, `n_q_head=32`)
- FlashInfer dispatch in this environment accepts compile-time head dims `{64, 128, 256, 512}` and raises on others (`flashinfer/data/include/flashinfer/utils.cuh`, `DISPATCH_HEAD_DIM`)
- every TP=1 attention point for `phi-2` therefore lands in `unsupported.json` with `Unsupported head_dim: 80`; this is a kernel support limit, not a profiling script bug

Latest reduced subset run (`--max_points 200`, TP=1):

- output dir: `profiling_outputs/attention/2026-09-13_06-59-23`
- `microsoft/phi-2`: `0` rows in `attention.csv`, `200` rows in `unsupported.json`
- `meta-llama/Meta-Llama-3-8B`: `0` rows in `attention.csv`, `0` rows in `unsupported.json` (no points selected after cap)

Latest full sweep run (TP=1, explicit output root):

- output dir: `profiling_outputs/full_attention_tp1/attention/2026-09-13_07-00-15`
- `microsoft/phi-2`: `0` rows in `attention.csv`, `10972` rows in `unsupported.json`
- `meta-llama/Meta-Llama-3-8B`: `10972` rows in `attention.csv`, `0` rows in `unsupported.json`

### Phase A NaN gate (attention target sparsity)

Root cause:

- this branch's FlashInfer wrapper scopes `CudaTimer` around both `ATTN_PREFILL` and `ATTN_DECODE` blocks even when that phase is not executed, so inactive-phase metrics were emitted as `0` instead of being absent.
- shipped A100 attention data stores inactive phase metrics as missing (`NaN`), and the predictor training path expects that sparsity pattern when it splits prefill and decode targets.

Fix (Vidur-side only; no new Sarathi change):

- in `vidur/profiling/attention/main.py`, after flattening `time_stats`, rows with `is_prefill=True` are forced to `NaN` for all `time_stats.attn_decode.{min,max,mean,median,std}` columns.
- rows with `is_prefill=False` are forced to `NaN` for all `time_stats.attn_prefill.{min,max,mean,median,std}` columns.

Validation rerun (mixed subset):

- command: `--models meta-llama/Meta-Llama-3-8B --num_tensor_parallel_workers 1 --max_points 200 --disable_ray`
- output: `profiling_outputs/phaseA_llama_subset_mixed/attention/2026-09-13_07-43-45`
- rows: `200` total (`100` prefill + `100` decode), `unsupported.json=0`
- NaN pattern now matches shipped schema directionally:
  - Spark subset: `attn_prefill.median` NaN on all decode rows, `attn_decode.median` NaN on all prefill rows.
  - A100 reference (`data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv`): same phase-exclusive NaN pattern.

Throwaway predictor ingestion/training check:

- pass when using this mixed attention subset with a dense MLP file (`data/profiling/compute/a100/meta-llama/Meta-Llama-3-8B/mlp.csv`): attention models `attn_kv_cache_save`, `attn_prefill`, and `attn_decode` train successfully.
- the previous `Input y contains NaN` failure reproduces only with the older local Spark MLP file from `profiling_outputs/mlp/2026-09-13_05-29-41/...`, which is a separate MLP-profiling issue.

### Qwen config notes

- Added `Qwen3_8BModelConfig` in `vidur/config/model_config.py` from HF config (`Qwen/Qwen3-8B`): 36 layers, 32 q heads, 8 kv heads, hidden 4096, MLP 12288, vocab 151936, rope theta 1e6, max position 40960, gated SiLU, RMSNorm, `use_qkv_bias=False`.
- Fidelity note: Vidur's layer model does not implement Qwen3's per-head Q/K RMSNorm.
- Qwen-72B QKV bias status: `GPTModel` path does honor `use_qkv_bias` in `CausalSelfAttention.qkv_proj` (`bias=config.use_bias or config.use_qkv_bias`).


## Web + source validation for blocker

I validated the API mismatch with both web sources and local source history:

1. FlashInfer API docs show `append_paged_kv_cache(append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len, ...)` (extra `batch_indices` and `positions` arguments compared to Sarathi call site):
   - <https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html>
2. Sarathi `vidur` branch call site passes the older positional shape:
   - `sarathi/model_executor/attention/flashinfer_attention_wrapper.py` lines 224-232.
3. `flashinfer` package name is not available on PyPI in this environment (404), while available package is `flashinfer-python`:
   - <https://pypi.org/project/flashinfer/>
   - <https://pypi.org/project/flashinfer-python/0.2.14.post1/>
4. FlashInfer git history check (including early commit in upstream repo) still shows the `batch_indices, positions` signature, confirming the current public package line does not match Sarathi's expected call convention.

## Additional attempted remediation (documented)

- Installed and tested `flashinfer-python==0.5.3` and `flashinfer-python==0.2.14.post1`.
- `0.5.3` installs and imports on `aarch64` but still exposes the same incompatible function signature.
- `0.2.14.post1` can be installed, but runtime JIT compilation fails on this stack due to C++ standard/toolchain mismatch in generated kernels (`-std=c++17` against PyTorch 2.14 C++20 headers).

Relevant install reports:

- `docs/install_reports/flashinfer_053_report.json`
- `docs/install_reports/flashinfer_0214_report.json`
- `docs/install_reports/flashinfer_0214_nodeps_report.json`
- `docs/install_reports/recover_torch_flashinfer_report.json`
- `docs/install_reports/flashinfer_restore_053_report.json`

Final currently active FlashInfer package in this env:

- `flashinfer-python==0.5.3`

## Transitive dependency source manifest

Source categories: `PyPI wheel` or `local editable (source tree)`.

```text
annotated-doc==0.0.5 | PyPI wheel
annotated-types==0.8.0 | PyPI wheel
anyio==4.15.1 | PyPI wheel
apache-tvm-ffi==0.1.13.post3 | PyPI wheel
argon2-cffi==25.1.0 | PyPI wheel
argon2-cffi-bindings==26.1.0 | PyPI wheel
arrow==1.4.0 | PyPI wheel
asttokens==3.0.2 | PyPI wheel
async-lru==2.3.0 | PyPI wheel
attrs==26.1.0 | PyPI wheel
babel==2.18.0 | PyPI wheel
beautifulsoup4==4.15.0 | PyPI wheel
bleach==6.4.0 | PyPI wheel
certifi==2026.7.22 | PyPI wheel
cffi==2.1.1 | PyPI wheel
charset-normalizer==3.5.1 | PyPI wheel
choreographer==1.3.0 | PyPI wheel
click==8.5.0 | PyPI wheel
cloudpickle==3.1.2 | PyPI wheel
comm==0.2.3 | PyPI wheel
contourpy==1.4.0 | PyPI wheel
cuda-bindings==13.4.1 | PyPI wheel
cuda-core==1.2.0 | PyPI wheel
cuda-pathfinder==1.8.1 | PyPI wheel
cuda-python==13.4.1 | PyPI wheel
cuda-tile==1.6.0 | PyPI wheel
cuda-toolkit==13.0.3.0 | PyPI wheel
cycler==0.12.1 | PyPI wheel
ddsketch==3.0.1 | PyPI wheel
debugpy==1.8.21 | PyPI wheel
defusedxml==0.7.1 | PyPI wheel
einops==0.8.2 | PyPI wheel
executing==2.2.1 | PyPI wheel
fastapi==0.141.1 | PyPI wheel
fasteners==0.20 | PyPI wheel
fastjsonschema==2.22.2 | PyPI wheel
filelock==3.32.6 | PyPI wheel
flashinfer-python==0.5.3 | PyPI wheel
fonttools==4.65.0 | PyPI wheel
formulaic==1.2.2 | PyPI wheel
fqdn==1.5.1 | PyPI wheel
fsspec==2026.7.0 | PyPI wheel
googleapis-common-protos==1.75.3 | PyPI wheel
grpcio==1.83.1 | PyPI wheel
h11==0.16.0 | PyPI wheel
hf-xet==1.6.0 | PyPI wheel
httpcore==1.0.9 | PyPI wheel
httpcore2==2.12.0 | PyPI wheel
httpx==0.28.1 | PyPI wheel
httpx2==2.12.0 | PyPI wheel
huggingface_hub==1.31.0 | PyPI wheel
idna==3.19 | PyPI wheel
interface_meta==2.0.1 | PyPI wheel
ipykernel==7.3.0 | PyPI wheel
ipython==9.17.1 | PyPI wheel
ipython_pygments_lexers==1.1.1 | PyPI wheel
isoduration==20.11.0 | PyPI wheel
jedi==0.20.0 | PyPI wheel
Jinja2==3.1.6 | PyPI wheel
jiter==0.17.0 | PyPI wheel
joblib==1.6.0 | PyPI wheel
json5==0.15.0 | PyPI wheel
jsonpointer==3.1.1 | PyPI wheel
jsonschema==4.26.0 | PyPI wheel
jsonschema-specifications==2025.9.1 | PyPI wheel
jupyter-events==0.12.1 | PyPI wheel
jupyter-lsp==2.3.1 | PyPI wheel
jupyter_builder==1.2.3 | PyPI wheel
jupyter_client==8.10.0 | PyPI wheel
jupyter_core==5.9.1 | PyPI wheel
jupyter_server==2.21.0 | PyPI wheel
jupyter_server_terminals==0.5.4 | PyPI wheel
jupyterlab==4.6.3 | PyPI wheel
jupyterlab_pygments==0.3.0 | PyPI wheel
jupyterlab_server==2.28.0 | PyPI wheel
kaleido==1.4.0 | PyPI wheel
kiwisolver==1.5.1 | PyPI wheel
lark==1.3.1 | PyPI wheel
logistro==2.0.1 | PyPI wheel
markdown-it-py==4.2.0 | PyPI wheel
MarkupSafe==3.0.3 | PyPI wheel
matplotlib==3.11.2 | PyPI wheel
matplotlib-inline==0.2.2 | PyPI wheel
mdurl==0.1.2 | PyPI wheel
mistune==3.3.4 | PyPI wheel
mpmath==1.3.0 | PyPI wheel
msgpack==1.2.2 | PyPI wheel
narwhals==2.26.0 | PyPI wheel
nbclient==0.11.0 | PyPI wheel
nbconvert==7.17.1 | PyPI wheel
nbformat==5.11.1 | PyPI wheel
nccl4py==0.5.0 | PyPI wheel
nest-asyncio2==1.7.2 | PyPI wheel
networkx==3.6.1 | PyPI wheel
ninja==1.13.2 | PyPI wheel
notebook_shim==0.2.4 | PyPI wheel
numpy==2.5.3 | PyPI wheel
nvidia-cublas==13.1.1.3 | PyPI wheel
nvidia-cuda-cupti==13.0.85 | PyPI wheel
nvidia-cuda-nvdisasm==13.4.49 | PyPI wheel
nvidia-cuda-nvrtc==13.0.88 | PyPI wheel
nvidia-cuda-runtime==13.0.96 | PyPI wheel
nvidia-cudnn-cu13==9.24.0.43 | PyPI wheel
nvidia-cudnn-frontend==1.28.0 | PyPI wheel
nvidia-cufft==12.0.0.61 | PyPI wheel
nvidia-cufile==1.15.1.6 | PyPI wheel
nvidia-curand==10.4.0.35 | PyPI wheel
nvidia-cusolver==12.0.4.66 | PyPI wheel
nvidia-cusparse==12.6.3.3 | PyPI wheel
nvidia-cusparselt-cu13==0.8.1 | PyPI wheel
nvidia-cutlass-dsl==4.8.0.dev0 | PyPI wheel
nvidia-cutlass-dsl-libs-base==4.8.0.dev0 | PyPI wheel
nvidia-cutlass-dsl-libs-core==4.8.0.dev0 | PyPI wheel
nvidia-cutlass-dsl-libs-cu12==4.8.0.dev0 | PyPI wheel
nvidia-ml-py==13.610.43 | PyPI wheel
nvidia-nccl-cu13==2.30.7 | PyPI wheel
nvidia-nvjitlink==13.4.52 | PyPI wheel
nvidia-nvshmem-cu13==3.4.5 | PyPI wheel
nvidia-nvtx==13.0.85 | PyPI wheel
openai==3.13.0 | PyPI wheel
opentelemetry-api==1.44.0 | PyPI wheel
opentelemetry-exporter-otlp-proto-common==1.44.0 | PyPI wheel
opentelemetry-exporter-otlp-proto-http==1.44.0 | PyPI wheel
opentelemetry-proto==1.44.0 | PyPI wheel
opentelemetry-sdk==1.44.0 | PyPI wheel
opentelemetry-semantic-conventions==0.65b0 | PyPI wheel
orjson==3.12.0 | PyPI wheel
packaging==26.3 | PyPI wheel
pandas==3.0.5 | PyPI wheel
pandocfilters==1.5.1 | PyPI wheel
parso==0.8.7 | PyPI wheel
patsy==1.0.3 | PyPI wheel
pexpect==4.9.0 | PyPI wheel
pillow==12.3.0 | PyPI wheel
pip==26.2.1 | PyPI wheel
platformdirs==4.11.8 | PyPI wheel
plotly==7.0.0 | PyPI wheel
plotly-express==0.4.1 | PyPI wheel
prometheus_client==0.26.0 | PyPI wheel
prompt_toolkit==3.0.53 | PyPI wheel
protobuf==7.36.1 | PyPI wheel
psutil==7.2.2 | PyPI wheel
ptyprocess==0.7.0 | PyPI wheel
pure_eval==0.2.4 | PyPI wheel
pyarrow==25.0.1 | PyPI wheel
pycparser==3.0 | PyPI wheel
pynvml==13.0.1 | PyPI wheel
pydantic==2.13.5 | PyPI wheel
pydantic_core==2.46.5 | PyPI wheel
Pygments==2.21.0 | PyPI wheel
pyparsing==3.3.2 | PyPI wheel
python-dateutil==2.9.0.post0 | PyPI wheel
python-json-logger==4.2.0 | PyPI wheel
PyYAML==6.0.3 | PyPI wheel
pyzmq==27.2.0 | PyPI wheel
ray==2.58.0 | PyPI wheel
referencing==0.37.0 | PyPI wheel
regex==2026.9.10 | PyPI wheel
requests==2.34.2 | PyPI wheel
rfc3339-validator==0.1.4 | PyPI wheel
rfc3986-validator==0.1.1 | PyPI wheel
rfc3987-syntax==1.1.0 | PyPI wheel
rich==15.0.0 | PyPI wheel
rpds-py==2026.6.3 | PyPI wheel
safetensors==0.8.0 | PyPI wheel
sarathi==0.1.7 | local editable (source tree)
scikit-learn==1.9.1 | PyPI wheel
scipy==1.18.1 | PyPI wheel
seaborn==0.13.2 | PyPI wheel
Send2Trash==2.1.0 | PyPI wheel
sentencepiece==0.2.2 | PyPI wheel
setuptools==84.0.0 | PyPI wheel
shellingham==1.5.4 | PyPI wheel
simplejson==4.1.2 | PyPI wheel
six==1.17.0 | PyPI wheel
sniffio==1.3.1 | PyPI wheel
soupsieve==2.9.2 | PyPI wheel
stack-data==0.6.3 | PyPI wheel
starlette==1.6.0 | PyPI wheel
statsmodels==0.15.0 | PyPI wheel
sympy==1.14.0 | PyPI wheel
tabulate==0.10.0 | PyPI wheel
terminado==0.18.1 | PyPI wheel
threadpoolctl==3.6.0 | PyPI wheel
tiktoken==0.14.0 | PyPI wheel
tinycss2==1.5.1 | PyPI wheel
tokenizers==0.23.2 | PyPI wheel
torch==2.14.0 | PyPI wheel
tornado==6.5.8 | PyPI wheel
tqdm==4.70.1 | PyPI wheel
traitlets==5.16.1 | PyPI wheel
transformers==5.17.0 | PyPI wheel
triton==3.8.0 | PyPI wheel
truststore==0.10.4 | PyPI wheel
typer==0.27.2 | PyPI wheel
typing-inspection==0.4.4 | PyPI wheel
typing_extensions==4.16.0 | PyPI wheel
tzdata==2026.4 | PyPI wheel
uri-template==1.3.0 | PyPI wheel
urllib3==2.7.0 | PyPI wheel
uvicorn==0.52.4 | PyPI wheel
vidur==0.0.1 | local editable (source tree)
wandb==0.30.0 | PyPI wheel
wcwidth==0.8.3 | PyPI wheel
webcolors==25.10.0 | PyPI wheel
webencodings==0.6.1 | PyPI wheel
websocket-client==1.9.2 | PyPI wheel
wheel==0.48.0 | PyPI wheel
wrapt==2.4.1 | PyPI wheel
xxhash==4.0.1 | PyPI wheel
```
