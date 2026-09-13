import argparse
import datetime
import json
import itertools
import os
import re
import subprocess
import time
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.logger import init_logger
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.telemetry import GpuTelemetryRecorder
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile

logger = init_logger(__name__)

MLP_TIMER_NAMES = [
    "emb",
    "input_layernorm",
    "attn_pre_proj",
    "attn_rope",
    "attn_post_proj",
    "mlp_up_proj",
    "mlp_act",
    "mlp_down_proj",
    "add",
]
MLP_TIMER_STATS = ["min", "max", "mean", "median", "std"]
MLP_BASE_COLUMNS = [
    "n_head",
    "n_kv_head",
    "n_embd",
    "n_expanded_embd",
    "vocab_size",
    "use_gated_mlp",
    "num_tokens",
    "num_tensor_parallel_workers",
]


def _read_amd_metric_payload() -> dict:
    output = subprocess.run(
        [
            "amd-smi",
            "metric",
            "--usage",
            "--mem-usage",
            "--power",
            "--temperature",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return json.loads(output)


def _safe_number(value: Any) -> Any:
    if isinstance(value, dict):
        value = value.get("value", value)
    if isinstance(value, (int, float)):
        return value
    return None


def _parse_amd_clock_snapshot() -> dict:
    output = subprocess.run(
        ["amd-smi", "static"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    sys_match = re.search(r"SYS:\s*\n\s*CURRENT_LEVEL:.*\n\s*CURRENT_FREQUENCY:\s*(\d+)MHz", output)
    mem_match = re.search(r"MEM:\s*\n\s*CURRENT_LEVEL:.*\n\s*CURRENT_FREQUENCY:\s*(\d+)MHz", output)
    return {
        "sys_clock_mhz": int(sys_match.group(1)) if sys_match else None,
        "mem_clock_mhz": int(mem_match.group(1)) if mem_match else None,
    }


def collect_iteration_runtime_snapshot(gpu_vendor: str) -> dict:
    if gpu_vendor != "amd":
        return {}
    try:
        payload = _read_amd_metric_payload()
        gpu_rows = payload.get("gpu_data", [])
        row = gpu_rows[0] if gpu_rows else {}

        mem_usage = row.get("mem_usage", {}) if isinstance(row, dict) else {}
        temperature = row.get("temperature", {}) if isinstance(row, dict) else {}
        power = row.get("power", {}) if isinstance(row, dict) else {}
        clocks = _parse_amd_clock_snapshot()

        return {
            "gpu_utilization_percent": _safe_number(row.get("usage")),
            "memory_used_mb": _safe_number(mem_usage.get("used_vram")),
            "memory_total_mb": _safe_number(mem_usage.get("total_vram")),
            "temperature_edge_c": _safe_number(temperature.get("edge")),
            "socket_power_w": _safe_number(power.get("socket_power")),
            **clocks,
        }
    except Exception as exc:
        return {"telemetry_error": str(exc)}


def _get_triton_cache_size_bytes() -> int:
    cache_root = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    if not os.path.exists(cache_root):
        return 0

    total_size = 0
    for root, _, files in os.walk(cache_root):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                continue
    return total_size


def _write_jsonl_record(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def normalize_mlp_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.json_normalize(df["time_stats"]).add_prefix("time_stats.").join(
        df.drop(columns=["time_stats"])
    )

    ordered_columns = []
    for timer_name in MLP_TIMER_NAMES:
        for stat_name in MLP_TIMER_STATS:
            column_name = f"time_stats.{timer_name}.{stat_name}"
            if column_name not in df.columns:
                df[column_name] = float("nan")
            ordered_columns.append(column_name)

    for base_column in MLP_BASE_COLUMNS:
        if base_column not in df.columns:
            df[base_column] = float("nan")
        ordered_columns.append(base_column)

    return df[ordered_columns]


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_vendor",
        default="auto",
        choices=["auto", "nvidia", "amd"],
        help="GPU telemetry backend. auto selects amd-smi or nvidia-smi from PATH.",
    )
    parser.add_argument(
        "--model_executor_backend",
        default="auto",
        choices=["auto", "sarathi", "vllm_rocm", "torch"],
        help="Model executor backend. auto preserves current vendor-based selection.",
    )
    args = parser.parse_args()

    args.output_dir = (
        f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tokens_to_profile: List[int],
    pbar: Any,
    iteration_log_path: str,
):
    model_config = ModelConfig.from_model_name(model)

    all_results = []
    unsupported = []

    if args.disable_ray:
        def _create_model_wrapper(rank: int):
            return MlpWrapper(
                model_config,
                num_tensor_parallel_workers,
                args.profile_method,
                rank,
                args.output_dir,
                args.gpu_vendor,
                args.model_executor_backend,
            )
    else:
        model_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            MlpWrapper,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

        def _create_model_wrapper(rank: int):
            return model_wrapper_actor.remote(
                model_config,
                num_tensor_parallel_workers,
                args.profile_method,
                rank,
                args.output_dir,
                args.gpu_vendor,
                args.model_executor_backend,
            )

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        model_wrappers = [_create_model_wrapper(rank) for rank in range(args.num_gpus)]
        for token_index, num_tokens in enumerate(num_tokens_to_profile):
            worker_id = token_index % args.num_gpus
            pending_input = {
                "model": model,
                "num_tensor_parallel_workers": num_tensor_parallel_workers,
                "num_tokens": num_tokens,
            }

            try:
                start_time = time.perf_counter()
                if args.disable_ray:
                    result = model_wrappers[worker_id].profile(num_tokens)
                else:
                    result = ray.get(model_wrappers[worker_id].profile.remote(num_tokens))
                wall_clock_ms = (time.perf_counter() - start_time) * 1e3
                all_results.append(result)
                status = "ok"
                error_message = None
            except Exception as exc:
                wall_clock_ms = (time.perf_counter() - start_time) * 1e3
                status = "unsupported"
                error_message = str(exc)
                unsupported.append(
                    {
                        **pending_input,
                        "error": error_message,
                    }
                )
                model_wrappers[worker_id] = _create_model_wrapper(worker_id)

            latest_result = all_results[-1] if (status == "ok" and all_results) else {}
            telemetry_snapshot = collect_iteration_runtime_snapshot(args.gpu_vendor)
            _write_jsonl_record(
                iteration_log_path,
                {
                    "iteration_index": token_index,
                    **pending_input,
                    "status": status,
                    "error": error_message,
                    "wall_clock_ms": wall_clock_ms,
                    "compile_warmup_wall_ms": latest_result.get("compile_warmup_wall_ms"),
                    "warmup_wall_ms": latest_result.get("warmup_wall_ms"),
                    "measured_kernel_wall_ms": latest_result.get("measured_kernel_wall_ms"),
                    "measured_kernel_mean_ms": latest_result.get("measured_kernel_mean_ms"),
                    "active_steps": latest_result.get("active_steps"),
                    "triton_cache_dir": os.environ.get(
                        "TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")
                    ),
                    "triton_cache_size_bytes": _get_triton_cache_size_bytes(),
                    **telemetry_snapshot,
                },
            )

            pbar.update(1)

    if not all_results:
        return pd.DataFrame(), unsupported

    df = pd.DataFrame(all_results)
    return normalize_mlp_results_df(df), unsupported


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    telemetry_recorder = None
    try:
        telemetry_recorder = GpuTelemetryRecorder(args.output_dir, args.gpu_vendor)
    except Exception as exc:
        logger.warning("GPU telemetry disabled: %s", exc)
        telemetry_recorder = None

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))
    iteration_log_path = f"{args.output_dir}/iteration_metrics.jsonl"

    for model in args.models:
        result_df, unsupported = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
            iteration_log_path,
        )
        # model name would contain '/', so create a directory as required
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)
        with open(f"{args.output_dir}/{model}/unsupported.json", "w") as f:
            json.dump(unsupported, f, indent=2)
        if telemetry_recorder:
            telemetry_recorder.capture(
                context={
                    "profiler": "mlp",
                    "model": model,
                }
            )


if __name__ == "__main__":
    main()
