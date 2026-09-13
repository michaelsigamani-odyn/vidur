import argparse
import datetime
import json
import os
import re
import subprocess
import time
from math import ceil
from typing import Any, List

import pandas as pd
import ray
import torch
from tqdm import tqdm

from vidur.logger import init_logger
from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.model_executor_backend import create_model_executor_backend
from vidur.profiling.telemetry import GpuTelemetryRecorder
from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks

logger = init_logger(__name__)

ATTENTION_TIMER_NAMES = [
    "attn_input_reshape",
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
    "attn_output_reshape",
]
ATTENTION_TIMER_STATS = ["min", "max", "mean", "median", "std"]


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


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Profiling")
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
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum context length model can serve",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum context length of input",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum decode batch size",
    )
    parser.add_argument(
        "--profile_only_decode",
        action="store_true",
        help="Only profile the decode",
    )
    parser.add_argument(
        "--profile_only_prefill",
        action="store_true",
        help="Only profile the prefill",
    )
    parser.add_argument(
        "--attention_backend",
        default="flashinfer",
        help="The attention backend to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",
    )
    parser.add_argument(
        "--gpu_vendor",
        default="auto",
        choices=["auto", "nvidia", "amd"],
        help="GPU telemetry backend. auto selects amd-smi or nvidia-smi from PATH.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional global cap on number of sweep points to run. 0 disables the cap.",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/attention/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    input_combinations: List[AttentionInput],
    max_num_blocks: int,
    dtype: torch.dtype,
    gpu_vendor: str,
    pbar: Any,
    iteration_log_path: str,
    max_points_to_run: int = 0,
):
    model_config = ModelConfig.from_model_name(model)
    model_executor_backend = create_model_executor_backend(gpu_vendor)
    parallel_config = model_executor_backend.create_parallel_config(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )

    all_results = []
    unsupported = []

    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        AttentionWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    model_wrappers = [
        model_wrapper_actor.remote(
            model_config,
            parallel_config,
            max_num_blocks,
            args.max_model_len,
            args.block_size,
            args.attention_backend,
            dtype,
            gpu_vendor,
        )
        for _ in range(args.num_gpus)
    ]

    def _create_wrapper(worker_rank: int):
        return model_wrapper_actor.remote(
            model_config,
            parallel_config,
            max_num_blocks,
            args.max_model_len,
            args.block_size,
            args.attention_backend,
            dtype,
            gpu_vendor,
        )

    points_ran = 0

    for input_index, attention_input in enumerate(input_combinations):
        if max_points_to_run and points_ran >= max_points_to_run:
            break

        worker_id = input_index % args.num_gpus
        pending_input = {
            "model": model,
            "num_tensor_parallel_workers": num_tensor_parallel_workers,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
        }

        start_time = time.perf_counter()
        try:
            result = ray.get(model_wrappers[worker_id].profile.remote(attention_input))
            if result:
                all_results.append(result)
            status = "ok"
            error_message = None
        except Exception as exc:
            status = "unsupported"
            error_message = str(exc)
            unsupported.append(
                {
                    **pending_input,
                    "error": error_message,
                }
            )
            model_wrappers[worker_id] = _create_wrapper(worker_id)

        wall_clock_ms = (time.perf_counter() - start_time) * 1e3
        latest_result = all_results[-1] if (status == "ok" and all_results) else {}
        telemetry_snapshot = collect_iteration_runtime_snapshot(gpu_vendor)
        cache_size_bytes = _get_triton_cache_size_bytes()
        _write_jsonl_record(
            iteration_log_path,
            {
                "iteration_index": points_ran,
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
                "triton_cache_size_bytes": cache_size_bytes,
                **telemetry_snapshot,
            },
        )
        points_ran += 1

        pbar.update(1)

    if not all_results:
        return pd.DataFrame(), unsupported

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    for timer_name in ATTENTION_TIMER_NAMES:
        for stat_name in ATTENTION_TIMER_STATS:
            col = f"time_stats.{timer_name}.{stat_name}"
            if col not in df.columns:
                df[col] = 0.0
    return df, unsupported


def run_attention_correctness_check(
    model_config: ModelConfig,
    num_tensor_parallel_workers: int,
    block_size: int,
    attention_backend: str,
    gpu_vendor: str,
    dtype: torch.dtype,
) -> dict:
    from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy
    from vidur.profiling.model_executor_backend import create_model_executor_backend

    backend = create_model_executor_backend(gpu_vendor)
    backend.patch_cuda_timer(CudaTimer)
    TimerStatsStore(profile_method="kineto").clear_stats()
    parallel_config = backend.create_parallel_config(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )
    wrapper = backend.create_attention_wrapper(
        model_config,
        parallel_config,
        block_size,
        torch.device("cuda"),
        attention_backend,
    )

    num_q_heads = model_config.get_num_q_heads(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_dim = model_config.get_head_size()
    repeat_factor = num_q_heads // num_kv_heads

    atol = 3e-2 if dtype == torch.float16 else 2e-2
    rtol = 3e-2 if dtype == torch.float16 else 2e-2

    test_cases = [
        {
            "name": "prefill_q8",
            "is_prefill": True,
            "q_lengths": [8],
            "processed_lengths": [0],
        },
        {
            "name": "prefill_q4",
            "is_prefill": True,
            "q_lengths": [4],
            "processed_lengths": [0],
        },
        {
            "name": "decode_b4",
            "is_prefill": False,
            "q_lengths": [1, 1, 1, 1],
            "processed_lengths": [0, 0, 0, 0],
        },
    ]

    max_abs_diff = 0.0
    case_results = []

    for case in test_cases:
        q_lengths = case["q_lengths"]
        processed_lengths = case["processed_lengths"]
        total_q_tokens = sum(q_lengths)
        kv_lengths = [q_len + processed_len for q_len, processed_len in zip(q_lengths, processed_lengths)]
        max_num_blocks = max(8, max(ceil(kv_len / block_size) for kv_len in kv_lengths))

        q = torch.randn(total_q_tokens, num_q_heads * head_dim, device="cuda", dtype=dtype)
        k = torch.randn(total_q_tokens, num_kv_heads * head_dim, device="cuda", dtype=dtype)
        v = torch.randn(total_q_tokens, num_kv_heads * head_dim, device="cuda", dtype=dtype)

        seq_metadata_list = []
        next_block_index = 0
        for q_len, kv_len, processed_len in zip(q_lengths, kv_lengths, processed_lengths):
            num_blocks = ceil(kv_len / block_size)
            seq_metadata_list.append(
                SequenceMetadataProxy(
                    is_prompt=case["is_prefill"],
                    total_len=kv_len,
                    processed_len=processed_len,
                    block_table=list(range(next_block_index, next_block_index + num_blocks)),
                )
            )
            next_block_index += num_blocks

        kv_cache = wrapper.get_cache_block(
            max_num_blocks=max_num_blocks,
            dtype=dtype,
            device=torch.device("cuda"),
        )
        wrapper.begin_forward(seq_metadata_list)
        triton_out = wrapper.forward(q, k, v, kv_cache)
        wrapper.end_forward()

        q_3d = q.view(total_q_tokens, num_q_heads, head_dim)
        k_3d = k.view(total_q_tokens, num_kv_heads, head_dim)
        v_3d = v.view(total_q_tokens, num_kv_heads, head_dim)

        ref_outputs = []
        token_cursor = 0
        for q_len in q_lengths:
            q_seq = q_3d[token_cursor : token_cursor + q_len]
            k_seq = k_3d[token_cursor : token_cursor + q_len]
            v_seq = v_3d[token_cursor : token_cursor + q_len]
            token_cursor += q_len

            q_ref = q_seq.transpose(0, 1).unsqueeze(0)
            k_ref = k_seq.repeat_interleave(repeat_factor, dim=1).transpose(0, 1).unsqueeze(0)
            v_ref = v_seq.repeat_interleave(repeat_factor, dim=1).transpose(0, 1).unsqueeze(0)
            ref_seq = torch.nn.functional.scaled_dot_product_attention(
                q_ref,
                k_ref,
                v_ref,
                is_causal=True,
            )
            ref_outputs.append(
                ref_seq.squeeze(0).transpose(0, 1).reshape(q_len, num_q_heads * head_dim)
            )

        ref_out = torch.cat(ref_outputs, dim=0)
        case_max_abs = torch.max(torch.abs(triton_out - ref_out)).item()
        max_abs_diff = max(max_abs_diff, case_max_abs)
        case_results.append(
            {
                "name": case["name"],
                "max_abs_diff": case_max_abs,
                "total_q_tokens": total_q_tokens,
            }
        )
        if not torch.allclose(triton_out, ref_out, atol=atol, rtol=rtol):
            raise RuntimeError(
                "Attention correctness check failed: "
                f"case={case['name']}, max_abs_diff={case_max_abs:.6f}, atol={atol}, rtol={rtol}"
            )

    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": max_abs_diff,
        "cases": case_results,
    }


def main():
    args = parse_args()

    model_executor_backend = create_model_executor_backend(args.gpu_vendor)
    args.attention_backend = model_executor_backend.resolve_attention_backend(
        args.attention_backend
    )

    telemetry_recorder = None
    try:
        telemetry_recorder = GpuTelemetryRecorder(args.output_dir, args.gpu_vendor)
    except Exception as exc:
        logger.warning("GPU telemetry disabled: %s", exc)
        telemetry_recorder = None

    dtype = torch.float16
    input_combinations = get_attention_input_combinations(
        args.max_seq_len,
        args.min_batch_size,
        args.max_batch_size,
        args.profile_only_prefill,
        args.profile_only_decode,
    )

    total_combos = {}
    max_num_blocks_dict = {}
    remaining_points = args.max_points if args.max_points > 0 else None
    for model in args.models:
        model_config = ModelConfig.from_model_name(model)
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            parallel_config = model_executor_backend.create_parallel_config(
                tensor_parallel_size=num_tensor_parallel_workers,
                pipeline_parallel_size=1,
            )
            max_num_blocks = get_max_num_blocks(
                model_config,
                parallel_config,
                args.block_size,
                dtype,
            )
            max_num_blocks_dict[(model, num_tensor_parallel_workers)] = max_num_blocks
            combos = list(
                filter(
                    lambda input_combination: input_combination.is_under_memory_limit(
                        max_num_blocks * args.block_size
                    ),
                    input_combinations,
                )
            )

            if remaining_points is not None:
                if remaining_points <= 0:
                    combos = []
                elif len(combos) > remaining_points:
                    combos = combos[:remaining_points]
                    remaining_points = 0
                else:
                    remaining_points -= len(combos)

            total_combos[(model, num_tensor_parallel_workers)] = combos

    pbar = tqdm(total=sum(len(v) for v in total_combos.values()))
    iteration_log_path = f"{args.output_dir}/iteration_metrics.jsonl"

    for model in args.models:
        model_config = ModelConfig.from_model_name(model)
        correctness = run_attention_correctness_check(
            model_config=model_config,
            num_tensor_parallel_workers=args.num_tensor_parallel_workers[0],
            block_size=args.block_size,
            attention_backend=args.attention_backend,
            gpu_vendor=args.gpu_vendor,
            dtype=dtype,
        )
        logger.info(
            "Attention correctness passed for %s with atol=%s rtol=%s max_abs_diff=%.6f",
            model,
            correctness["atol"],
            correctness["rtol"],
            correctness["max_abs_diff"],
        )

        result_df = pd.DataFrame()
        unsupported = []
        points_done = 0
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            model_tp_combos = total_combos[(model, num_tensor_parallel_workers)]
            if not model_tp_combos:
                continue

            max_points_for_call = 0
            if args.max_points > 0:
                max_points_for_call = max(args.max_points - points_done, 0)

            worker_df, worker_unsupported = profile_model(
                args,
                model,
                num_tensor_parallel_workers,
                model_tp_combos,
                max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                dtype,
                args.gpu_vendor,
                pbar,
                iteration_log_path,
                max_points_for_call,
            )
            unsupported.extend(worker_unsupported)
            if not worker_df.empty:
                result_df = pd.concat([result_df, worker_df])
            points_done += len(model_tp_combos)

            if args.max_points > 0 and points_done >= args.max_points:
                break

        # model name would contain '/', so create a directory as required
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/attention.csv", index=False)
        with open(f"{args.output_dir}/{model}/unsupported.json", "w") as f:
            json.dump(unsupported, f, indent=2)
        if telemetry_recorder:
            telemetry_recorder.capture(
                context={
                    "profiler": "attention",
                    "model": model,
                }
            )


if __name__ == "__main__":
    main()
