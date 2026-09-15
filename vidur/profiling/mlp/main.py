import argparse
import datetime
import itertools
import json
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile


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
    args = parser.parse_args()

    args.output_dir = (
        f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)

    promises = []
    all_results = []
    unsupported = []
    overhead_records = []

    model_wrapper_actor = None
    if not args.disable_ray:
        model_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            MlpWrapper,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        if args.disable_ray:
            model_wrappers = [
                MlpWrapper(
                    model_config,
                    num_tensor_parallel_workers,
                    args.profile_method,
                    rank,
                    args.output_dir,
                )
                for rank in range(args.num_gpus)
            ]
        else:
            model_wrappers = [
                model_wrapper_actor.remote(
                    model_config,
                    num_tensor_parallel_workers,
                    args.profile_method,
                    rank,
                    args.output_dir,
                )
                for rank in range(args.num_gpus)
            ]
        for num_tokens in num_tokens_to_profile:
            worker_id = len(promises)
            if args.disable_ray:
                result = model_wrappers[worker_id].profile(num_tokens)
                if "error" in result:
                    unsupported.append(result)
                else:
                    all_results.append(result)
            else:
                promise = model_wrappers[worker_id].profile.remote(
                    num_tokens,
                )
                promises.append(promise)

            if not args.disable_ray and len(promises) >= args.num_gpus:
                results = ray.get(promises)
                for result in results:
                    if "error" in result:
                        unsupported.append(result)
                    else:
                        all_results.append(result)
                promises = []

            pbar.update(1)

    if not args.disable_ray:
        results = ray.get(promises)
        for result in results:
            if "error" in result:
                unsupported.append(result)
            else:
                all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
        df = (
            pd.json_normalize(df["time_stats"])
            .add_prefix("time_stats.")
            .join(df.drop(columns=["time_stats"]))
        )
    else:
        df = pd.DataFrame()

    return df, unsupported


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    triton_cache_dir = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    triton_cache_size_bytes_once = 0
    triton_cache_scan_wall_ms_once = 0.0
    if args.cache_size_scan_mode == "once":
        cache_scan_once_start = time.perf_counter()
        triton_cache_size_bytes_once = _get_triton_cache_size_bytes()
        triton_cache_scan_wall_ms_once = (time.perf_counter() - cache_scan_once_start) * 1e3

    telemetry_sampler = None
    if args.telemetry_mode == "background":
        try:
            telemetry_sampler = BackgroundGpuTelemetrySampler(
                output_dir=args.output_dir,
                gpu_vendor=args.gpu_vendor,
                interval_seconds=args.telemetry_interval_seconds,
            )
            telemetry_sampler.start()
        except Exception as exc:
            logger.warning("Background telemetry disabled: %s", exc)
            telemetry_sampler = None

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))

    for model in args.models:
        result_df, unsupported = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )
        # model name would contain '/', so create a directory as required
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)
        with open(f"{args.output_dir}/{model}/unsupported.json", "w", encoding="utf-8") as f:
            json.dump(unsupported, f, indent=2)


if __name__ == "__main__":
    main()
