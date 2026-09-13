import argparse
import datetime
import json
import os
from typing import Any, List

import pandas as pd
import ray
import torch
from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import AttentionBackend
from tqdm import tqdm

from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks


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
        default=AttentionBackend.FLASHINFER,
        choices=[e.value for e in AttentionBackend],
        help="The attention backend to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Maximum number of attention points to run across the full sweep",
    )
    args = parser.parse_args()

    args.output_dir = f"{args.output_dir}/attention/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def _select_mixed_points(input_combinations: List[AttentionInput], limit: int) -> List[AttentionInput]:
    if limit <= 0:
        return []

    prefill_points = [combo for combo in input_combinations if combo.is_prefill]
    decode_points = [combo for combo in input_combinations if not combo.is_prefill]

    selected = []
    prefill_index = 0
    decode_index = 0

    while len(selected) < limit and (
        prefill_index < len(prefill_points) or decode_index < len(decode_points)
    ):
        if prefill_index < len(prefill_points):
            selected.append(prefill_points[prefill_index])
            prefill_index += 1
            if len(selected) >= limit:
                break
        if decode_index < len(decode_points):
            selected.append(decode_points[decode_index])
            decode_index += 1

    return selected


def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    input_combinations: List[AttentionInput],
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
):
    model_config = ModelConfig.from_model_name(model)
    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )

    promises = []
    all_results = []
    unsupported = []

    if args.disable_ray:
        model_wrappers = [
            AttentionWrapper(
                model_config,
                parallel_config,
                max_num_blocks,
                args.max_model_len,
                args.block_size,
                args.attention_backend,
                dtype,
            )
            for _ in range(args.num_gpus)
        ]
    else:
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
            )
            for _ in range(args.num_gpus)
        ]

    for attention_input in input_combinations:
        worker_id = len(promises)
        if args.disable_ray:
            result = model_wrappers[worker_id].profile(attention_input)
            if result and "error" in result:
                unsupported.append(result)
            elif result:
                all_results.append(result)
        else:
            promise = model_wrappers[worker_id].profile.remote(attention_input)
            promises.append(promise)

        if not args.disable_ray and len(promises) >= args.num_gpus:
            results = ray.get(promises)
            for result in results:
                if result and "error" in result:
                    unsupported.append(result)
                elif result:
                    all_results.append(result)
            promises = []

        pbar.update(1)

    if not args.disable_ray:
        results = ray.get(promises)
        for result in results:
            if result and "error" in result:
                unsupported.append(result)
            elif result:
                all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
        df = (
            pd.json_normalize(df["time_stats"])
            .add_prefix("time_stats.")
            .join(df.drop(columns=["time_stats"]))
        )

        # Keep phase-specific metrics sparse to match shipped profiling schema:
        # prefill rows do not have decode timing targets and decode rows do not
        # have prefill timing targets.
        prefill_mask = df["is_prefill"] == True
        decode_mask = ~prefill_mask
        for stat_name in ("min", "max", "mean", "median", "std"):
            decode_col = f"time_stats.attn_decode.{stat_name}"
            prefill_col = f"time_stats.attn_prefill.{stat_name}"
            if decode_col in df.columns:
                df.loc[prefill_mask, decode_col] = float("nan")
            if prefill_col in df.columns:
                df.loc[decode_mask, prefill_col] = float("nan")
    else:
        df = pd.DataFrame()
    return df, unsupported


def main():
    args = parse_args()

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
    for model in args.models:
        model_config = ModelConfig.from_model_name(model)
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            max_num_blocks = get_max_num_blocks(
                model_config,
                ParallelConfig(
                    tensor_parallel_size=num_tensor_parallel_workers,
                    pipeline_parallel_size=1,
                ),
                args.block_size,
                dtype,
            )
            max_num_blocks_dict[(model, num_tensor_parallel_workers)] = max_num_blocks
            total_combos[(model, num_tensor_parallel_workers)] = list(
                filter(
                    lambda input_combination: input_combination.is_under_memory_limit(
                        max_num_blocks * args.block_size
                    ),
                    input_combinations,
                )
            )

    if args.max_points is not None:
        remaining_points = max(args.max_points, 0)
        for key in total_combos:
            if remaining_points <= 0:
                total_combos[key] = []
                continue
            model_combos = total_combos[key]
            total_combos[key] = _select_mixed_points(model_combos, remaining_points)
            remaining_points -= len(total_combos[key])

    pbar = tqdm(total=sum(len(v) for v in total_combos.values()))

    for model in args.models:
        result_df = pd.DataFrame()
        unsupported_results = []
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            profile_df, unsupported = profile_model(
                args,
                model,
                num_tensor_parallel_workers,
                total_combos[(model, num_tensor_parallel_workers)],
                max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                dtype,
                pbar,
            )
            result_df = pd.concat([result_df, profile_df])
            unsupported_results.extend(unsupported)
        # model name would contain '/', so create a directory as required
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/attention.csv", index=False)
        with open(f"{args.output_dir}/{model}/unsupported.json", "w", encoding="utf-8") as f:
            json.dump(unsupported_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
