from math import ceil
import time
from typing import List

import numpy as np
import torch

from vidur.profiling.common.accelerator import get_torch_device, synchronize_device
from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.model_executor_backend import (
    ParallelConfig,
    create_model_executor_backend,
)

from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


class AttentionWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_num_blocks: int,
        max_model_len: int,
        block_size: int,
        attention_backend: str,
        dtype: torch.dtype,
        gpu_vendor: str,
    ):
        self.time_stats_store = TimerStatsStore(profile_method="kineto")

        self._model_config = model_config
        self._parallel_config = parallel_config
        self._dtype = dtype
        self._device = get_torch_device()

        self._max_model_len = max_model_len
        self._n_worker_q_heads = self._model_config.get_num_q_heads(
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()

        self._block_size = block_size

        self._model_executor_backend = create_model_executor_backend(gpu_vendor)
        self._model_executor_backend.patch_cuda_timer(CudaTimer)
        self._attention_backend = attention_backend
        self._is_first_profile_call = True
        self._attention_wrapper = self._model_executor_backend.create_attention_wrapper(
            self._model_config,
            self._parallel_config,
            self._block_size,
            self._device,
            self._attention_backend,
        )
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)
        # We create (big) KV tensors and reuse them
        self.max_num_blocks = max_num_blocks
        self.kv_cache = self._attention_wrapper.get_cache_block(
            self.max_num_blocks, dtype=self._dtype, device=self._device
        )

    def _get_input_tensors(
        self,
        attention_input: AttentionInput,
    ):
        num_tokens_per_seq = (
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size
        query = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # Create SequenceMetadataProxy objects corresponding to AttentionInput
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):
            num_blocks = ceil(
                (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
            )
            # TODO(nitinkedia7): Investigate why high=max_num_blocks fails with a CUDA illegal memory access
            seq_metadata = SequenceMetadataProxy(
                is_prompt=attention_input.is_prefill,
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,
                processed_len=attention_input.kv_cache_size,
                block_table=np.random.default_rng()
                .integers(low=0, high=self.max_num_blocks - 1, size=num_blocks)
                .tolist(),
            )
            seq_metadata_list.append(seq_metadata)
        return seq_metadata_list, query, key, value, self.kv_cache

    @torch.inference_mode()
    def profile(
        self,
        attention_input: AttentionInput,
    ):
        # batch size is always 1 for prefill and can be different for decode
        assert attention_input.is_valid(self._max_model_len)

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(
            attention_input,
        )
        self._attention_wrapper.begin_forward(seq_metadata_list)

        warmup_start = time.perf_counter()
        for _ in range(WARMUP_STEPS):
            self._attention_wrapper.forward(query, key, value, kv_cache)
        synchronize_device()
        warmup_wall_ms = (time.perf_counter() - warmup_start) * 1e3

        self.time_stats_store.clear_stats()

        active_start = time.perf_counter()
        for _ in range(ACTIVE_STEPS):
            self._attention_wrapper.forward(query, key, value, kv_cache)
        synchronize_device()
        active_wall_ms = (time.perf_counter() - active_start) * 1e3

        self._attention_wrapper.end_forward()

        compile_warmup_wall_ms = warmup_wall_ms if self._is_first_profile_call else 0.0
        self._is_first_profile_call = False
        measured_kernel_wall_ms = active_wall_ms
        measured_kernel_mean_ms = measured_kernel_wall_ms / ACTIVE_STEPS

        return {
            "time_stats": self.time_stats_store.get_stats(),
            "compile_warmup_wall_ms": compile_warmup_wall_ms,
            "warmup_wall_ms": warmup_wall_ms,
            "measured_kernel_wall_ms": measured_kernel_wall_ms,
            "measured_kernel_mean_ms": measured_kernel_mean_ms,
            "active_steps": ACTIVE_STEPS,
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": self._attention_backend,
        }
