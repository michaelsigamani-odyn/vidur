from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil
from typing import Any, List, Type

import torch

from vidur.profiling.common.accelerator import resolve_runtime_gpu_vendor


@dataclass(frozen=True)
class ParallelConfig:
    tensor_parallel_size: int
    pipeline_parallel_size: int = 1


class ModelExecutorBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def needs_manual_linear_timers(self) -> bool:
        return False

    @property
    def needs_manual_embedding_timer(self) -> bool:
        return False

    def create_parallel_config(
        self, tensor_parallel_size: int, pipeline_parallel_size: int = 1
    ) -> ParallelConfig:
        return ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

    def prepare_mlp_runtime(
        self,
        model_name: str,
        tensor_parallel_size: int,
        rank: int,
    ) -> None:
        del model_name, tensor_parallel_size, rank

    def cleanup_mlp_runtime(self) -> None:
        return

    def mlp_execution_context(self) -> Any:
        return nullcontext()

    @abstractmethod
    def patch_cuda_timer(self, cuda_timer_cls: Type[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize_dummy_weights(self, model: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_silu_and_mul_cls(self) -> Type[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_rms_norm_cls(self) -> Type[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_rope_fn(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_column_parallel_linear_cls(self) -> Type[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_row_parallel_linear_cls(self) -> Type[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_vocab_parallel_embedding_cls(self) -> Type[Any]:
        raise NotImplementedError

    @abstractmethod
    def list_attention_backends(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def resolve_attention_backend(self, requested_backend: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_attention_wrapper(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_llm_engine_from_args(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def create_sampling_params(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_cpu_operation_metrics_enum(self) -> Any:
        raise NotImplementedError


class _SarathiAttentionWrapperAdapter:
    def __init__(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
    ) -> None:
        from sarathi.config import ParallelConfig as SarathiParallelConfig
        from sarathi.model_executor.attention import (
            get_attention_wrapper,
            set_attention_backend,
        )

        sarathi_parallel_config = SarathiParallelConfig(
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )

        self._attention_backend = attention_backend
        self._wrapper = get_attention_wrapper()
        set_attention_backend(attention_backend)
        self._wrapper.init(
            model_config,
            sarathi_parallel_config,
            block_size,
            device,
        )

    def get_cache_block(self, max_num_blocks: int, dtype: Any, device: Any) -> Any:
        return self._wrapper.get_cache_block(max_num_blocks, dtype=dtype, device=device)

    def begin_forward(self, seq_metadata_list: Any) -> None:
        self._wrapper.begin_forward(seq_metadata_list)

    def forward(self, query: Any, key: Any, value: Any, kv_cache: Any) -> Any:
        return self._wrapper.forward(query, key, value, kv_cache)

    def end_forward(self) -> None:
        self._wrapper.end_forward()


class _VllmTritonAttentionWrapperAdapter:
    def __init__(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
        cuda_timer_cls: Type[Any],
    ) -> None:
        if attention_backend.lower() != "triton":
            raise ValueError(
                f"Unsupported attention backend '{attention_backend}' for AMD ROCm path. Use triton."
            )

        self._model_config = model_config
        self._parallel_config = parallel_config
        self._block_size = block_size
        self._device = device
        self._seq_metadata_list: Any = None
        self._is_prefill_mode = False

        self._num_q_heads = model_config.get_num_q_heads(parallel_config)
        self._num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self._head_dim = model_config.get_head_size()
        self._softmax_scale = self._head_dim**-0.5
        self._cuda_timer_cls = cuda_timer_cls

    def get_cache_block(self, max_num_blocks: int, dtype: Any, device: Any) -> Any:
        k_cache = torch.zeros(
            (max_num_blocks, self._block_size, self._num_kv_heads, self._head_dim),
            dtype=dtype,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        return (k_cache, v_cache)

    def begin_forward(self, seq_metadata_list: Any) -> None:
        self._seq_metadata_list = seq_metadata_list
        if not seq_metadata_list:
            self._is_prefill_mode = False
            return
        first_mode = bool(seq_metadata_list[0].is_prompt)
        if any(bool(seq_metadata.is_prompt) != first_mode for seq_metadata in seq_metadata_list):
            raise ValueError("Mixed prefill/decode batch is not supported")
        self._is_prefill_mode = first_mode

    def _reshape_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = query.view(-1, self._num_q_heads, self._head_dim)
        k = key.view(-1, self._num_kv_heads, self._head_dim)
        v = value.view(-1, self._num_kv_heads, self._head_dim)
        return q, k, v

    def _materialize_sequence_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        seq_index: int,
        seq_total_len: int,
        seq_processed_len: int,
        block_table: list[int],
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> None:
        kv_len = seq_total_len
        if kv_len <= 0:
            return

        old_len = max(seq_processed_len, 0)
        new_len = max(kv_len - old_len, 0)

        if old_len > 0:
            old_k = torch.randn(
                old_len,
                self._num_kv_heads,
                self._head_dim,
                dtype=k_cache.dtype,
                device=k_cache.device,
            )
            old_v = torch.randn_like(old_k)
        else:
            old_k = new_key.new_empty((0, self._num_kv_heads, self._head_dim))
            old_v = new_value.new_empty((0, self._num_kv_heads, self._head_dim))

        if new_len > 0:
            seq_new_k = new_key[:new_len]
            seq_new_v = new_value[:new_len]
        else:
            seq_new_k = new_key.new_empty((0, self._num_kv_heads, self._head_dim))
            seq_new_v = new_value.new_empty((0, self._num_kv_heads, self._head_dim))

        full_k = torch.cat((old_k, seq_new_k), dim=0)
        full_v = torch.cat((old_v, seq_new_v), dim=0)

        num_blocks = ceil(kv_len / self._block_size)
        if len(block_table) < num_blocks:
            raise ValueError(
                f"block_table too short for seq_index={seq_index}: "
                f"required={num_blocks}, actual={len(block_table)}"
            )

        for token_idx in range(kv_len):
            block_idx = int(block_table[token_idx // self._block_size])
            block_offset = token_idx % self._block_size
            k_cache[block_idx, block_offset] = full_k[token_idx]
            v_cache[block_idx, block_offset] = full_v[token_idx]

    def _build_block_table_tensor(self, seq_metadata_list: Any, max_num_blocks: int) -> torch.Tensor:
        block_table = torch.full(
            (len(seq_metadata_list), max_num_blocks),
            fill_value=0,
            dtype=torch.int32,
            device=self._device,
        )
        for i, seq_metadata in enumerate(seq_metadata_list):
            blocks = torch.tensor(
                seq_metadata.block_table[:max_num_blocks],
                dtype=torch.int32,
                device=self._device,
            )
            block_table[i, : blocks.numel()] = blocks
        return block_table

    def forward(self, query: Any, key: Any, value: Any, kv_cache: Any) -> Any:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        if self._seq_metadata_list is None:
            raise RuntimeError("begin_forward must be called before forward")

        seq_metadata_list = self._seq_metadata_list
        k_cache, v_cache = kv_cache
        with self._cuda_timer_cls("attn_input_reshape"):
            q, k_new, v_new = self._reshape_qkv(query, key, value)

        q_lengths: list[int] = []
        kv_lengths: list[int] = []
        max_q_len = 0
        max_kv_len = 0
        q_cursor = 0
        kv_cursor = 0

        max_num_blocks = max(len(seq.block_table) for seq in seq_metadata_list)

        with self._cuda_timer_cls("attn_kv_cache_save"):
            for seq_index, seq_metadata in enumerate(seq_metadata_list):
                seq_total_len = int(seq_metadata.seq.get_len())
                seq_processed_len = int(
                    seq_metadata.seq.get_num_prompt_tokens_processed()
                )
                seq_q_len = int(seq_metadata.prompt_chunk_len) if seq_metadata.is_prompt else 1
                seq_kv_len = seq_total_len

                seq_key = k_new[kv_cursor : kv_cursor + seq_q_len]
                seq_value = v_new[kv_cursor : kv_cursor + seq_q_len]
                self._materialize_sequence_cache(
                    k_cache,
                    v_cache,
                    seq_index,
                    seq_total_len,
                    seq_processed_len,
                    seq_metadata.block_table,
                    seq_key,
                    seq_value,
                )

                q_lengths.append(seq_q_len)
                kv_lengths.append(seq_kv_len)
                max_q_len = max(max_q_len, seq_q_len)
                max_kv_len = max(max_kv_len, seq_kv_len)
                q_cursor += seq_q_len
                kv_cursor += seq_q_len

        cu_seqlens_q = torch.zeros(len(q_lengths) + 1, device=self._device, dtype=torch.int32)
        if q_lengths:
            cu_seqlens_q[1:] = torch.cumsum(
                torch.tensor(q_lengths, device=self._device, dtype=torch.int32),
                dim=0,
            )
        seqused_k = torch.tensor(kv_lengths, device=self._device, dtype=torch.int32)

        block_table = self._build_block_table_tensor(seq_metadata_list, max_num_blocks)
        out = torch.empty_like(q)

        attn_timer_name = "attn_prefill" if self._is_prefill_mode else "attn_decode"
        with self._cuda_timer_cls(attn_timer_name):
            unified_attention(
                q=q,
                k=k_cache,
                v=v_cache,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_q_len,
                seqused_k=seqused_k,
                max_seqlen_k=max_kv_len,
                softmax_scale=self._softmax_scale,
                causal=True,
                window_size=(-1, -1),
                block_table=block_table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
            )

        with self._cuda_timer_cls("attn_output_reshape"):
            return out.reshape(query.shape[0], self._num_q_heads * self._head_dim)

    def end_forward(self) -> None:
        self._seq_metadata_list = None
        self._is_prefill_mode = False


class SarathiBackend(ModelExecutorBackend):
    @property
    def name(self) -> str:
        return "sarathi"

    @property
    def needs_manual_linear_timers(self) -> bool:
        return False

    @property
    def needs_manual_embedding_timer(self) -> bool:
        return False

    def patch_cuda_timer(self, cuda_timer_cls: Type[Any]) -> None:
        import sarathi.metrics.cuda_timer

        sarathi.metrics.cuda_timer.CudaTimer = cuda_timer_cls

    def initialize_dummy_weights(self, model: Any) -> None:
        from sarathi.model_executor.weight_utils import initialize_dummy_weights

        initialize_dummy_weights(model)

    def get_silu_and_mul_cls(self) -> Type[Any]:
        from sarathi.model_executor.layers.activation import SiluAndMul

        return SiluAndMul

    def get_rms_norm_cls(self) -> Type[Any]:
        from sarathi.model_executor.layers.layernorm import RMSNorm

        return RMSNorm

    def get_rope_fn(self) -> Any:
        from sarathi.model_executor.layers.rotary_embedding import get_rope

        return get_rope

    def get_column_parallel_linear_cls(self) -> Type[Any]:
        from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
            ColumnParallelLinear,
        )

        return ColumnParallelLinear

    def get_row_parallel_linear_cls(self) -> Type[Any]:
        from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
            RowParallelLinear,
        )

        return RowParallelLinear

    def get_vocab_parallel_embedding_cls(self) -> Type[Any]:
        from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
            VocabParallelEmbedding,
        )

        return VocabParallelEmbedding

    def list_attention_backends(self) -> List[str]:
        from sarathi.model_executor.attention import AttentionBackend

        return [backend.value for backend in AttentionBackend]

    def resolve_attention_backend(self, requested_backend: str) -> str:
        available_backends = self.list_attention_backends()
        normalized = requested_backend.lower()

        by_name = {backend.lower(): backend for backend in available_backends}
        if normalized in by_name:
            return by_name[normalized]

        raise ValueError(
            f"Unknown attention backend '{requested_backend}'. "
            f"Available backends: {available_backends}"
        )

    def create_attention_wrapper(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
    ) -> Any:
        return _SarathiAttentionWrapperAdapter(
            model_config=model_config,
            parallel_config=parallel_config,
            block_size=block_size,
            device=device,
            attention_backend=attention_backend,
        )

    def create_llm_engine_from_args(self, **kwargs: Any) -> Any:
        from sarathi import LLMEngine

        return LLMEngine.from_engine_args(**kwargs)

    def create_sampling_params(self, **kwargs: Any) -> Any:
        from sarathi import SamplingParams

        return SamplingParams(**kwargs)

    def get_cpu_operation_metrics_enum(self) -> Any:
        from sarathi.metrics.constants import CpuOperationMetrics

        return CpuOperationMetrics


class VllmRocmBackend(ModelExecutorBackend):
    def __init__(self) -> None:
        self._vllm_runtime_context = None
        self._vllm_runtime_token = None
        self._cuda_timer_cls: Type[Any] | None = None

    @property
    def name(self) -> str:
        return "vllm_rocm"

    @property
    def needs_manual_linear_timers(self) -> bool:
        return True

    @property
    def needs_manual_embedding_timer(self) -> bool:
        return True

    def patch_cuda_timer(self, cuda_timer_cls: Type[Any]) -> None:
        self._cuda_timer_cls = cuda_timer_cls

    def prepare_mlp_runtime(
        self,
        model_name: str,
        tensor_parallel_size: int,
        rank: int,
    ) -> None:
        import vllm.model_executor.parameter as parameter_mod
        import vllm.model_executor.layers.linear as linear_mod
        import vllm.model_executor.layers.vocab_parallel_embedding as embedding_mod
        from vllm import EngineArgs
        try:
            from vllm.config.vllm import set_current_vllm_config
        except ModuleNotFoundError:
            from vllm.config import set_current_vllm_config

        for module in (parameter_mod, linear_mod, embedding_mod):
            module.get_tensor_model_parallel_rank = lambda: rank
            module.get_tensor_model_parallel_world_size = lambda: tensor_parallel_size

        engine_args = EngineArgs(
            model=model_name,
            tokenizer=model_name,
            load_format="dummy",
            tensor_parallel_size=1,
            max_model_len=2048,
            disable_log_stats=True,
            trust_remote_code=True,
        )
        vllm_config = engine_args.create_engine_config(usage_context=None)
        self._vllm_runtime_context = set_current_vllm_config(vllm_config)
        self._vllm_runtime_token = self._vllm_runtime_context.__enter__()

    def cleanup_mlp_runtime(self) -> None:
        if self._vllm_runtime_context is None:
            return
        self._vllm_runtime_context.__exit__(None, None, None)
        self._vllm_runtime_context = None
        self._vllm_runtime_token = None

    def mlp_execution_context(self) -> Any:
        if self._vllm_runtime_context is None:
            raise RuntimeError("VllmRocmBackend runtime is not initialized")
        return nullcontext(self._vllm_runtime_token)

    def initialize_dummy_weights(self, model: Any) -> None:
        for _, parameter in model.named_parameters(recurse=True):
            if not parameter.requires_grad:
                continue
            if parameter.dtype.is_floating_point:
                parameter.data.normal_(mean=0.0, std=0.02)
            else:
                parameter.data.zero_()

    def get_silu_and_mul_cls(self) -> Type[Any]:
        from vllm.model_executor.layers.activation import SiluAndMul

        return SiluAndMul

    def get_rms_norm_cls(self) -> Type[Any]:
        from vllm.model_executor.layers.layernorm import RMSNorm

        return RMSNorm

    def get_rope_fn(self) -> Any:
        from vllm.model_executor.layers.rotary_embedding import get_rope as vllm_get_rope

        def _get_rope_adapter(
            head_size: int,
            rotary_dim: int,
            max_position: int,
            base: float,
            is_neox_style: bool,
            rope_scaling: dict[str, Any] | None,
        ) -> Any:
            rope_parameters: dict[str, Any] = {
                "rope_theta": base,
                "rope_dim": rotary_dim,
            }
            if rope_scaling:
                rope_parameters.update(rope_scaling)

            return vllm_get_rope(
                head_size=head_size,
                max_position=max_position,
                is_neox_style=is_neox_style,
                rope_parameters=rope_parameters,
            )

        return _get_rope_adapter

    def get_column_parallel_linear_cls(self) -> Type[Any]:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear as VllmColumnParallelLinear,
        )

        class ColumnParallelLinear(VllmColumnParallelLinear):
            def __init__(
                self,
                input_size: int,
                output_size: int,
                bias: bool = True,
                gather_output: bool = False,
                linear_metric_name: str = "",
                world_size: int = 1,
            ):
                super().__init__(
                    input_size=input_size,
                    output_size=output_size,
                    bias=bias,
                    gather_output=gather_output,
                    prefix=linear_metric_name,
                    tp_size=world_size,
                    tp_rank=0,
                    return_bias=True,
                )

        return ColumnParallelLinear

    def get_row_parallel_linear_cls(self) -> Type[Any]:
        from vllm.model_executor.layers.linear import (
            RowParallelLinear as VllmRowParallelLinear,
        )

        class RowParallelLinear(VllmRowParallelLinear):
            def __init__(
                self,
                input_size: int,
                output_size: int,
                bias: bool = True,
                input_is_parallel: bool = True,
                reduce_results: bool = True,
                linear_metric_name: str = "",
                world_size: int = 1,
            ):
                del world_size
                effective_bias = bias and reduce_results
                super().__init__(
                    input_size=input_size,
                    output_size=output_size,
                    bias=effective_bias,
                    input_is_parallel=input_is_parallel,
                    reduce_results=reduce_results,
                    prefix=linear_metric_name,
                    return_bias=True,
                )

        return RowParallelLinear

    def get_vocab_parallel_embedding_cls(self) -> Type[Any]:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding as VllmVocabParallelEmbedding,
        )

        class VocabParallelEmbedding(VllmVocabParallelEmbedding):
            def __init__(
                self,
                num_embeddings: int,
                embedding_dim: int,
                linear_metric_name: str = "",
                reduce_results: bool = False,
                world_size: int = 1,
                rank: int = 0,
            ):
                del linear_metric_name, reduce_results, world_size, rank
                super().__init__(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    prefix="emb",
                )

        return VocabParallelEmbedding

    def list_attention_backends(self) -> List[str]:
        return ["triton"]

    def resolve_attention_backend(self, requested_backend: str) -> str:
        normalized = requested_backend.lower()
        if normalized == "flashinfer":
            return "triton"
        if normalized in self.list_attention_backends():
            return normalized
        raise ValueError(
            f"Unsupported attention backend '{requested_backend}' for VllmRocmBackend. "
            "Use triton."
        )

    def create_attention_wrapper(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
    ) -> Any:
        if self._cuda_timer_cls is None:
            raise RuntimeError("patch_cuda_timer must be called before creating attention wrapper")
        return _VllmTritonAttentionWrapperAdapter(
            model_config=model_config,
            parallel_config=parallel_config,
            block_size=block_size,
            device=device,
            attention_backend=attention_backend,
            cuda_timer_cls=self._cuda_timer_cls,
        )

    def create_llm_engine_from_args(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    def create_sampling_params(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    def get_cpu_operation_metrics_enum(self) -> Any:
        raise NotImplementedError


class _TorchSiluAndMul(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2


class _TorchRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed.to(dtype=input_dtype)) * self.weight.to(dtype=input_dtype)


class _TorchRotaryEmbedding:
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool,
        rope_scaling: dict[str, Any] | None,
    ) -> None:
        if rotary_dim > head_size:
            raise ValueError(
                f"rotary_dim={rotary_dim} must be <= head_size={head_size}"
            )
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim={rotary_dim} must be even")

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = float(base)
        self.max_position = max_position
        self.is_neox_style = is_neox_style
        self.rope_scaling = rope_scaling

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def _get_cos_sin(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        pos = positions.to(dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == 0:
            return x
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x_rotated = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return torch.cat((x_rotated, x_pass), dim=-1)

    def __call__(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[-1] % self.head_size != 0 or key.shape[-1] % self.head_size != 0:
            raise ValueError("query/key hidden dimensions must be divisible by head dimension")
        q = query.view(query.shape[0], -1, self.head_size)
        k = key.view(key.shape[0], -1, self.head_size)
        cos, sin = self._get_cos_sin(positions, q.device, q.dtype)
        q_out = self._apply_rope(q, cos, sin).reshape_as(query)
        k_out = self._apply_rope(k, cos, sin).reshape_as(key)
        return q_out, k_out


class _TorchColumnParallelLinear(torch.nn.Module):
    _timer_cls: Type[Any] | None = None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        linear_metric_name: str = "",
        world_size: int = 1,
    ) -> None:
        super().__init__()
        del gather_output
        if output_size % world_size != 0:
            raise ValueError(
                f"output_size={output_size} must be divisible by world_size={world_size}"
            )
        output_size_per_partition = output_size // world_size
        self.linear = torch.nn.Linear(input_size, output_size_per_partition, bias=bias)
        self._metric_name = linear_metric_name
        self._timer = (
            self._timer_cls(self._metric_name)
            if self._timer_cls is not None and self._metric_name
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self._timer is None:
            return self.linear(x), None
        with self._timer:
            return self.linear(x), None


class _TorchRowParallelLinear(torch.nn.Module):
    _timer_cls: Type[Any] | None = None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        linear_metric_name: str = "",
        world_size: int = 1,
    ) -> None:
        super().__init__()
        del input_is_parallel
        if input_size % world_size != 0:
            raise ValueError(
                f"input_size={input_size} must be divisible by world_size={world_size}"
            )
        input_size_per_partition = input_size // world_size
        effective_bias = bias and reduce_results
        self.linear = torch.nn.Linear(
            input_size_per_partition,
            output_size,
            bias=effective_bias,
        )
        self._metric_name = linear_metric_name
        self._timer = (
            self._timer_cls(self._metric_name)
            if self._timer_cls is not None and self._metric_name
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self._timer is None:
            return self.linear(x), None
        with self._timer:
            return self.linear(x), None


class _TorchVocabParallelEmbedding(torch.nn.Module):
    _timer_cls: Type[Any] | None = None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        linear_metric_name: str = "",
        reduce_results: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__()
        del reduce_results
        if num_embeddings % world_size != 0:
            raise ValueError(
                f"num_embeddings={num_embeddings} must be divisible by world_size={world_size}"
            )
        self._vocab_size_per_partition = num_embeddings // world_size
        self._vocab_start_index = rank * self._vocab_size_per_partition
        self._vocab_end_index = self._vocab_start_index + self._vocab_size_per_partition
        self.embedding = torch.nn.Embedding(self._vocab_size_per_partition, embedding_dim)
        self._metric_name = linear_metric_name
        self._timer = (
            self._timer_cls(self._metric_name)
            if self._timer_cls is not None and self._metric_name
            else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        local_ids = input_ids - self._vocab_start_index
        out_of_range = (input_ids < self._vocab_start_index) | (
            input_ids >= self._vocab_end_index
        )
        if torch.any(out_of_range):
            raise ValueError("input_ids contain token outside local vocab shard")
        if self._timer is None:
            return self.embedding(local_ids)
        with self._timer:
            return self.embedding(local_ids)


class TorchRocmBackend(ModelExecutorBackend):
    """Pure-torch ROCm backend scaffold.

    Incomplete: attention wrapper is intentionally unimplemented in Phase 1.
    """

    @property
    def name(self) -> str:
        return "torch"

    @property
    def needs_manual_linear_timers(self) -> bool:
        return False

    @property
    def needs_manual_embedding_timer(self) -> bool:
        return False

    def patch_cuda_timer(self, cuda_timer_cls: Type[Any]) -> None:
        _TorchColumnParallelLinear._timer_cls = cuda_timer_cls
        _TorchRowParallelLinear._timer_cls = cuda_timer_cls
        _TorchVocabParallelEmbedding._timer_cls = cuda_timer_cls

    def initialize_dummy_weights(self, model: Any) -> None:
        for _, parameter in model.named_parameters(recurse=True):
            if not parameter.requires_grad:
                continue
            if parameter.dtype.is_floating_point:
                parameter.data.normal_(mean=0.0, std=0.02)
            else:
                parameter.data.zero_()

    def get_silu_and_mul_cls(self) -> Type[Any]:
        return _TorchSiluAndMul

    def get_rms_norm_cls(self) -> Type[Any]:
        return _TorchRMSNorm

    def get_rope_fn(self) -> Any:
        def _get_rope_adapter(
            head_size: int,
            rotary_dim: int,
            max_position: int,
            base: float,
            is_neox_style: bool,
            rope_scaling: dict[str, Any] | None,
        ) -> Any:
            return _TorchRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position=max_position,
                base=base,
                is_neox_style=is_neox_style,
                rope_scaling=rope_scaling,
            )

        return _get_rope_adapter

    def get_column_parallel_linear_cls(self) -> Type[Any]:
        return _TorchColumnParallelLinear

    def get_row_parallel_linear_cls(self) -> Type[Any]:
        return _TorchRowParallelLinear

    def get_vocab_parallel_embedding_cls(self) -> Type[Any]:
        return _TorchVocabParallelEmbedding

    def list_attention_backends(self) -> List[str]:
        return ["triton_fa", "sdpa"]

    def resolve_attention_backend(self, requested_backend: str) -> str:
        normalized = requested_backend.lower()
        if normalized == "auto":
            return "sdpa"
        if normalized in self.list_attention_backends():
            return normalized
        raise ValueError(
            f"Unsupported attention backend '{requested_backend}' for TorchRocmBackend. "
            "Use triton_fa or sdpa."
        )

    def create_attention_wrapper(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
        block_size: int,
        device: Any,
        attention_backend: str,
    ) -> Any:
        del model_config, parallel_config, block_size, device, attention_backend
        raise NotImplementedError

    def create_llm_engine_from_args(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    def create_sampling_params(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    def get_cpu_operation_metrics_enum(self) -> Any:
        raise NotImplementedError


def create_model_executor_backend(
    gpu_vendor: str = "auto",
    model_executor_backend: str = "auto",
) -> ModelExecutorBackend:
    normalized_backend = model_executor_backend.lower()
    if normalized_backend == "sarathi":
        return SarathiBackend()
    if normalized_backend == "vllm_rocm":
        return VllmRocmBackend()
    if normalized_backend == "torch":
        return TorchRocmBackend()
    if normalized_backend != "auto":
        raise ValueError(
            f"Unknown model executor backend '{model_executor_backend}'. "
            "Use one of: auto, sarathi, vllm_rocm, torch."
        )

    resolved_vendor = resolve_runtime_gpu_vendor(gpu_vendor)
    if resolved_vendor == "amd":
        return VllmRocmBackend()
    return SarathiBackend()
