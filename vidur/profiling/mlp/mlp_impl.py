from typing import Optional

import torch

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.model_executor_backend import ModelExecutorBackend

REUSE_MEMORY = True


class CausalSelfAttention(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        model_executor_backend: ModelExecutorBackend,
    ):
        super().__init__()
        assert config.embedding_dim % config.num_q_heads == 0
        assert config.embedding_dim % world_size == 0
        assert config.num_q_heads % world_size == 0
        assert config.num_kv_heads % world_size == 0

        ColumnParallelLinear = model_executor_backend.get_column_parallel_linear_cls()
        RowParallelLinear = model_executor_backend.get_row_parallel_linear_cls()
        get_rope = model_executor_backend.get_rope_fn()

        self.head_dim = config.embedding_dim // config.num_q_heads
        self.num_q_heads_per_worker = config.num_q_heads // world_size
        self.num_kv_heads_per_worker = config.num_kv_heads // world_size

        self.q_size = self.num_q_heads_per_worker * self.head_dim
        self.kv_size = self.num_kv_heads_per_worker * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._manual_linear_timers = model_executor_backend.name == "vllm_rocm"

        self.qkv_proj = ColumnParallelLinear(
            config.embedding_dim,
            (config.num_q_heads + 2 * config.num_kv_heads) * self.head_dim,
            bias=config.use_bias or config.use_qkv_bias,
            gather_output=False,
            linear_metric_name="attn_pre_proj",
            world_size=world_size,
        )

        self.o_proj = RowParallelLinear(
            config.num_q_heads * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            world_size=world_size,
        )
        self.rotary_emb = None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling,
            )
        self._attn_rope_timer = CudaTimer("attn_rope")
        self._attn_pre_proj_timer = CudaTimer("attn_pre_proj")
        self._attn_post_proj_timer = CudaTimer("attn_post_proj")

    def forward(self, hidden_states, positions):
        if self._manual_linear_timers:
            with self._attn_pre_proj_timer:
                qkv, _ = self.qkv_proj(hidden_states)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        # output from attn has the same shape as q
        attn_output = torch.randn_like(q)
        if self._manual_linear_timers:
            with self._attn_post_proj_timer:
                output, _ = self.o_proj(attn_output)
        else:
            output, _ = self.o_proj(attn_output)
        return output


class MLP(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        model_executor_backend: ModelExecutorBackend,
    ):
        super().__init__()

        assert config.embedding_dim % world_size == 0
        self._manual_linear_timers = model_executor_backend.name == "vllm_rocm"

        ColumnParallelLinear = model_executor_backend.get_column_parallel_linear_cls()
        RowParallelLinear = model_executor_backend.get_row_parallel_linear_cls()
        SiluAndMul = model_executor_backend.get_silu_and_mul_cls()

        if config.use_gated_mlp:
            self.up_proj = ColumnParallelLinear(
                config.embedding_dim,
                2 * config.mlp_hidden_dim,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
            )
            self.act = SiluAndMul()
        else:
            self.up_proj = ColumnParallelLinear(
                config.embedding_dim,
                config.mlp_hidden_dim,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
            )
            self.act = torch.nn.GELU()

        self.down_proj = RowParallelLinear(
            config.mlp_hidden_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            world_size=world_size,
            reduce_results=False,
            linear_metric_name="mlp_down_proj",
        )

        self.mlp_act_timer = CudaTimer("mlp_act")
        self.mlp_up_proj_timer = CudaTimer("mlp_up_proj")
        self.mlp_down_proj_timer = CudaTimer("mlp_down_proj")

    def forward(self, hidden_states):
        if self._manual_linear_timers:
            with self.mlp_up_proj_timer:
                hidden_states, _ = self.up_proj(hidden_states)
        else:
            hidden_states, _ = self.up_proj(hidden_states)
        with self.mlp_act_timer:
            hidden_states = self.act(hidden_states)
        if self._manual_linear_timers:
            with self.mlp_down_proj_timer:
                hidden_states, _ = self.down_proj(hidden_states)
        else:
            hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class GPTBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        model_executor_backend: ModelExecutorBackend,
    ):
        super().__init__()

        RMSNorm = model_executor_backend.get_rms_norm_cls()

        if config.norm == "layer_norm":
            self.input_layernorm = torch.nn.LayerNorm(config.embedding_dim)
        elif config.norm == "rms_norm":
            self.input_layernorm = RMSNorm(config.embedding_dim)
        else:
            raise ValueError(f"Unknown norm: {config.norm} for input_layernorm")

        self._post_attn_norm = config.post_attn_norm
        if config.post_attn_norm:
            if config.norm == "rms_norm":
                self.post_attention_layernorm = RMSNorm(config.embedding_dim)
            else:
                raise ValueError(
                    f"Unknown norm: {config.norm} for post_attention_layernorm"
                )

        self.attn = CausalSelfAttention(config, world_size, model_executor_backend)
        self.mlp = MLP(config, world_size, model_executor_backend)

        self.input_layernorm_timer = CudaTimer("input_layernorm")
        self.post_attention_layernorm_timer = CudaTimer("post_attention_layernorm")
        self.add_timer = CudaTimer("add")

    def forward(self, positions, hidden_states, residual):
        if self._post_attn_norm:
            return self._forward_with_post_attn_norm(positions, hidden_states, residual)
        else:
            return self._forward_without_post_attn_norm(positions, hidden_states)

    def _forward_with_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        # Self Attention
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        with self.post_attention_layernorm_timer:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        with self.add_timer:
            hidden_states = residual + hidden_states
        return hidden_states

    def _forward_without_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        with self.add_timer:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class GPTModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        model_executor_backend: ModelExecutorBackend,
        num_repeat_steps: int = 1,
    ):
        super().__init__()

        self.num_repeat_steps = num_repeat_steps
        self._manual_embedding_timer = model_executor_backend.name == "vllm_rocm"

        VocabParallelEmbedding = model_executor_backend.get_vocab_parallel_embedding_cls()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.embedding_dim,
            linear_metric_name="emb",
            reduce_results=False,
            world_size=world_size,
            rank=0,
        )

        self.block = GPTBlock(
            config,
            world_size=world_size,
            model_executor_backend=model_executor_backend,
        )
        self.embed_timer = CudaTimer("emb")

    def forward(self, input_ids, positions):
        if self._manual_embedding_timer:
            with self.embed_timer:
                hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states
        for _ in range(self.num_repeat_steps):
            if self._manual_embedding_timer:
                with self.embed_timer:
                    hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = self.embed_tokens(input_ids)
            hidden_states = self.block(
                positions,
                hidden_states,
                residual,
            )

        return hidden_states
