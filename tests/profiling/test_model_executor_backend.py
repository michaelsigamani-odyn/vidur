import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.model_executor_backend import (
    ParallelConfig,
    SarathiBackend,
    TorchRocmBackend,
    VllmRocmBackend,
    create_model_executor_backend,
)
from vidur.profiling.utils import ProfileMethod


class ModelExecutorBackendFactoryTest(unittest.TestCase):
    def test_factory_selects_sarathi_for_nvidia(self):
        with patch(
            "vidur.profiling.model_executor_backend.resolve_runtime_gpu_vendor",
            return_value="nvidia",
        ):
            backend = create_model_executor_backend("auto")
        self.assertIsInstance(backend, SarathiBackend)

    def test_factory_selects_vllm_rocm_for_amd(self):
        with patch(
            "vidur.profiling.model_executor_backend.resolve_runtime_gpu_vendor",
            return_value="amd",
        ):
            backend = create_model_executor_backend("auto")
        self.assertIsInstance(backend, VllmRocmBackend)

    def test_factory_selects_backend_by_name(self):
        self.assertIsInstance(
            create_model_executor_backend(
                gpu_vendor="auto", model_executor_backend="sarathi"
            ),
            SarathiBackend,
        )
        self.assertIsInstance(
            create_model_executor_backend(
                gpu_vendor="nvidia", model_executor_backend="vllm_rocm"
            ),
            VllmRocmBackend,
        )
        self.assertIsInstance(
            create_model_executor_backend(
                gpu_vendor="amd",
                model_executor_backend="torch",
            ),
            TorchRocmBackend,
        )

    def test_backend_timer_capabilities(self):
        sarathi_backend = SarathiBackend()
        vllm_backend = VllmRocmBackend()
        torch_backend = TorchRocmBackend()

        self.assertFalse(sarathi_backend.needs_manual_linear_timers)
        self.assertFalse(sarathi_backend.needs_manual_embedding_timer)
        self.assertTrue(vllm_backend.needs_manual_linear_timers)
        self.assertTrue(vllm_backend.needs_manual_embedding_timer)
        self.assertFalse(torch_backend.needs_manual_linear_timers)
        self.assertFalse(torch_backend.needs_manual_embedding_timer)

    def test_torch_backend_is_constructible(self):
        backend = TorchRocmBackend()
        self.assertEqual(backend.name, "torch")

    def test_rocm_backend_resolves_flashinfer_to_triton(self):
        backend = VllmRocmBackend()
        self.assertEqual(backend.resolve_attention_backend("flashinfer"), "triton")


class WrapperBackendWiringTest(unittest.TestCase):
    def _make_model_config(self) -> ModelConfig:
        return ModelConfig(
            name="test/model",
            num_layers=2,
            num_q_heads=8,
            num_kv_heads=8,
            embedding_dim=64,
            mlp_hidden_dim=256,
            max_position_embeddings=128,
            use_gated_mlp=True,
            use_bias=False,
            use_qkv_bias=False,
            activation="silu",
            norm="rms_norm",
            post_attn_norm=False,
            vocab_size=32000,
            rope_theta=10000,
        )

    def test_mlp_wrapper_uses_backend_interface(self):
        backend = MagicMock()
        backend.patch_cuda_timer = MagicMock()
        backend.initialize_dummy_weights = MagicMock()

        with patch(
            "vidur.profiling.mlp.mlp_wrapper.create_model_executor_backend",
            return_value=backend,
        ):
            with patch("vidur.profiling.mlp.mlp_wrapper.GPTModel") as gpt_model_cls:
                model = torch.nn.Module()
                model.to = MagicMock(return_value=model)
                model.eval = MagicMock(return_value=model)
                gpt_model_cls.return_value = model

                MlpWrapper(
                    model_config=self._make_model_config(),
                    num_tensor_parallel_workers=1,
                    profile_method=ProfileMethod.PERF_COUNTER.value,
                    rank=0,
                    output_dir="/tmp",
                    gpu_vendor="nvidia",
                    model_executor_backend="sarathi",
                )

        backend.patch_cuda_timer.assert_called_once()
        backend.initialize_dummy_weights.assert_called_once()
        gpt_model_cls.assert_called_once()

    def test_attention_wrapper_uses_backend_interface(self):
        backend = MagicMock()
        backend.patch_cuda_timer = MagicMock()

        attention_kernel = MagicMock()
        attention_kernel.get_cache_block.return_value = object()
        backend.create_attention_wrapper = MagicMock(return_value=attention_kernel)

        with patch(
            "vidur.profiling.attention.attention_wrapper.create_model_executor_backend",
            return_value=backend,
        ):
            AttentionWrapper(
                model_config=self._make_model_config(),
                parallel_config=ParallelConfig(tensor_parallel_size=1),
                max_num_blocks=8,
                max_model_len=128,
                block_size=16,
                attention_backend="triton",
                dtype=torch.float16,
                gpu_vendor="amd",
                model_executor_backend="vllm_rocm",
            )

        backend.patch_cuda_timer.assert_called_once()
        backend.create_attention_wrapper.assert_called_once()
        attention_kernel.get_cache_block.assert_called_once()


class TorchMlpSchemaTest(unittest.TestCase):
    _MLP_TIMER_NAMES = [
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
    _MLP_TIMER_STATS = ["min", "max", "mean", "median", "std"]
    _MLP_BASE_COLUMNS = [
        "n_head",
        "n_kv_head",
        "n_embd",
        "n_expanded_embd",
        "vocab_size",
        "use_gated_mlp",
        "num_tokens",
        "num_tensor_parallel_workers",
    ]

    def _make_model_config(self) -> ModelConfig:
        return ModelConfig(
            name="test/model",
            num_layers=2,
            num_q_heads=8,
            num_kv_heads=8,
            embedding_dim=64,
            mlp_hidden_dim=256,
            max_position_embeddings=128,
            use_gated_mlp=True,
            use_bias=False,
            use_qkv_bias=False,
            activation="silu",
            norm="rms_norm",
            post_attn_norm=False,
            vocab_size=32000,
            rope_theta=10000,
        )

    def test_torch_backend_mlp_columns_match_a100_fixture(self):
        with TemporaryDirectory() as output_dir:
            wrapper = MlpWrapper(
                model_config=self._make_model_config(),
                num_tensor_parallel_workers=1,
                profile_method=ProfileMethod.PERF_COUNTER.value,
                rank=0,
                output_dir=output_dir,
                gpu_vendor="nvidia",
                model_executor_backend="torch",
            )
            result = wrapper.profile(num_tokens=16)

        result_df = pd.DataFrame([result])
        result_df = pd.json_normalize(result_df["time_stats"]).add_prefix("time_stats.").join(
            result_df.drop(columns=["time_stats"])
        )

        ordered_columns = []
        for timer_name in self._MLP_TIMER_NAMES:
            for stat_name in self._MLP_TIMER_STATS:
                column_name = f"time_stats.{timer_name}.{stat_name}"
                if column_name not in result_df.columns:
                    result_df[column_name] = float("nan")
                ordered_columns.append(column_name)
        for base_column in self._MLP_BASE_COLUMNS:
            if base_column not in result_df.columns:
                result_df[base_column] = float("nan")
            ordered_columns.append(base_column)
        result_df = result_df[ordered_columns]

        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "data/profiling/compute/a100/microsoft/phi-2/mlp.csv"
        )
        fixture_columns = pd.read_csv(fixture_path, nrows=0).columns.tolist()

        self.assertEqual(result_df.columns.tolist(), fixture_columns)


if __name__ == "__main__":
    unittest.main()
