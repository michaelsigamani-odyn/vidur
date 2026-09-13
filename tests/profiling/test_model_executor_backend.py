import unittest
from unittest.mock import MagicMock, patch

import torch

from vidur.profiling.attention.attention_wrapper import AttentionWrapper
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.model_executor_backend import (
    ParallelConfig,
    SarathiBackend,
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
            )

        backend.patch_cuda_timer.assert_called_once()
        backend.create_attention_wrapper.assert_called_once()
        attention_kernel.get_cache_block.assert_called_once()


if __name__ == "__main__":
    unittest.main()
