import unittest

from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.types import DeviceSKUType, NodeSKUType


class Story1SkuConfigTest(unittest.TestCase):
    def test_every_device_sku_type_has_config(self):
        for sku_type in DeviceSKUType:
            config = BaseDeviceSKUConfig.create_from_type(sku_type)
            self.assertEqual(config.get_type(), sku_type)

    def test_every_node_sku_type_maps_to_device_sku_type(self):
        for node_type in NodeSKUType:
            config = BaseNodeSKUConfig.create_from_type(node_type)
            self.assertIn(config.device_sku_type, list(DeviceSKUType))


class Story1QwenModelConfigTest(unittest.TestCase):
    def _estimate_parameter_count(self, config: BaseModelConfig) -> int:
        kv_dim = config.embedding_dim * config.num_kv_heads // config.num_q_heads
        attention = 2 * config.embedding_dim**2 + 2 * config.embedding_dim * kv_dim
        mlp = 3 * config.embedding_dim * config.mlp_hidden_dim
        biases = config.embedding_dim + 2 * kv_dim if config.use_qkv_bias else 0
        layer_norms = 2 * config.embedding_dim
        per_layer = attention + mlp + biases + layer_norms
        embeddings = 2 * config.vocab_size * config.embedding_dim
        return config.num_layers * per_layer + embeddings + config.embedding_dim

    def _assert_model_size(self, model_name: str, hf_parameter_count: int) -> None:
        config = BaseModelConfig.create_from_name(model_name)
        estimated = self._estimate_parameter_count(config)
        relative_error = abs(estimated - hf_parameter_count) / hf_parameter_count
        self.assertLessEqual(relative_error, 0.02)

    def test_qwen_2_5_7b_config_matches_hf(self):
        config = BaseModelConfig.create_from_name("Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config.num_layers, 28)
        self.assertEqual(config.num_q_heads, 28)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertTrue(config.use_qkv_bias)
        self._assert_model_size("Qwen/Qwen2.5-7B-Instruct", 7_610_000_000)

    def test_qwen_2_5_14b_config_matches_hf(self):
        config = BaseModelConfig.create_from_name("Qwen/Qwen2.5-14B-Instruct")
        self.assertEqual(config.num_layers, 48)
        self.assertEqual(config.num_q_heads, 40)
        self.assertEqual(config.num_kv_heads, 8)
        self.assertTrue(config.use_qkv_bias)
        self._assert_model_size("Qwen/Qwen2.5-14B-Instruct", 14_700_000_000)


if __name__ == "__main__":
    unittest.main()
