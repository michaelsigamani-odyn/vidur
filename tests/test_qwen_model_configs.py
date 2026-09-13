from vidur.config.model_config import BaseModelConfig
from vidur.profiling.utils import get_attention_input_combinations, get_num_tokens_to_profile


def test_qwen_model_configs_load():
    qwen72 = BaseModelConfig.create_from_name("Qwen/Qwen-72B")
    qwen3 = BaseModelConfig.create_from_name("Qwen/Qwen3-8B")

    assert qwen72.num_layers == 80
    assert qwen72.use_qkv_bias is True

    assert qwen3.num_layers == 36
    assert qwen3.num_q_heads == 32
    assert qwen3.num_kv_heads == 8
    assert qwen3.embedding_dim == 4096
    assert qwen3.mlp_hidden_dim == 12288
    assert qwen3.vocab_size == 151936
    assert qwen3.rope_theta == 1000000
    assert qwen3.use_qkv_bias is False


def test_qwen_profile_grids_non_empty():
    for model_name in ("Qwen/Qwen-72B", "Qwen/Qwen3-8B"):
        model_config = BaseModelConfig.create_from_name(model_name)
        num_tokens = get_num_tokens_to_profile(max_num_tokens=1024)
        assert len(num_tokens) > 0

        combos = get_attention_input_combinations(
            max_seq_len=min(model_config.max_position_embeddings, 4096),
            min_batch_size=1,
            max_batch_size=128,
            profile_only_prefill=False,
            profile_only_decode=False,
        )
        assert len(combos) > 0
