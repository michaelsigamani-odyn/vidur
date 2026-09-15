import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vidur.profiling.validate import (
    summarize,
    validate_attention_monotonicity,
    validate_completeness,
    validate_mlp_monotonicity,
    validate_repeat_variance,
    validate_roofline,
    validate_telemetry,
    validate_unsupported_points,
)


class ValidateProfilesTest(unittest.TestCase):
    def _write_mlp(self, file_path: Path) -> None:
        dataframe = pd.DataFrame(
            {
                "time_stats.mlp_up_proj.median": [10.0, 12.0],
                "time_stats.mlp_down_proj.median": [8.0, 9.0],
                "n_head": [40, 40],
                "n_kv_head": [8, 8],
                "n_embd": [5120, 5120],
                "n_expanded_embd": [13824, 13824],
                "vocab_size": [152064, 152064],
                "use_gated_mlp": [True, True],
                "num_tokens": [128, 256],
                "num_tensor_parallel_workers": [1, 1],
            }
        )
        dataframe.to_csv(file_path, index=False)

    def _write_attention(self, file_path: Path) -> None:
        dataframe = pd.DataFrame(
            {
                "time_stats.attn_prefill.median": [10.0, 11.0],
                "time_stats.attn_decode.median": [0.0, 0.0],
                "n_embd": [5120, 5120],
                "n_q_head": [40, 40],
                "n_kv_head": [8, 8],
                "block_size": [16, 16],
                "num_tensor_parallel_workers": [1, 1],
                "max_model_len": [4096, 4096],
                "batch_size": [4, 4],
                "prefill_chunk_size": [128, 128],
                "kv_cache_size": [128, 256],
                "is_prefill": [True, True],
                "attention_backend": ["triton", "triton"],
            }
        )
        dataframe.to_csv(file_path, index=False)

    def test_happy_path_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            mlp_path = model_dir / "mlp.csv"
            attention_path = model_dir / "attention.csv"
            self._write_mlp(mlp_path)
            self._write_attention(attention_path)
            telemetry = model_dir / "gpu_telemetry.jsonl"
            telemetry.write_text(json.dumps({"power_draw_watts": 500.0}) + "\n", encoding="utf-8")

            issues = []
            issues.extend(validate_completeness(mlp_path))
            issues.extend(validate_roofline(mlp_path, "mi300x"))
            issues.extend(validate_mlp_monotonicity(mlp_path))
            issues.extend(validate_attention_monotonicity(attention_path))
            issues.extend(validate_unsupported_points(model_dir))
            issues.extend(validate_telemetry(model_dir, "mi300x"))

            summary = summarize(issues)
            self.assertEqual(summary.hard_failures, 0)

    def test_repeat_variance_flags_large_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left.csv"
            right = Path(tmpdir) / "right.csv"
            self._write_mlp(left)
            self._write_mlp(right)
            dataframe = pd.read_csv(right)
            dataframe["time_stats.mlp_up_proj.median"] = [100.0, 120.0]
            dataframe.to_csv(right, index=False)
            issues = validate_repeat_variance(
                left,
                right,
                [
                    "n_head",
                    "n_kv_head",
                    "n_embd",
                    "n_expanded_embd",
                    "vocab_size",
                    "use_gated_mlp",
                    "num_tokens",
                    "num_tensor_parallel_workers",
                ],
            )
            self.assertGreaterEqual(len(issues), 1)
            self.assertEqual(issues[0].check, "repeat_variance")


if __name__ == "__main__":
    unittest.main()
