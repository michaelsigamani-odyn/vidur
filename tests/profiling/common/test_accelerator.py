import os
import unittest
from unittest.mock import patch

from vidur.profiling.common.accelerator import (
    get_runtime_trace_categories,
    resolve_attention_backend_for_vendor,
    resolve_runtime_gpu_vendor,
    set_visible_device,
)


class AcceleratorUtilsTest(unittest.TestCase):
    def test_set_visible_device_sets_cuda_and_hip(self):
        with patch.dict(os.environ, {}, clear=True):
            set_visible_device(3)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(os.environ["HIP_VISIBLE_DEVICES"], "3")

    def test_resolve_runtime_gpu_vendor_explicit(self):
        self.assertEqual(resolve_runtime_gpu_vendor("amd"), "amd")
        self.assertEqual(resolve_runtime_gpu_vendor("nvidia"), "nvidia")

    def test_resolve_runtime_gpu_vendor_auto_prefers_hip_runtime(self):
        with patch("vidur.profiling.common.accelerator.torch.version.hip", new="6.2"):
            with patch(
                "vidur.profiling.common.accelerator.torch.version.cuda", new=None
            ):
                self.assertEqual(resolve_runtime_gpu_vendor("auto"), "amd")

    def test_resolve_runtime_gpu_vendor_auto_prefers_cuda_runtime(self):
        with patch("vidur.profiling.common.accelerator.torch.version.hip", new=None):
            with patch(
                "vidur.profiling.common.accelerator.torch.version.cuda", new="12.4"
            ):
                self.assertEqual(resolve_runtime_gpu_vendor("auto"), "nvidia")

    def test_resolve_attention_backend_for_amd_falls_back_from_flashinfer(self):
        with patch("vidur.profiling.common.accelerator.torch.version.hip", new="6.2"):
            resolved = resolve_attention_backend_for_vendor(
                attention_backend="flashinfer",
                available_backends=["flashinfer", "triton", "xformers"],
                gpu_vendor="auto",
            )
        self.assertEqual(resolved, "triton")

    def test_resolve_attention_backend_keeps_requested_for_non_amd(self):
        with patch("vidur.profiling.common.accelerator.torch.version.hip", new=None):
            with patch(
                "vidur.profiling.common.accelerator.torch.version.cuda", new="12.4"
            ):
                resolved = resolve_attention_backend_for_vendor(
                    attention_backend="flashinfer",
                    available_backends=["flashinfer", "triton"],
                    gpu_vendor="auto",
                )
        self.assertEqual(resolved, "flashinfer")

    def test_resolve_attention_backend_for_amd_without_fallback_raises(self):
        with patch("vidur.profiling.common.accelerator.torch.version.hip", new="6.2"):
            with self.assertRaises(RuntimeError):
                resolve_attention_backend_for_vendor(
                    attention_backend="flashinfer",
                    available_backends=["flashinfer"],
                    gpu_vendor="auto",
                )

    def test_runtime_trace_categories_include_cuda_and_hip(self):
        categories = get_runtime_trace_categories()
        self.assertIn("cuda_runtime", categories)
        self.assertIn("hip_runtime", categories)


if __name__ == "__main__":
    unittest.main()
