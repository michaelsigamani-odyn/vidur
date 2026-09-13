import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vidur.profiling.telemetry.backends import (
    AmdSmiBackend,
    NvidiaSmiBackend,
    create_gpu_telemetry_backend,
)
from vidur.profiling.telemetry.recorder import GpuTelemetryRecorder


class GpuTelemetryBackendsTest(unittest.TestCase):
    def setUp(self):
        self.fixture_dir = Path(__file__).parent / "fixtures"
        self.nvidia_output = (self.fixture_dir / "nvidia_smi_query.csv").read_text(
            encoding="utf-8"
        )
        self.amd_metric_output = (self.fixture_dir / "amd_smi_metric.json").read_text(
            encoding="utf-8"
        )
        self.amd_static_output = (self.fixture_dir / "amd_smi_static.json").read_text(
            encoding="utf-8"
        )

    def test_nvidia_backend_parses_query_output(self):
        observed_commands = []

        def runner(command):
            observed_commands.append(command)
            return self.nvidia_output

        backend = NvidiaSmiBackend(command_runner=runner)
        sample = backend.collect(device_index=1)

        self.assertEqual(
            observed_commands[0],
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
        )
        self.assertEqual(sample.device_index, 1)
        self.assertEqual(sample.device_name, "NVIDIA A100-SXM4-80GB")
        self.assertEqual(sample.gpu_utilization_percent, 41.0)
        self.assertEqual(sample.memory_used_mib, 11264.0)
        self.assertEqual(sample.memory_total_mib, 81920.0)
        self.assertEqual(sample.power_draw_watts, 164.5)
        self.assertEqual(sample.temperature_celsius, 66.0)
        self.assertEqual(sample.vram_total_mib, 81920.0)

    def test_amd_backend_parses_json_output(self):
        calls = []

        def runner(command):
            calls.append(command)
            if command[1] == "metric":
                return self.amd_metric_output
            if command[1] == "static":
                return self.amd_static_output
            raise AssertionError(f"Unexpected command: {command}")

        backend = AmdSmiBackend(command_runner=runner)
        sample = backend.collect(device_index=0)

        self.assertEqual(calls[0], ["amd-smi", "metric", "--usage", "--mem-usage", "--power", "--temperature", "--json"])
        self.assertEqual(calls[1], ["amd-smi", "static", "--asic", "--vram", "--json"])
        self.assertEqual(sample.device_index, 0)
        self.assertEqual(sample.device_name, "AMD Radeon PRO W7900")
        self.assertEqual(sample.gpu_utilization_percent, 76.0)
        self.assertEqual(sample.memory_used_mib, 4096.0)
        self.assertEqual(sample.memory_total_mib, 16384.0)
        self.assertEqual(sample.power_draw_watts, 145.0)
        self.assertEqual(sample.temperature_celsius, 68.0)
        self.assertEqual(sample.vram_total_mib, 16384.0)

    def test_amd_backend_missing_metric_returns_none(self):
        payload = json.loads(self.amd_metric_output)
        del payload["gpu_metrics"][0]["average_socket_power"]

        def runner(command):
            if command[1] == "metric":
                return json.dumps(payload)
            if command[1] == "static":
                return self.amd_static_output
            raise AssertionError(f"Unexpected command: {command}")

        backend = AmdSmiBackend(command_runner=runner)
        sample = backend.collect(device_index=0)
        self.assertIsNone(sample.power_draw_watts)

    def test_auto_detect_prefers_nvidia_when_both_tools_exist(self):
        with patch("shutil.which") as which:
            which.side_effect = ["/usr/bin/nvidia-smi", "/opt/rocm/bin/amd-smi"]
            backend = create_gpu_telemetry_backend(
                gpu_vendor="auto",
                command_runner=lambda _: self.nvidia_output,
            )
            self.assertIsInstance(backend, NvidiaSmiBackend)

    def test_recorder_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = GpuTelemetryRecorder(output_dir=tmpdir, gpu_vendor="nvidia")
            recorder._backend = NvidiaSmiBackend(command_runner=lambda _: self.nvidia_output)
            recorder.capture(context={"profiler": "mlp", "model": "meta-llama/Llama-2-7b-hf"})

            lines = Path(tmpdir, "gpu_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["gpu_vendor"], "nvidia")
            self.assertEqual(record["profiler"], "mlp")
            self.assertEqual(record["memory_total_mib"], 81920.0)


if __name__ == "__main__":
    unittest.main()
