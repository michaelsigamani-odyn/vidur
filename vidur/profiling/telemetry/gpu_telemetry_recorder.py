import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class GpuTelemetrySample:
    timestamp: str
    gpu_index: int
    utilization_percent: Optional[float]
    memory_used_bytes: int
    memory_total_bytes: int
    power_watts: Optional[float]
    temperature_c: Optional[float]

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class GpuTelemetryRecorder:
    def __init__(self, gpu_vendor: str = "nvidia", gpu_index: int = 0):
        if gpu_vendor.lower() != "nvidia":
            raise ValueError("GpuTelemetryRecorder currently supports gpu_vendor='nvidia' only")
        self._gpu_index = gpu_index

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        value = value.strip()
        if value in {"N/A", "[N/A]", ""}:
            return None
        return float(value)

    def sample(self) -> GpuTelemetrySample:
        query = "utilization.gpu,power.draw,temperature.gpu"
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={self._gpu_index}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        utilization, power, temperature = [value.strip() for value in output.split(",")]

        free, total = torch.cuda.mem_get_info(device=self._gpu_index)
        used = total - free

        return GpuTelemetrySample(
            timestamp=datetime.now(UTC).isoformat(),
            gpu_index=self._gpu_index,
            utilization_percent=self._to_float(utilization),
            memory_used_bytes=int(used),
            memory_total_bytes=int(total),
            power_watts=self._to_float(power),
            temperature_c=self._to_float(temperature),
        )

    def record(self, output_path: str, duration_s: int = 10, interval_s: float = 1.0) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        end_time = time.time() + duration_s
        with out_path.open("w", encoding="utf-8") as handle:
            while time.time() < end_time:
                handle.write(f"{self.sample().to_json()}\n")
                handle.flush()
                time.sleep(interval_s)

        return out_path
