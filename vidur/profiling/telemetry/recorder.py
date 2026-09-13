import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from vidur.logger import init_logger
from vidur.profiling.telemetry.backends import create_gpu_telemetry_backend

logger = init_logger(__name__)


class GpuTelemetryRecorder:
    def __init__(self, output_dir: str, gpu_vendor: str = "auto"):
        self._output_path = Path(output_dir) / "gpu_telemetry.jsonl"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._backend = create_gpu_telemetry_backend(gpu_vendor)

    @property
    def backend_vendor(self) -> str:
        return self._backend.vendor

    def capture(
        self,
        context: Optional[Dict[str, Any]] = None,
        device_index: int = 0,
    ) -> None:
        try:
            sample = self._backend.collect(device_index=device_index)
        except Exception as exc:
            logger.warning("Unable to collect GPU telemetry sample: %s", exc)
            return

        record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_vendor": self._backend.vendor,
            **asdict(sample),
        }
        if context:
            record.update(context)

        with self._output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
