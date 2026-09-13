import json
import threading
import time
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


class BackgroundGpuTelemetrySampler:
    def __init__(
        self,
        output_dir: str,
        gpu_vendor: str = "auto",
        interval_seconds: float = 1.0,
        device_index: int = 0,
    ):
        self._backend = create_gpu_telemetry_backend(gpu_vendor)
        self._interval_seconds = interval_seconds
        self._device_index = device_index
        self._output_path = Path(output_dir) / "gpu_telemetry_background.jsonl"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_snapshot: Dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds * 2 + 1.0)

    def latest_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_snapshot)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.perf_counter()
            snapshot: Dict[str, Any]
            try:
                sample = self._backend.collect(device_index=self._device_index)
                snapshot = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_vendor": self._backend.vendor,
                    **asdict(sample),
                }
            except Exception as exc:
                snapshot = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu_vendor": self._backend.vendor,
                    "telemetry_error": str(exc),
                }

            with self._lock:
                self._latest_snapshot = snapshot

            try:
                with self._output_path.open("a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(snapshot, sort_keys=True) + "\n")
            except Exception as exc:
                logger.warning("Unable to persist background telemetry sample: %s", exc)

            elapsed = time.perf_counter() - started
            sleep_seconds = max(self._interval_seconds - elapsed, 0.0)
            if self._stop_event.wait(timeout=sleep_seconds):
                break
