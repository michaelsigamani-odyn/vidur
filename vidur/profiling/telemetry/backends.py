import csv
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vidur.logger import init_logger

logger = init_logger(__name__)

CommandRunner = Callable[[List[str]], str]


@dataclass
class GpuTelemetrySample:
    device_index: int
    device_name: Optional[str]
    vram_total_mib: Optional[float]
    gpu_utilization_percent: Optional[float]
    memory_used_mib: Optional[float]
    memory_total_mib: Optional[float]
    power_draw_watts: Optional[float]
    temperature_celsius: Optional[float]


class GpuTelemetryBackend(ABC):
    vendor: str

    @abstractmethod
    def collect(self, device_index: int = 0) -> GpuTelemetrySample:
        raise NotImplementedError


def _default_command_runner(command: List[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _normalize_text(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return str(value)


def _try_parse_float(value: Optional[Any], unit_hint: Optional[str] = None) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        numeric = float(value)
        if unit_hint:
            return _convert_to_mib(numeric, unit_hint)
        return numeric

    if isinstance(value, dict):
        for key in ["value", "val", "current", "average", "reading"]:
            if key in value:
                nested_unit = value.get("unit", unit_hint)
                return _try_parse_float(value.get(key), nested_unit)
        return None

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if lowered in {"n/a", "na", "none", "not supported", "[not supported]", "nan"}:
        return None

    tokens = stripped.replace(",", "").split()
    if not tokens:
        return None

    try:
        numeric = float(tokens[0])
    except ValueError:
        return None

    inferred_unit = unit_hint
    if len(tokens) > 1:
        inferred_unit = tokens[1]

    if inferred_unit:
        return _convert_to_mib(numeric, inferred_unit)
    return numeric


def _convert_to_mib(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"b", "byte", "bytes"}:
        return value / (1024 * 1024)
    if normalized in {"kb", "kib"}:
        return value / 1024
    if normalized in {"mb", "mib"}:
        return value
    if normalized in {"gb", "gib"}:
        return value * 1024
    return value


def _parse_metric_with_units(raw_value: Optional[Any], raw_unit: Optional[Any]) -> Optional[float]:
    unit = _normalize_text(raw_unit)
    return _try_parse_float(raw_value, unit)


def _warn_unavailable(metric_name: str, vendor: str, device_index: int) -> None:
    logger.warning(
        "GPU telemetry metric unavailable: vendor=%s device_index=%s metric=%s",
        vendor,
        device_index,
        metric_name,
    )


class NvidiaSmiBackend(GpuTelemetryBackend):
    vendor = "nvidia"

    _QUERY_FIELDS = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "power.draw",
        "temperature.gpu",
    ]

    def __init__(self, command_runner: Optional[CommandRunner] = None):
        self._command_runner = command_runner or _default_command_runner

    def collect(self, device_index: int = 0) -> GpuTelemetrySample:
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(self._QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        output = self._command_runner(command)
        rows = [row for row in csv.reader(output.splitlines()) if row]
        if not rows:
            raise RuntimeError("nvidia-smi returned no rows")

        parsed_rows = [self._parse_row(row) for row in rows]

        selected = None
        for row in parsed_rows:
            if row["device_index"] == device_index:
                selected = row
                break

        if selected is None:
            selected = parsed_rows[0]
            logger.warning(
                "Requested GPU index %s not found in nvidia-smi output; using index %s",
                device_index,
                selected["device_index"],
            )

        for metric_name in [
            "gpu_utilization_percent",
            "memory_used_mib",
            "memory_total_mib",
            "power_draw_watts",
            "temperature_celsius",
        ]:
            if selected[metric_name] is None:
                _warn_unavailable(metric_name, self.vendor, selected["device_index"])

        return GpuTelemetrySample(**selected)

    def _parse_row(self, row: Sequence[str]) -> Dict[str, Optional[float]]:
        if len(row) != len(self._QUERY_FIELDS):
            raise RuntimeError(
                f"Expected {len(self._QUERY_FIELDS)} fields from nvidia-smi, got {len(row)}"
            )

        index = _try_parse_float(row[0])
        if index is None:
            raise RuntimeError(f"Unable to parse GPU index from nvidia-smi row: {row}")

        return {
            "device_index": int(index),
            "device_name": _normalize_text(row[1]),
            "vram_total_mib": _try_parse_float(row[2]),
            "gpu_utilization_percent": _try_parse_float(row[4]),
            "memory_used_mib": _try_parse_float(row[3]),
            "memory_total_mib": _try_parse_float(row[2]),
            "power_draw_watts": _try_parse_float(row[5]),
            "temperature_celsius": _try_parse_float(row[6]),
        }


class AmdSmiBackend(GpuTelemetryBackend):
    vendor = "amd"

    _METRIC_COMMAND = [
        "amd-smi",
        "metric",
        "--usage",
        "--mem-usage",
        "--power",
        "--temperature",
        "--json",
    ]
    _STATIC_COMMAND = ["amd-smi", "static", "--asic", "--vram", "--json"]

    def __init__(self, command_runner: Optional[CommandRunner] = None):
        self._command_runner = command_runner or _default_command_runner

    def collect(self, device_index: int = 0) -> GpuTelemetrySample:
        metric_payload = json.loads(self._command_runner(self._METRIC_COMMAND))
        static_payload = json.loads(self._command_runner(self._STATIC_COMMAND))

        metric_devices = _extract_device_records(metric_payload)
        static_devices = _extract_device_records(static_payload)

        metric_device = _get_device_record(metric_devices, device_index)
        static_device = _get_device_record(static_devices, device_index)

        gpu_utilization = _read_first_numeric(
            metric_device,
            [
                ("usage", None),
                ("gpu_usage", None),
                ("gpu_utilization", None),
                ("gfx_activity", None),
                ("gpu_activity", None),
            ],
        )
        memory_used = _read_first_numeric(
            metric_device,
            [
                ("vram_used", "MiB"),
                ("vram_used_mb", "MiB"),
                ("vram_usage", "MiB"),
                ("used_vram", "MiB"),
                ("used_memory", "MiB"),
                ("used", "MiB"),
                ("mem_usage.used_vram", "MiB"),
                ("mem_usage.used_visible_vram", "MiB"),
            ],
        )
        memory_total = _read_first_numeric(
            metric_device,
            [
                ("vram_total", "MiB"),
                ("vram_total_mb", "MiB"),
                ("total_vram", "MiB"),
                ("total_memory", "MiB"),
                ("total", "MiB"),
                ("mem_usage.total_vram", "MiB"),
                ("mem_usage.total_visible_vram", "MiB"),
            ],
        )
        power_draw = _read_first_numeric(
            metric_device,
            [
                ("average_socket_power", "W"),
                ("average_power", "W"),
                ("power", "W"),
                ("current_socket_power", "W"),
                ("socket_power", "W"),
                ("power.socket_power", "W"),
            ],
        )
        temperature = _read_first_numeric(
            metric_device,
            [
                ("temperature_edge", "C"),
                ("edge_temperature", "C"),
                ("temperature", "C"),
                ("temp", "C"),
                ("temperature_junction", "C"),
                ("junction_temperature", "C"),
                ("temperature_memory", "C"),
                ("memory_temperature", "C"),
                ("temperature.edge", "C"),
                ("temperature.hotspot", "C"),
                ("temperature.mem", "C"),
            ],
        )

        device_name = _read_first_text(
            static_device,
            [
                "product_name",
                "market_name",
                "card_model",
                "device_name",
                "name",
                "asic.market_name",
            ],
        )
        vram_total = _read_first_numeric(
            static_device,
            [
                ("vram_total", "MiB"),
                ("vram_total_mb", "MiB"),
                ("vram_size_mb", "MiB"),
                ("vram_size", "MiB"),
                ("vram.size", "MiB"),
            ],
        )
        if vram_total is None:
            vram_total = memory_total

        metrics: Dict[str, Optional[float]] = {
            "gpu_utilization_percent": gpu_utilization,
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
            "power_draw_watts": power_draw,
            "temperature_celsius": temperature,
            "vram_total_mib": vram_total,
        }
        for metric_name, value in metrics.items():
            if value is None:
                _warn_unavailable(metric_name, self.vendor, device_index)

        return GpuTelemetrySample(
            device_index=device_index,
            device_name=device_name,
            vram_total_mib=vram_total,
            gpu_utilization_percent=gpu_utilization,
            memory_used_mib=memory_used,
            memory_total_mib=memory_total,
            power_draw_watts=power_draw,
            temperature_celsius=temperature,
        )


def _extract_device_records(payload: Any) -> List[Dict[str, Any]]:
    discovered: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if _looks_like_device_record(node):
                discovered.append(node)
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return discovered


def _looks_like_device_record(candidate: Dict[str, Any]) -> bool:
    keys = {key.lower() for key in candidate.keys()}
    interesting = {
        "gpu",
        "gpu_id",
        "device_id",
        "bdf",
        "name",
        "product_name",
        "market_name",
        "usage",
        "gpu_usage",
        "gpu_utilization",
        "gfx_activity",
        "vram_total",
        "vram_used",
        "power",
        "temperature",
    }
    return bool(keys.intersection(interesting))


def _extract_device_index(record: Dict[str, Any]) -> Optional[int]:
    for key in ["gpu", "gpu_id", "device_id", "card", "index"]:
        if key not in record:
            continue
        index = _try_parse_float(record[key])
        if index is not None:
            return int(index)
    return None


def _get_device_record(records: List[Dict[str, Any]], device_index: int) -> Dict[str, Any]:
    if not records:
        return {}

    for record in records:
        record_index = _extract_device_index(record)
        if record_index is not None and record_index == device_index:
            return record

    if device_index < len(records):
        return records[device_index]

    logger.warning(
        "Requested GPU index %s not found in amd-smi output; using first discovered GPU record",
        device_index,
    )
    return records[0]


def _read_first_numeric(
    record: Dict[str, Any], candidates: List[Tuple[str, Optional[str]]]
) -> Optional[float]:
    for field_name, unit_hint in candidates:
        raw_value = _get_by_path(record, field_name)
        if raw_value is None:
            continue

        unit_value = _get_by_path(record, f"{field_name}_unit")
        if unit_value is None:
            unit_value = unit_hint
        parsed = _parse_metric_with_units(raw_value, unit_value)
        if parsed is not None:
            return parsed
    return None


def _read_first_text(record: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for field_name in candidates:
        text = _normalize_text(_get_by_path(record, field_name))
        if text:
            return text
    return None


def _get_by_path(record: Dict[str, Any], field_name: str) -> Optional[Any]:
    current: Any = record
    for token in field_name.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def create_gpu_telemetry_backend(
    gpu_vendor: str = "auto",
    command_runner: Optional[CommandRunner] = None,
) -> GpuTelemetryBackend:
    resolved_vendor = _resolve_gpu_vendor(gpu_vendor)
    if resolved_vendor == "nvidia":
        return NvidiaSmiBackend(command_runner=command_runner)
    if resolved_vendor == "amd":
        return AmdSmiBackend(command_runner=command_runner)
    raise ValueError(f"Unsupported gpu_vendor: {gpu_vendor}")


def _resolve_gpu_vendor(gpu_vendor: str) -> str:
    normalized_vendor = gpu_vendor.lower()
    if normalized_vendor in {"nvidia", "amd"}:
        return normalized_vendor
    if normalized_vendor != "auto":
        raise ValueError(f"Unsupported gpu_vendor: {gpu_vendor}")

    has_nvidia = shutil.which("nvidia-smi") is not None
    has_amd = shutil.which("amd-smi") is not None

    if has_nvidia and has_amd:
        logger.info(
            "Detected both nvidia-smi and amd-smi. Defaulting to nvidia backend for compatibility."
        )
        return "nvidia"
    if has_nvidia:
        return "nvidia"
    if has_amd:
        return "amd"

    raise RuntimeError(
        "Could not auto-detect GPU telemetry backend. Install nvidia-smi or amd-smi, or set --gpu_vendor explicitly."
    )
