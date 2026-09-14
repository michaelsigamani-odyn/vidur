from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int
    memory_bandwidth_gbps: int


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45
    memory_bandwidth_gbps: int = 696

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80
    memory_bandwidth_gbps: int = 2039

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80
    memory_bandwidth_gbps: int = 3350

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class RadeonProW7900DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 123
    total_memory_gb: int = 48
    memory_bandwidth_gbps: int = 864

    @staticmethod
    def get_type():
        return DeviceSKUType.RADEON_PRO_W7900


@dataclass
class Mi300XDeviceSKUConfig(BaseDeviceSKUConfig):
    # Source: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
    fp16_tflops: int = 1307
    total_memory_gb: int = 192
    memory_bandwidth_gbps: int = 5300

    @staticmethod
    def get_type():
        return DeviceSKUType.MI300X


@dataclass
class Radeon8060SDeviceSKUConfig(BaseDeviceSKUConfig):
    # FP16 TFLOPS derived from AMD stream processors + boost clock product page specs.
    # Source: https://www.amd.com/en/products/graphics/amd-radeon-8060s.html
    fp16_tflops: int = 30
    # Configured allocatable GPU memory read from odyn-radeon2 via `amd-smi static --vram`.
    total_memory_gb: int = 64
    # LPDDR5X-8000 256-bit system bandwidth is shared with the CPU on Strix Halo.
    # Source: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
    memory_bandwidth_gbps: int = 256

    @staticmethod
    def get_type():
        return DeviceSKUType.RADEON_8060S
