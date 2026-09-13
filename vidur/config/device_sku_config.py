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
class Radeon8060SDeviceSKUConfig(BaseDeviceSKUConfig):
    # Source: AMD Ryzen AI Max+ 395 specs list "up to 96 GB" graphics memory.
    # https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
    total_memory_gb: int = 96
    # Source: LPDDR5X-8000 MT/s and 256-bit memory interface for Strix Halo platform.
    # Bandwidth = 8000e6 * 32 bytes / s = 256 GB/s.
    # https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
    memory_bandwidth_gbps: int = 256
    # Source: Radeon 8060S has 40 CUs (2560 shaders) up to 2.9 GHz boost.
    # Dense FP16 ~= FP32 * 2 = (2560 * 2 * 2.9e9) * 2 = 29.7 TFLOPS.
    # https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
    fp16_tflops: int = 30

    @staticmethod
    def get_type():
        return DeviceSKUType.RADEON_8060S
