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
