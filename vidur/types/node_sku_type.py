from vidur.types.base_int_enum import BaseIntEnum


class NodeSKUType(BaseIntEnum):
    A40_PAIRWISE_NVLINK = 1
    A100_PAIRWISE_NVLINK = 2
    H100_PAIRWISE_NVLINK = 3
    A100_DGX = 4
    H100_DGX = 5
    RADEON_PRO_W7900_SINGLE = 6
    MI300X_SINGLE = 7
    MI300X_8X_INFINITY_FABRIC = 8
    RADEON_8060S_SINGLE = 9
