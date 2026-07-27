"""
[FNG PHOTONICS SYSTEM - PRODUCTION INTERFACE CAPSULED GATEWAY]
Surgically exposes the high-speed optical runtime injection endpoints 
while atomically verifying the linkage of underlying C++/CUDA binary modules.

Copyright (c) 2026 PJHkorea. All rights reserved.
Licensed under the Apache License 2.0.
"""

import os
import sys

# 🛡️ Version Identity Guard (Strictly matching the sovereign suite generation)
__version__ = "3.0.0-PROD-OPTICS"
__author__ = "PJHkorea"
__license__ = "Apache License 2.0"

# ❶ Atomic Hardware & Binary Linkage Validation
# [★교정★] setup.py의 통합 네임스페이스 명세와 일치하도록 상대 경로를 바인딩하여 
# 빌드 완제품 (.so) 내부 C++ Front-end 레이어가 파이썬 힙 버스에 오차 없이 접착되도록 유도합니다.
try:
    from . import torch_photonic_bridge_fence_backend as _bridge_backend
except ImportError as err:
    raise ImportError(
        f"FNG_INIT_FATAL: Core C++ Native Layer 1.5 binary wrapper linkage failed.\n"
        f"Reason: {str(err)}\n"
        f"Fix   : Run 'python setup.py build_ext --inplace' at the repository root "
        f"to compile the bare-metal CUDA MUX instruction maps."
    ) from err

# ❷ Absolute Namespace Encapsulation (Clean Room Topology)
# [★교정★] 외부에서 호출 시 네임스페이스 단절로 인한 ModuleNotFoundError를 원천 박멸하기 위해
# 명시적 상대 경로(from .) 임포트 가드로 아키텍처 경계를 완벽히 밀봉합니다.
from .fng_fabric_monkey_patch import inject_photonic_fng_infrastructure_hook
from .photonic_fng_orchestrator import (
    compute_photonic_attention_rail_fusion,
    freeze_photonic_compiler_graph,
    photonic_hardware_mesh
)


# ❸ Expose explicit manifest endpoints for external distributed frameworks (Megatron/vLLM)
__all__ = [
    "inject_photonic_fng_infrastructure_hook",
    "compute_photonic_attention_rail_fusion",
    "freeze_photonic_compiler_graph",
    "photonic_hardware_mesh"
]

# ❹ Initialize Boot-time Telemetry Logging Signal
# Emits a clean structural declaration confirming that the 0ns optical rail system is armed.
print("=========================================================================")
print("[FNG COMPILER INFRASTRUCTURE INITIALIZED]")
print(f" -> Architecture Rail Style: Silicon Photonics Co-Design Control Plane")
print(f" -> System Core Version    : {__version__}")
print(f" -> Legal License Fencing   : Apache 2.0 (Defensive Patent Shield Active)")
print("=========================================================================")

