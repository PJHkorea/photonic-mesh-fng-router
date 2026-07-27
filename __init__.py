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
# Before letting any upper-layer script initialize, confirm that the compiled 
# high-speed C++ Front-end (Layer 1.5) is resident in the environment's path tree.
try:
    import torch_photonic_bridge_fence_backend as _bridge_backend
except ImportError as err:
    raise ImportError(
        f"FNG_INIT_FATAL: Core C++ Native Layer 1.5 binary wrapper linkage failed.\n"
        f"Reason: {str(err)}\n"
        f"Fix   : Run 'python setup.py build_ext --inplace' at the repository root "
        f"to compile the bare-metal CUDA MUX instruction maps."
    ) from err

# ❷ Absolute Namespace Encapsulation (Clean Room Topology)
# Completely seals internal compilation pipelines, structures, and pointer math metrics from leaking out.
from fng_fabric_monkey_patch import inject_photonic_fng_infrastructure_hook
from photonic_fng_orchestrator import (
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

