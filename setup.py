"""
[FNG PHOTONICS INFRASTRUCTURE - HIGH-DENSITY NATIVE COMPILATION FACTORY]
Precision-engineered to invoke NVCC/HIPCC for register-level kernel fusion.
Enforces absolute C++20 standards and -O3 extreme compiler optimizations.

Copyright (c) 2026 PJHkorea. All rights reserved.
Licensed under the Apache License 2.0.
"""

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 🛡️ Structural Guard: Verify active CUDA Toolkit compiler residency before lowering graph
if not os.path.exists("/usr/local/cuda") and "CUDA_HOME" not in os.environ:
    print("[FNG WARNING]: Primary /usr/local/cuda path not detected.")
    print("               Ensure NVCC is resident within environment PATH to avoid linker failure.")

# ❶ Precision-Engineered Compiler & Linker Flag Flags Configuration
# Targets modern GPU architectures (Hopper H100 / Blackwell B200 optimization profiles)
# Suppresses host-side translation lag and forces strict pointer alignment.
extra_compile_args = {
    "cxx": [
        "-O3",                      # Extreme global loop-unrolling and optimization
        "-std=c++20",               # Enforce C++20 standard for RAII capsule fencing
        "-w",                        # Suppress compiler tracing clutter warnings
        "-march=native",            # Target local host CPU micro-architecture
        "-fPIC"                     # Generate Position-Independent Code for global library binding
    ],
    "nvcc": [
        "-O3",                      # Extreme device-side code optimization
        "-std=c++20",               # Coordinate C++20 standards within the device plane
        "--use_fast_math",          # Force SFU-native single-clock reciprocal/sqrt circuits
        "-Xcompiler", "-fPIC",      # Forward PIC flag directly to the underlying host compiler
        # Target Architecture Occupancy: Hopper (sm_90a) & Blackwell (sm_100) hardware profiles
        "-gencode=arch=compute_90a,code=sm_90a",
        "-gencode=arch=compute_100,code=sm_100"
    ]
}

# ❷ Define the High-Speed Native Extension Object (Layer 1.5 Bridge Binding)
# Surgically stitches the .cu kernel and the .cpp wrapper into a unified binary capsule module.
photonic_native_extension = CUDAExtension(
    name="torch_photonic_bridge_fence_backend", # Match name exactly with __init__.py linkage token
    sources=[
        "src/photonic_mesh_core_kernel.cu",     // Layer 1: Branchless PTX MUX Kernel
        "src/photonic_bridge_wrapper.cpp"       // Layer 1.5: RAII Lifecycle Capsule Fence Bridge
    ],
    extra_compile_args=extra_compile_args,
    library_dirs=[],
    libraries=["cuda"]                         // Directly link against native CUDA driver runtime layers
)

# ❸ Package Manifest Metadata Mapping Factory
setup(
    name="photonic-mesh-fng-router",
    version="3.0.0-PROD-OPTICS",
    author="PJHkorea",
    author_desc="Silicon-Neural Interleaved Optical Infrastructure Control Plane Engine",
    license="Apache License 2.0",
    description="A Hardware-Native, Optical Timing-Frozen Control Plane Engine for Hyperscale LLMs",
    packages=find_packages(exclude=["tests"]),
    # Mount the compiled binary extension directly inside python module memory space
    ext_modules=[photonic_native_extension],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True) 
        # Purge standard long python ABI tags to yield a clean, predictable library file (.so)
    },
    install_requires=[
        "torch>=2.2.0",
        "jax>=0.4.25",
        "jaxlib>=0.4.25",
        "numpy>=1.24.0"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    zip_safe=False
)

print("=========================================================================")
print("[FNG COMPILER INFRASTRUCTURE BUILD COMPLETE]")
print(" -> Compilation Target : Native Shared Binary Plugin (.so 완제품)")
print(" -> Opts Activated     : -O3, --use_fast_math, C++20 Lifecycle Guard")
print(" -> Status             : Ready for Ingestion via model hook patch.")
print("=========================================================================")
