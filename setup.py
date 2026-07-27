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

# ❶ Precision-Engineered Compiler & Linker Flag Configuration
# [리팩토링] 하위 호환성 및 다양한 환경에서의 성공률을 확보하기 위해 하드웨어 가드 명세를 확장합니다.
# NVCC 컴파일러 버전이 sm_100을 인식하지 못하더라도 안정적으로 빌드될 수 있도록 Ampere와 Ada 세대를 촘촘히 보충합니다.
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
        # Target Architecture Occupancy: 범용 가속기 환경(A100, H100, RTX4090)을 광범위하게 보호합니다.
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90a,code=sm_90a"
    ]
}

# ❷ Define the High-Speed Native Extension Object (Layer 1.5 Bridge Binding)
# Surgically stitches the .cu kernel and the .cpp wrapper into a unified binary capsule module.
photonic_native_extension = CUDAExtension(
    name="torch_photonic_bridge_fence_backend", # Match name exactly with __init__.py linkage token
    sources=[
        # [교정 완료] 실제 레포지토리 물리 구조에 맞춰 'src/' 경로 노이즈를 말끔히 청소했습니다.
        # 이로 인해 소스 주소 하이재킹 링커가 단번에 주소를 찾아 파일 컴파일을 개시합니다.
        "photonic_mesh_core_kernel.cu",     # // Layer 1: Branchless PTX MUX Kernel
        "photonic_bridge_wrapper.cpp"      # // Layer 1.5: RAII Lifecycle Capsule Fence Bridge
    ],
    extra_compile_args=extra_compile_args,
    library_dirs=[],
    libraries=["cuda"]                        # // Directly link against native CUDA driver runtime layers
)

# ❸ Package Manifest Metadata Mapping Factory
setup(
    name="photonic-mesh-fng-router",
    version="3.0.0-PROD-OPTICS",
    author="PJHkorea",
    license="Apache License 2.0",
    description="A Hardware-Native, Optical Timing-Frozen Control Plane Engine for Hyperscale LLMs",
    long_description="Silicon-Neural Interleaved Optical Infrastructure Control Plane Engine",
    long_description_content_type="text/plain",
    
    # [교정] 루트 디렉토리에 노출되어 있는 핵심 파이썬 소스 코드 자산들을 
    # 설치 타깃 모듈 명세에 완벽히 포함시켜 ModuleNotFoundError를 원천 차단합니다.
    py_modules=[
        "photonic_fng_orchestrator",
        "fng_fabric_monkey_patch",
        "test_photonic_pipeline"
    ],
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

