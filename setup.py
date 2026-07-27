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
# [리팩토링 완료] nvcc 표준 옵션 대시 규격 타이포를 수정하고, 글로벌 상용 가속기 노드 배포 범용성을 확보합니다.
extra_compile_args = {
    "cxx": [
        "-O3",                      # Extreme global loop-unrolling and optimization
        "-std=c++20",               # Enforce C++20 standard for RAII capsule fencing
        "-w",                       # Suppress compiler tracing clutter warnings
        "-march=native",            # Target local host CPU micro-architecture
        "-fPIC"                     # Generate Position-Independent Code for global library binding
    ],
    "nvcc": [
        "-O3",                      # Extreme device-side code optimization
        "--std=c++20",              # [★교정★] nvcc 파서 전용 더블 대시(--) 명세로 컴파일러 충돌 완전 제거
        "--use_fast_math",          # Force SFU-native single-clock reciprocal/sqrt circuits
        "-Xcompiler", "-fPIC",      # Forward PIC flag directly to the underlying host compiler
        # Target Architecture Occupancy: 범용 가속기 환경(A100, H100, RTX4090)을 완벽하게 수용합니다.
        "-gencode=arch=compute_80,code=sm_80",   # Ampere (A100 / GA100)
        "-gencode=arch=compute_86,code=sm_86",   # Ampere (RTX 3090 / A6000)
        "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace (RTX 4090 / L40)
        "-gencode=arch=compute_90,code=sm_90"    # [★교정★] Hopper 글로벌 범용 호환 코드 명세로 정렬 (H100)
    ]
}


# ❷ Define the High-Speed Native Extension Object (Layer 1.5 Bridge Binding)
# Surgically stitches the .cu kernel and the .cpp wrapper into a unified binary capsule module.
# [★교정★] 익스텐션 모듈의 네임스페이스를 파이썬 패키지 내부 가상 디렉터리 경로명과 무결하게 동기화합니다.
# 이로 인해 링커가 바이너리를 엉뚱한 루트 공간에 버려두지 않고 패키지 내부 버스에 완벽하게 일체화시킵니다.
photonic_native_extension = CUDAExtension(
    name="photonic_mesh_fng_router.torch_photonic_bridge_fence_backend", 
    sources=[
        "photonic_mesh_core_kernel.cu",     # Layer 1: Branchless PTX MUX Kernel
        "photonic_bridge_wrapper.cpp"       # Layer 1.5: RAII Lifecycle Capsule Fence Bridge
    ],
    extra_compile_args=extra_compile_args,
    library_dirs=[],
    libraries=["cuda"]                        # Directly link against native CUDA driver runtime layers
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
    
    # [★교정★] 개별 py_modules 탑레이어 노출 대신, 컴파일된 바이너리와 파이썬 오케스트레이터 자산들이 
    # 'photonic_mesh_fng_router'라는 단일 통일 통합 패키지 네임스페이스 공간으로 묶이도록 유도합니다.
    # 이 구조적 매핑 전환을 통해 인스톨 즉시 발생하는 모든 ModuleNotFoundError 오탐지를 원천 차단합니다.
    packages=find_packages(),
    
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
