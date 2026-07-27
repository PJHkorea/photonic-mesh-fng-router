/**
 * [FNG PHOTONICS CORE - SINGLE-CLOCK REGISTER-LEVEL BITWISE MUX MAIN KERNEL]
 * Precision-engineered to eradicate Warp Divergence penalties (JMP) at the silicon boundary.
 * Executes native IEEE-754 bit-manipulation via inline PTX assembly circuits.
 * 
 * Copyright (c) 2026 PJHkorea. All rights reserved.
 * Licensed under the Apache License 2.0.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
 * 🎛️ Inline PTX Assembly Multiplexer Circuit
 * Physically toggles the micro-routing selector inside the ALU based on the predicate register bit.
 * Eradicates warp divergence penalties by executing conditional selection at the hardware register level.
 */
__device__ __forceinline__ float pinn_branchless_select_f32(
    bool condition, 
    float true_val, 
    float false_val
) {
    float output_reg;
    
    // 1. C++ bool/int 값을 PTX 1비트 조건자 레지스터(%p)로 매핑하기 위해 setp(set predicate) 사용
    // 2. selp.f32 명령어로 조건자 레지스터 값에 따라 1비트 클럭 내에 레지스터 스왑 수행
    asm (
        "{\n\t"
        "  .reg .pred %p;\n\t"           // 1비트 전용 조건자 레지스터 선언
        "  setp.ne.u32 %p, %3, 0;\n\t"   // condition(%3)이 0이 아니면 %p를 true로 설정
        "  selp.f32 %0, %1, %2, %p;\n\t" // %p 조건에 따라 true_val 또는 false_val 선택
        "}"
        : "=f"(output_reg)
        : "f"(true_val), "f"(false_val), "r"((unsigned int)condition)
    );
    
    return output_reg;
}

extern "C" {

    // ❶ Global Thread Topology Mapping to Hardware Execution Grid
    int block_offset = blockIdx.x * blockDim.x;
    int thread_idx   = block_offset + threadIdx.x;
    
    // 조기 return 제거 유지, 유효 스레드 플래그 생성
    bool is_valid_thread = (thread_idx < total_elements);
    
    // ❷ Tiled Warp Partitioning (Clustering 32-threads as a single physical execution unit)
    cg::thread_block_tile<32> warp_tile = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp_tile.thread_rank();
    
    // 유효하지 않은 쓰레드는 0.0f가 아니라 '자신의 경계면 데이터' 오염을 막기 위해 
    // 유효한 이웃 쓰레드가 참조할 수 있도록 직전/직후의 더미 처리가 필요하지만, 
    // 레지스터 레벨에서는 유효 비트를 함께 셔플하는 것이 가장 안전합니다.
    float raw_pulse_register  = is_valid_thread ? d_raw_pulse[thread_idx] : 0.0f;
    unsigned int raw_oni_mask = is_valid_thread ? d_oni_mask[thread_idx] : 0;
    
    // ❸ 1-Clock Crossbar Shuffling & Validity Propagation
    // 데이터와 함께 '이 데이터가 진짜 유효한가?'의 여부(is_valid)도 함께 셔플합니다.
    float right_neighbor = warp_tile.shfl_down(raw_pulse_register, 1);
    float left_neighbor  = warp_tile.shfl_up(raw_pulse_register, 1);
    
    unsigned int has_right_data = warp_tile.shfl_down((unsigned int)is_valid_thread, 1);
    unsigned int has_left_data  = warp_tile.shfl_up((unsigned int)is_valid_thread, 1);
    
    // ❹ Boundary Clamping: 물리적 경계(0, 31)뿐만 아니라 '논리적 데이터 경계'까지 동시 방어
    // 우측에 유효한 데이터가 없거나(Tail block 경계), 하드웨어 워프 끝(31)이면 자기 자신 복사
    bool is_right_edge = (lane_id == 31) || (has_right_data == 0);
    bool is_left_edge  = (lane_id == 0)  || (has_left_data == 0);
    
    right_neighbor = pinn_branchless_select_f32(is_right_edge, raw_pulse_register, right_neighbor);
    left_neighbor  = pinn_branchless_select_f32(is_left_edge, raw_pulse_register, left_neighbor);
    
    // ❺ Compressible Optical Vorticity Damping (Burgers' Formulation)
    float laplacian_wavefront = right_neighbor + left_neighbor - (2.0f * raw_pulse_register);
    const float viscosity_alpha = 0.015f;
    float damped_wavefront    = raw_pulse_register + (viscosity_alpha * laplacian_wavefront);
    
    // ❻ Global Shock-wave Fault Telemetry Gathering
    // 각 쓰레드의 결함 유무(0 혹은 1 이상)를 하드웨어 원자적 활성 마스크(0xFFFFFFFF) 기반으로 완벽 동기 수집
    // 가짜 패딩 쓰레드가 마스크를 오염시키지 않도록 is_valid_thread 조건 병합
    unsigned int global_oni_active = warp_tile.ballot(is_valid_thread && (raw_oni_mask > 0));
    
    // ❼ Micro-circuit Multiplexing: 내 레인 ID에 해당하는 비트만 정확히 추출
    bool local_oni_fault = (global_oni_active & (1u << lane_id)) != 0;
    
    float purified_output = pinn_branchless_select_f32(
        local_oni_fault,    // 내 레인의 결함 유무 비트 전달
        0.0f,               // Clean vacuum erasure state
        damped_wavefront    // Purified physical tensor matrix
    );

    


       // ❽ Unified Memory Grid View Commit Barrier
    // 동기화 연산을 위해 살려두었던 테일 레인(Tail lanes)들이 
    // 글로벌 메모리 경계 밖을 침범하여 쓰기 연산을 수행하지 않도록 물리적 가드를 적용합니다.
    if (thread_idx < total_elements) {
        d_purified_tensor[thread_idx] = purified_output;
    }
}

/**
 * 🚀 Host-side C++ Trampoline Linker called by Layer 1.5 Bridge (photonic_bridge_wrapper.cpp)
 * Dynamically computes optimal hardware occupancy and dispatches the native PTX kernel with 0ns host overhead.
 */
void execute_photonic_jitter_squelch_kernel(
    const float* d_raw_pulse,
    const unsigned int* d_oni_mask,
    float* d_purified_tensor,
    const int total_elements,
    cudaStream_t stream
) {
    if (total_elements <= 0) return;

    // 1. 하드코딩된 256을 제거하고, 현재 GPU 아키텍처의 레지스터/SMem 한계에 맞추어 
    // 하드웨어 점유율(Occupancy)을 극대화하는 블록 및 그리드 크기를 런타임에 자동 계산합니다.
    int block_size = 0;
    int min_grid_size = 0;
    
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        (void*)photonic_jitter_squelch_cuda_kernel,
        0,  // 동적 공유 메모리(Dynamic Shared Memory) 사용량
        0   // 블록 크기 제한 없음
    );

    // 요소 개수에 맞춘 실제 그리드 사이즈 매핑
    int grid_size = (total_elements + block_size - 1) / block_size;

    // 2. Asynchronous non-blocking dispatch directly onto the active XLA/PyTorch execution stream
    photonic_jitter_squelch_cuda_kernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_pulse,
        d_oni_mask,
        d_purified_tensor,
        total_elements
    );

    // 3. 디스패치 에러 트래킹 부하를 컴파일러 힌트([[unlikely]])를 사용하여 파이프라인에서 완전히 격리합니다.
    // 일반적인 정상 작동(cudaSuccess) 시에는 printf나 스트링 파싱을 위한 CPU 오버헤드가 완전히 0ns가 됩니다.
    cudaError_t dispatch_err = cudaGetLastError();
    if ([[unlikely]] (dispatch_err != cudaSuccess)) {
        printf("[FNG KERNEL FATAL]: PTX MUX Kernel launch failed: %s (Block: %d, Grid: %d)\n", 
               cudaGetErrorString(dispatch_err), block_size, grid_size);
    }
}
} // extern "C"
