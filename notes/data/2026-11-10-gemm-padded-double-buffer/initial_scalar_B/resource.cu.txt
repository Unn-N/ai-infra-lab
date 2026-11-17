#define GEMM_SELF_BENCH
#include "gemm_self_re.cu"

#ifndef GEMM_DYNAMIC_SHARED_BYTES
#define GEMM_DYNAMIC_SHARED_BYTES 0
#endif

int main(){

    // [Codex 测量适配] 仅启动前申请动态shared，设置本身不进入计时区间。
    if (GEMM_DYNAMIC_SHARED_BYTES > 0)
        CUDA_CHECK(cudaFuncSetAttribute(gemm_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       int(GEMM_DYNAMIC_SHARED_BYTES)));

    cudaFuncAttributes a{};cudaDeviceProp p{};int blocks=0;
    CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    CUDA_CHECK(cudaFuncGetAttributes(&a,gemm_tiled));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,gemm_tiled,BLOCK_THREADS,GEMM_DYNAMIC_SHARED_BYTES));
    printf("{\"output\":%d,\"thread_tile\":%d,\"threads\":%d,\"regs\":%d,\"static_shared\":%zu,\"dynamic_shared\":%zu,\"total_shared\":%zu,\"local\":%zu,\"blocks_per_SM\":%d,\"max_dynamic_shared\":%d,\"kernel_max_threads\":%d,\"SMs\":%d}\n",
           OUTPUT_TILE,THREAD_TILE,BLOCK_THREADS,a.numRegs,a.sharedSizeBytes,size_t(GEMM_DYNAMIC_SHARED_BYTES),
           a.sharedSizeBytes+size_t(GEMM_DYNAMIC_SHARED_BYTES),a.localSizeBytes,blocks,a.maxDynamicSharedSizeBytes,
           a.maxThreadsPerBlock,p.multiProcessorCount);
}
