#define GEMM_SELF_BENCH
#include "gemm_self_re.cu"

#ifndef GEMM_DYNAMIC_SHARED_BYTES
#define GEMM_DYNAMIC_SHARED_BYTES 0
#endif

#include <cuda_profiler_api.h>
#include <vector>
#include <random>
int main(int argc,char** argv) {
    // [Codex 测量适配] 仅启动前申请动态shared，设置本身不进入计时区间。
    if (GEMM_DYNAMIC_SHARED_BYTES > 0)
        CUDA_CHECK(cudaFuncSetAttribute(gemm_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       int(GEMM_DYNAMIC_SHARED_BYTES)));

    if(argc!=4) return 2;
    const int M=std::atoi(argv[1]),N=std::atoi(argv[2]),K=std::atoi(argv[3]);
    std::vector<float>a(size_t(M)*K),b(size_t(K)*N);
    std::mt19937 rng(20261107);
    std::uniform_real_distribution<float> value(-1.0f,1.0f);
    // Match the unchanged five-shape benchmark input stream for the 2048 target.
    for(size_t i=0;i<2*size_t(128)*128+2*size_t(512)*512;++i)(void)value(rng);
    for(float& x:a)x=value(rng);
    for(float& x:b)x=value(rng);
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db,b.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc,size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
    const dim3 block(BLOCK_SIDE,BLOCK_SIDE);
    const dim3 grid((N+OUTPUT_TILE-1)/OUTPUT_TILE,(M+OUTPUT_TILE-1)/OUTPUT_TILE);
    // Validation-equivalent call plus twenty warmups; only call 22 is profiled.
    for(int i=0;i<22;++i){
        if(i==21)CUDA_CHECK(cudaProfilerStart());
        gemm_tiled<<<grid,block,GEMM_DYNAMIC_SHARED_BYTES>>>(da,db,dc,M,N,K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaProfilerStop());
    float first=0;
    CUDA_CHECK(cudaMemcpy(&first,dc,sizeof(float),cudaMemcpyDeviceToHost));
    printf("Profile %d,%d,%d config=%d/%d/%d C[0]=%g\n",M,N,K,OUTPUT_TILE,K_TILE,THREAD_TILE,first);
    CUDA_CHECK(cudaFree(da));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dc));
}
