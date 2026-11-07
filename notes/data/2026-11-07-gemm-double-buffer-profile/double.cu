#define GEMM_SELF_BENCH
#include "../../gemm/gemm_self_double_buffer.cu"
#include <vector>
#include <random>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int M=std::atoi(argv[1]), N=std::atoi(argv[2]), K=std::atoi(argv[3]);
    std::vector<float> a(size_t(M)*K), b(size_t(K)*N);
    std::mt19937 rng(20261107);
    std::uniform_real_distribution<float> value(-1.0f,1.0f);
    // Match the input sequence of the unchanged five-shape benchmark for 2048^3.
    for (size_t i=0; i<2*size_t(128)*128 + 2*size_t(512)*512 + (M==128 ? 2*size_t(2048)*2048 : 0); ++i) (void)value(rng);
    for(float& x:a) x=value(rng);
    for(float& x:b) x=value(rng);
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db,b.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc,size_t(M)*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
    const dim3 block(OUTPUT_TILE/THREAD_TILE,OUTPUT_TILE/THREAD_TILE);
    const dim3 grid((N+OUTPUT_TILE-1)/OUTPUT_TILE,(M+OUTPUT_TILE-1)/OUTPUT_TILE);
    // Six unprofiled calls match validation + five warmups; collect only call seven.
    for(int i=0;i<7;++i) {
        gemm_tiled<<<grid,block>>>(da,db,dc,M,N,K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float first=0;
    CUDA_CHECK(cudaMemcpy(&first,dc,sizeof(float),cudaMemcpyDeviceToHost));
    std::printf("Profile target M=%d N=%d K=%d, config=%d/%d/%d, C[0]=%g\n",
                M,N,K,OUTPUT_TILE,K_TILE,THREAD_TILE,first);
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
}
