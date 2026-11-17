#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>
#include <vector>
#define GEMM_SELF_BENCH
namespace reg4 {
#include "reg4/gemm_self_re.cu"
}
#undef GEMM_DYNAMIC_SHARED_BYTES
#undef CUDA_CHECK
namespace outer {
#include "outer/gemm_self_re.cu"
}
#undef GEMM_DYNAMIC_SHARED_BYTES
#undef CUDA_CHECK
namespace inner {
#include "inner/gemm_self_re.cu"
}
#undef GEMM_DYNAMIC_SHARED_BYTES
#undef CUDA_CHECK

#define CHECK(expr) do { auto s=(expr); if(s!=cudaSuccess){std::fprintf(stderr,"%s line %d\n",cudaGetErrorString(s),__LINE__);std::exit(1);} }while(0)
int main(){
    constexpr int M=2048,N=2048,K=2048,repeats=200;
    CHECK(cudaFuncSetAttribute(outer::gemm_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,82944));
    CHECK(cudaFuncSetAttribute(inner::gemm_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,82944));
    std::mt19937 rng(20261107);std::uniform_real_distribution<float> value(-1,1);
    for(size_t i=0;i<size_t(128)*128*2+size_t(512)*512*2;++i)(void)value(rng);
    std::vector<float>a(size_t(M)*K),b(size_t(K)*N),c(size_t(M)*N);
    for(float&x:a)x=value(rng);for(float&x:b)x=value(rng);
    float *da,*db,*dc;CHECK(cudaMalloc(&da,a.size()*4));CHECK(cudaMalloc(&db,b.size()*4));CHECK(cudaMalloc(&dc,c.size()*4));
    CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    auto launch=[&](int v){
        if(v==0)reg4::gemm_tiled<<<dim3(32,32),dim3(16,16)>>>(da,db,dc,M,N,K);
        if(v==1)outer::gemm_tiled<<<dim3(16,16),dim3(16,16),82944>>>(da,db,dc,M,N,K);
        if(v==2)inner::gemm_tiled<<<dim3(16,16),dim3(16,16),82944>>>(da,db,dc,M,N,K);
    };
    const char*names[]={"reg4","outer","inner"};
    for(int v=0;v<3;++v){
        CHECK(cudaMemset(dc,0xff,c.size()*4));launch(v);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(c.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));
        double worst=0;
        for(int s=0;s<64;++s){int ends[]={0,M/2,M-1};int r=s<9?ends[s/3]:(s*37+11)%M;int q=s<9?ends[s%3]:(s*71+13)%N;
            double ref=0;for(int k=0;k<K;++k)ref+=double(a[size_t(r)*K+k])*b[size_t(k)*N+q];
            double e=std::fabs(double(c[size_t(r)*N+q])-ref);worst=std::max(worst,e);
            if(!std::isfinite(c[size_t(r)*N+q])||e>1e-4+1e-4*std::fabs(ref))return 2;
        }
        std::printf("CHECK %s PASS maxerr=%g\n",names[v],worst);
        for(int i=0;i<20;++i)launch(v);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    }
    cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
    // 六种排列各用两次，避免固定先后顺序；一轮中每版200次，比较间隔约0.3秒。
    const int orders[6][3]={{0,1,2},{2,1,0},{1,0,2},{2,0,1},{0,2,1},{1,2,0}};
    for(int round=0;round<12;++round)for(int pos=0;pos<3;++pos){
        int v=orders[round%6][pos];float ms;
        CHECK(cudaEventRecord(start));for(int i=0;i<repeats;++i)launch(v);CHECK(cudaEventRecord(stop));
        CHECK(cudaGetLastError());CHECK(cudaEventSynchronize(stop));CHECK(cudaEventElapsedTime(&ms,start,stop));
        std::printf("TIME %d %s %.6f\n",round,names[v],ms/repeats);std::fflush(stdout);
    }
    CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(dc));
}
