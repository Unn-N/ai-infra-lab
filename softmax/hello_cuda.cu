// 环境自检：最小可验证程序
//
// 编译: nvcc -O3 -arch=sm_120 hello_cuda.cu -o hello_cuda
// 运行: ./hello_cuda
//
// 它能一次验证三件事:
//   1. 编译通过  -> 工具链认识 sm_120（否则 nvcc 直接报 unsupported architecture）
//   2. 运行通过  -> 驱动能加载 kernel（否则报 "no kernel image is available"）
//   3. 打印属性  -> 确认 WSL 真的把 5070 透传进来了

#include <cstdio>
#include <cuda_runtime.h>

__global__ void touch(int* out) {
    if (threadIdx.x == 0) out[0] = 42;
}

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        printf("[FAIL] cudaGetDevice: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        printf("[FAIL] cudaGetDeviceProperties\n");
        return 1;
    }

    printf("name      = %s\n", p.name);
    printf("compute   = %d.%d\n", p.major, p.minor);
    printf("SMs       = %d\n", p.multiProcessorCount);
    printf("globalMem = %.1f GB\n", p.totalGlobalMem / 1e9);

    int* d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        printf("[FAIL] cudaMalloc\n");
        return 1;
    }

    touch<<<1, 32>>>(d);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("[FAIL] kernel launch: %s\n", cudaGetErrorString(e));
        printf("       常见原因: -arch 与显卡不匹配(需要 sm_120), 或驱动过旧\n");
        return 1;
    }

    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("readback  = %d\n", h);
    printf("\n[OK] 环境可用\n");
    return 0;
}
