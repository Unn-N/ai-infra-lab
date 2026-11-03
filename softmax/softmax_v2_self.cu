#include <cstddef>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cfloat>

#define BLOCK 256
#define WARP 32

// 冲刷缓存，清除整个卡的所有残留数据，——这个任务可以无脑高并发，所有线程都可以参与
__global__ void flush(int* p, size_t n) {
    const size_t stride = (long long)gridDim.x * blockDim.x; // 步长就是所有线程数量
    size_t i = (long long)threadIdx.x +
               blockIdx.x * blockDim.x; // 线程起点就是该线程的线程编号，也就是某种相对位置
    for (; i < n; i += stride) {
        p[i] = 0;
    }
}
// wrap内规约,不经过上层memory,直接通过shfl读取别的线程自己的寄存器内部的值
__device__ __forceinline__ float reduce_max(float v) {
    for (int off = WARP / 2; off > 0; off >>= 1) { // 前一半线程干正事

        v = fmaxf(
            v,
            __shfl_down_sync(
                0xffffffffu, v,
                off)); // mask是一个16进制数字,也就代表了wrap内部的所有线程的1或0,ffffffff即代表所有线程都是1,全部参加.u为unsigned
    }
    return __shfl_sync(
        0xffffffffu, v,
        0); // 广播最后得到的最大值,也就是lane0所拥有的v的值,以防止错误结果.offset设为0即可广播wrap内全部线程
}
// wrap内规约得到线程内部分和,基本等同max函数
__device__ __forceinline__ float reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

__global__ void softmax_rows(float* __restrict__ x, float* __restrict__ y, int M, int N) {
    // 确认每个线程自己的身份,自己是第几个warp,第几个lane,
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    // 负责哪一行,
    int row = blockIdx.x * (BLOCK / WARP) + warp;
    if (row >= M) row = M - 1;
    // 从哪个地址开始,
    const float* xr = x + (long long)row * N;
    float* yr       = y + (long long)row * N;
    // 步长
    const int stride = 32;
    // 局部最大值,每一个线程得到自己负责的数据里的最大值
    float m = -FLT_MAX;
    for (int i = lane; i < N; i += stride) {
        m = fmaxf(m, xr[i]);
    }
    // warp内规约得到全行的max
    m = reduce_max(m);

    // 属于该线程的局部exp和
    float s = 0.0f;
    for (int i = lane; i < N; i += stride) {
        s += expf(xr[i] - m);
    }
    // warp规约得到全行的sum
    s = reduce_sum(s);
    // 写回
    float inv = 1.0f / s;
    for (int i = lane; i < N; i += stride) {
        yr[i] = expf(xr[i] - m) * inv;
    }
}