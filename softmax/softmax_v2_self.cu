#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP  32
#define WARPS_PER_BLOCK (BLOCK / WARP)   // 8

// 冲刷缓存：用无关数据把 L2 填满，让被测 kernel 从冷缓存开始。
// 这个任务可以无脑高并发，所有线程都参与。
__global__ void flush(int* p, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;  // 步长就是所有线程数量
    size_t i = (size_t)threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    for (; i < n; i += stride) {
        p[i] = 0;
    }
}

// warp 内规约：不经过任何内存，直接读别的 lane 的寄存器。
__device__ __forceinline__ float reduce_max(float v) {
    for (int off = WARP / 2; off > 0; off >>= 1) {
        // 32 个 lane 全部执行，但只有前 off 个的结果会被用到
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    // 归约完只有 lane 0 是对的，必须广播
    return __shfl_sync(0xffffffffu, v, 0);
}

// warp 内求和，结构同 reduce_max
__device__ __forceinline__ float reduce_sum(float v) {
    for (int off = WARP / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return __shfl_sync(0xffffffffu, v, 0);
}

// ------------------------------------------------------------------
// W = 每行用几个 warp（1 / 2 / 4 / 8，必须整除 WARPS_PER_BLOCK）
//   W=1 → 一个 warp 一行，零屏障（短行最优）
//   W=8 → 八个 warp 合干一行（长行最优）
//
// 归约分两层：warp 内用 shfl（无屏障），跨 warp 才用 shared + 2 次屏障。
// ------------------------------------------------------------------
__global__ void softmax_rows(float* __restrict__ x, float* __restrict__ y, int M, int N, int W) {
    // 1. 我是第几个 warp、warp 里第几个 lane
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int R = WARPS_PER_BLOCK / W;   // 每 block 处理几行
    const int row_in_block = warp / W;   // 本 block 的第几行
    const int warp_in_row  = warp % W;   // 该行内的第几个 warp

    // 2. 我负责哪一行（尾部不足一个 block 时夹取，不能 return，否则屏障死锁）
    int row = blockIdx.x * R + row_in_block;
    if (row >= M) row = M - 1;

    // 3. 这一行的起点
    const float* xr = x + (long long)row * N;
    float* yr       = y + (long long)row * N;

    // 4. 一行由 W 个 warp 分，所以跨越 W*32 个线程
    const int stride = W * 32;
    const int start  = warp_in_row * 32 + lane;

    __shared__ float partial[WARPS_PER_BLOCK];  // 每个 warp 一格，跨 warp 汇总用

    // ---------------- 第 1 遍：最大值 ----------------
    float m = -FLT_MAX;
    for (int i = start; i < N; i += stride) {
        m = fmaxf(m, xr[i]);
    }
    m = reduce_max(m);  // warp 内合并

    if (W > 1) {  // 跨 warp 合并。W 对整个 block 相同，所以屏障不会有人缺席
        if (lane == 0) partial[warp] = m;
        __syncthreads();

        if (warp_in_row == 0) {  // 每行的第一个 warp 负责汇总本行的 W 个 partial
            float t = (lane < W) ? partial[row_in_block * W + lane] : -FLT_MAX;
            t = reduce_max(t);
            if (lane == 0) partial[row_in_block * W] = t;
        }
        __syncthreads();
        m = partial[row_in_block * W];
    }

    // ---------------- 第 2 遍：指数和 ----------------
    float s = 0.0f;
    for (int i = start; i < N; i += stride) {
        s += expf(xr[i] - m);
    }
    s = reduce_sum(s);

    if (W > 1) {
        if (lane == 0) partial[warp] = s;
        __syncthreads();

        if (warp_in_row == 0) {
            float t = (lane < W) ? partial[row_in_block * W + lane] : 0.0f;
            t = reduce_sum(t);
            if (lane == 0) partial[row_in_block * W] = t;
        }
        __syncthreads();
        s = partial[row_in_block * W];
    }

    // ---------------- 第 3 遍：写回 ----------------
    const float inv = 1.0f / s;
    for (int i = start; i < N; i += stride) {
        yr[i] = expf(xr[i] - m) * inv;
    }
}

// ------------------------------------------------------------------
// dispatcher：按行宽选 W
//   N 小 → 一行喂不满多个 warp，让一个 warp 独占一行（多行并行）
//   N 大 → 一行足够长，拆给多个 warp（行内并行）
// ------------------------------------------------------------------
inline int softmax_pick_w(int N) {
    // 实测：W=2 这一档没有收益，直接二选一
    return (N <= 512) ? 1 : 8;
}

inline int softmax_grid(int M, int W) {
    const int R = WARPS_PER_BLOCK / W;
    return (M + R - 1) / R;
}
