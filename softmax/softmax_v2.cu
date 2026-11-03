// ============================================================
// softmax_v2.cu — 多行共享一个 block + warp 内归约
//
// 编译: nvcc -O3 -arch=sm_120 softmax_v2.cu -o softmax_v2
// 运行:
//   ./softmax_v2        自动选择 W（N<=512 用 1，否则用 8）
//   ./softmax_v2 1       强制 W=1（一个 warp 一行）
//   ./softmax_v2 8       强制 W=8（一个 block 一行，v1 的形状）
//
// ------------------------------------------------------------
// v1 → v2 改了两件事，对应两个不同的瓶颈：
//
// 1) 一个 block 处理 R 行，而不是 1 行
//    设 W = 每行用几个 warp，则 R = 8 / W。
//      W=1 → R=8：每 warp 负责一整行，8 个 warp 全忙（N 小的时候不浪费线程）
//      W=8 → R=1：8 个 warp 合干一行（N 大的时候每线程元素数合理）
//    这就是"让 warp 不空转"的具体做法。
//
// 2) 归约从 shared 树形改成两层：
//    warp 内用 __shfl_down_sync（5 步，无屏障）
//    跨 warp 才用 shared + 2 次 __syncthreads（原来是 9 次）
//
// ------------------------------------------------------------
// 看什么：
//   N=128 那几行应该大幅改善（v1 只有 19%）
//   N=8192 那几行应该小幅改善（v1 已经 85%，只剩 barrier 的收益）
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>

#define WARPS_PER_BLOCK 8
#define BLOCK           (WARPS_PER_BLOCK * 32)

// ---------------- warp 内归约（无屏障）----------------
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------- 主 kernel ----------------
// W = 每行用几个 warp（必须能整除 8）
template <int W>
__global__ __launch_bounds__(BLOCK)
void softmax_rows(const float* __restrict__ x, float* __restrict__ y,
                  int M, int N) {
    constexpr int R = WARPS_PER_BLOCK / W;   // 每 block 处理几行

    const int warp = threadIdx.x >> 5;       // 0..7
    const int lane = threadIdx.x & 31;       // 0..31
    const int row_in_block = warp / W;       // 本 block 的第几行
    const int warp_in_row  = warp % W;       // 该行内的第几个 warp

    int row = blockIdx.x * R + row_in_block;
    // 尾部不足一个 block 时，夹到合法范围。
    // 不能直接 return —— 有人退出有人不退出会导致 __syncthreads 死锁。
    if (row >= M) row = M - 1;

    const float* xr = x + (long long)row * N;
    float* yr = y + (long long)row * N;

    const int stride = W * 32;
    const int start  = warp_in_row * 32 + lane;

    __shared__ float smem[WARPS_PER_BLOCK];   // 每个 warp 一个部分结果

    // ---------------- 第 1 遍：最大值 ----------------
    float m = -FLT_MAX;
    for (int i = start; i < N; i += stride) m = fmaxf(m, xr[i]);
    m = warp_max(m);
    m = __shfl_sync(0xffffffffu, m, 0);       // 广播到本 warp 所有 lane

    if (W > 1) {
        if (lane == 0) smem[warp] = m;
        __syncthreads();
        if (warp_in_row == 0) {
            float t = (lane < W) ? smem[row_in_block * W + lane] : -FLT_MAX;
            t = warp_max(t);
            if (lane == 0) smem[row_in_block * W] = t;
        }
        __syncthreads();
        m = smem[row_in_block * W];
    }

    // ---------------- 第 2 遍：指数和 ----------------
    float s = 0.0f;
    for (int i = start; i < N; i += stride) s += expf(xr[i] - m);
    s = warp_sum(s);
    s = __shfl_sync(0xffffffffu, s, 0);

    if (W > 1) {
        if (lane == 0) smem[warp] = s;
        __syncthreads();
        if (warp_in_row == 0) {
            float t = (lane < W) ? smem[row_in_block * W + lane] : 0.0f;
            t = warp_sum(t);
            if (lane == 0) smem[row_in_block * W] = t;
        }
        __syncthreads();
        s = smem[row_in_block * W];
    }

    // ---------------- 第 3 遍：写回 ----------------
    const float inv = 1.0f / s;
    for (int i = start; i < N; i += stride) yr[i] = expf(xr[i] - m) * inv;
}

// ---------------- 参考实现 ----------------
static void cpu_softmax_row(const float* x, float* y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i) m = std::fmax(m, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(static_cast<double>(x[i]) - m);
    for (int i = 0; i < n; ++i)
        y[i] = static_cast<float>(std::exp(static_cast<double>(x[i]) - m) / s);
}

// ---------------- 计时（带 L2 冲刷）----------------
__global__ void flush_l2(int* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = 0;
}

template <typename F>
double time_flushed(F launch, int warmup, int iters,
                    int* d_flush, size_t flush_i) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        flush_l2<<<2048, 256>>>(d_flush, flush_i);
        cudaEventRecord(a);
        launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        total += ms;
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return total / iters;
}

int main(int argc, char** argv) {
    int forceW = 0;
    if (argc > 1) forceW = std::atoi(argv[1]);
    if (forceW != 0 && forceW != 1 && forceW != 2 && forceW != 4 && forceW != 8) {
        printf("W 只能是 1/2/4/8\n");
        return 1;
    }

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    const double spec_bw = 672.0;

    const size_t flush_n = (size_t)64 * 1024 * 1024;
    int* d_flush = nullptr;
    cudaMalloc(&d_flush, flush_n * sizeof(int));

    printf("=================================================================\n");
    printf("GPU   : %s   SMs=%d\n", p.name, p.multiProcessorCount);
    printf("spec  : %.0f GB/s\n", spec_bw);
    printf("mode  : %s\n", forceW ? "强制 W" : "自动（N<=512 用 W=1，否则 W=8）");
    printf("=================================================================\n\n");

    printf("%6s %6s %3s %3s %10s %9s %9s %8s %7s %9s\n",
           "M", "N", "W", "R", "time(ms)", "GB/s(2x)", "GB/s(4x)", "%spec",
           "ideal", "maxerr");
    printf("-------------------------------------------------------------------------------\n");

    const int Ms[] = {512, 2048, 8192};
    const int Ns[] = {128, 512, 2048, 8192};

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 4; ++b) {
            const int M = Ms[a], N = Ns[b];
            const size_t n = (size_t)M * N;
            const size_t bytes = n * sizeof(float);

            const int W = forceW ? forceW : (N <= 512 ? 1 : 8);
            const int R = WARPS_PER_BLOCK / W;
            const dim3 grid((M + R - 1) / R);

            float *dx, *dy;
            cudaMalloc(&dx, bytes);
            cudaMalloc(&dy, bytes);

            float* hx = (float*)malloc(bytes);
            for (size_t i = 0; i < n; ++i) hx[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;
            cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

            auto launch = [&]() {
                switch (W) {
                    case 1: softmax_rows<1><<<grid, BLOCK>>>(dx, dy, M, N); break;
                    case 2: softmax_rows<2><<<grid, BLOCK>>>(dx, dy, M, N); break;
                    case 4: softmax_rows<4><<<grid, BLOCK>>>(dx, dy, M, N); break;
                    default: softmax_rows<8><<<grid, BLOCK>>>(dx, dy, M, N); break;
                }
            };

            const int iters = (n <= (1u << 22)) ? 20 : 8;
            const double ms = time_flushed(launch, 3, iters, d_flush, flush_n);

            launch();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                printf("[FAIL] M=%d N=%d: %s\n", M, N, cudaGetErrorString(e));
                return 1;
            }

            float* hy = (float*)malloc(bytes);
            cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);

            double maxerr = 0.0;
            float* ref = (float*)malloc(N * sizeof(float));
            const int rows = (M <= 64) ? M : 16;
            for (int r = 0; r < rows; ++r) {
                const int row = (M <= 64) ? r : (r * (M / rows));
                cpu_softmax_row(hx + (long long)row * N, ref, N);
                for (int i = 0; i < N; ++i) {
                    double d = std::fabs((double)hy[(long long)row * N + i] - ref[i]);
                    if (d > maxerr) maxerr = d;
                }
            }

            const double gbps2 = 8.0 * n / (ms * 1e6);
            const double gbps4 = 16.0 * n / (ms * 1e6);
            const double ideal = 8.0 * n / (spec_bw * 1e9) * 1e3;

            printf("%6d %6d %3d %3d %10.4f %9.1f %9.1f %7.1f%% %7.4f %9.2e\n",
                   M, N, W, R, ms, gbps2, gbps4,
                   100.0 * gbps2 / spec_bw, ideal, maxerr);

            free(hx); free(hy); free(ref);
            cudaFree(dx); cudaFree(dy);
        }
    }

    cudaFree(d_flush);
    printf("\nW=每行用几个 warp，R=每 block 处理几行（W*R = 8）\n");
    printf("对比 v1：N=128 原来 ~19%%，N=8192 原来 ~85%%\n");
    return 0;
}
