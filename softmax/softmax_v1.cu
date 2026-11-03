// ============================================================
// softmax_v1.cu — 逐行 Softmax 的朴素基线（v1）
//
// 编译: nvcc -O3 -arch=sm_120 softmax_v1.cu -o softmax_v1
// 运行: ./softmax_v1
//
// ------------------------------------------------------------
// 变化：
//   a. 测量：加 L2 冲刷，带宽按 2 次访问算（见下）
//   b. 实验：加 exp 消融开关，用 -DNO_EXP / -DFAST_EXP 切换
//
// 第一版跑出了 >100% 的带宽占比，那是不可能的。原因有两个：
//   1) 小尺寸的整个张量常驻 L2，50 次迭代测的是 L2 不是显存
//   2) 一行的数据（<=32KB）在三遍之间也在 L2 里，
//      所以"读三遍"根本没有变成三倍的 DRAM 流量
//
// 修法：
//   a. 每次迭代前用 256MB 的 buffer 把 L2 冲掉（缓存失效的标准做法）
//   b. 带宽按"不可再少的 2 次访问"算，另附 4 次访问的值作对比
//   c. 打印工作集大小，让你一眼看出哪些行根本不在测显存
// ============================================================

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256

// ---------------- 消融开关 ----------------
// 一次只改这一个变量，其余代码完全相同，测出来才能比较。
//
//   默认        : expf      —— 基准，结果正确
//   -DFAST_EXP  : __expf    —— 硬件快速指数，结果仍然正确（精度略降）
//   -DNO_EXP    : x * 0.5f  —— 纯粹为了测时间下界，结果故意是错的
#ifdef NO_EXP
#define SM_EXP(x) ((x) * 0.5f)
#define SM_EXP_NAME "none (x*0.5)  <- 结果错误，只测时间"
#elif defined(FAST_EXP)
#define SM_EXP(x) __expf(x)
#define SM_EXP_NAME "__expf"
#else
#define SM_EXP(x) expf(x)
#define SM_EXP_NAME "expf"
#endif

// ---------------- L2 冲刷 ----------------
__global__ void flush_l2(int* p, size_t n) {
    size_t i            = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        p[i] = 0;
}

// ---------------- block 内归约 ----------------
__device__ float reduce_max(float v) {
    __shared__ float s[BLOCK];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
        }
        __syncthreads();
    }
    return s[0];
}

__device__ float reduce_sum(float v) {
    __shared__ float s[BLOCK];
    s[threadIdx.x] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            s[threadIdx.x] += s[threadIdx.x + off];
        }
        __syncthreads();
    }
    return s[0];
}

// ---------------- v1 kernel（与第一版完全相同）----------------
__global__ void softmax_rows_naive(const float* __restrict__ x, float* __restrict__ y, int M,
                                   int N) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const float* xr  = x + (long long)row * N;
    float* yr        = y + (long long)row * N;
    const int stride = static_cast<int>(blockDim.x);

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += stride)
        m = fmaxf(m, xr[i]);
    m = reduce_max(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += stride)
        s += SM_EXP(xr[i] - m);
    s = reduce_sum(s);

    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < N; i += stride)
        yr[i] = SM_EXP(xr[i] - m) * inv;
}

// ---------------- 正确性对照 ----------------
static void cpu_softmax_row(const float* x, float* y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i)
        m = std::fmax(m, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += std::exp(static_cast<double>(x[i]) - m);
    for (int i = 0; i < n; ++i) {
        y[i] = static_cast<float>(std::exp(static_cast<double>(x[i]) - m) / s);
    }
}

// ---------------- 每次迭代前冲刷 L2 的计时 ----------------
template <typename F>
double time_flushed(F launch, int warmup, int iters, int* d_flush, size_t flush_i,
                    int flush_blocks) {
    for (int i = 0; i < warmup; ++i)
        launch();
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        flush_l2<<<flush_blocks, 256>>>(d_flush, flush_i); // 不计入时间
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

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);

    // CUDA 13 移除了 clockRate / memoryClockRate /
    // memoryBusWidth，直接用厂商规格。 RTX 5070 (GB205)：12GB GDDR7，192-bit，28
    // Gbps  ->  ~672 GB/s
    const double spec_bw_gbps = 672.0;

    // 256MB 冲刷缓冲，明显大于 50 系的 L2
    const size_t flush_n = (size_t)64 * 1024 * 1024; // 64M ints = 256MB
    int* d_flush         = nullptr;
    cudaMalloc(&d_flush, flush_n * sizeof(int));

    printf("=================================================================\n");
    printf("GPU            : %s\n", p.name);
    printf("compute        : %d.%d   SMs=%d\n", p.major, p.minor, p.multiProcessorCount);
    printf("spec bandwidth : %.0f GB/s\n", spec_bw_gbps);
    printf("L2 flush       : 256 MB / iteration\n");
    printf("exp mode       : %s\n", SM_EXP_NAME);
    printf("=================================================================\n\n");

    printf("%6s %6s %9s %9s %9s %8s %8s %9s %9s\n", "M", "N", "workset", "time(ms)", "GB/s(2x)",
           "%spec", "GB/s(4x)", "ideal_ms", "maxerr");
    printf("---------------------------------------------------------------------"
           "------\n");

    const int Ms[] = {512, 2048, 8192};
    const int Ns[] = {128, 512, 2048, 8192};

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 4; ++b) {
            const int M = Ms[a], N = Ns[b];
            const size_t n     = (size_t)M * N;
            const size_t bytes = n * sizeof(float);

            float *dx = nullptr, *dy = nullptr;
            cudaMalloc(&dx, bytes);
            cudaMalloc(&dy, bytes);

            float* hx = (float*)malloc(bytes);
            for (size_t i = 0; i < n; ++i)
                hx[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;
            cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

            auto launch     = [&]() { softmax_rows_naive<<<M, BLOCK>>>(dx, dy, M, N); };
            const int iters = (n <= (1u << 22)) ? 20 : 8;
            const double ms = time_flushed(launch, 3, iters, d_flush, flush_n, 2048);

            // ---- 正确性 ----
            launch();
            cudaDeviceSynchronize();
            float* hy = (float*)malloc(bytes);
            cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);

            double maxerr        = 0.0;
            float* ref           = (float*)malloc(N * sizeof(float));
            const int check_rows = (M <= 64) ? M : 16;
            for (int r = 0; r < check_rows; ++r) {
                const int row = (M <= 64) ? r : (r * (M / check_rows));
                cpu_softmax_row(hx + (long long)row * N, ref, N);
                for (int i = 0; i < N; ++i) {
                    double d = std::fabs((double)hy[(long long)row * N + i] - ref[i]);
                    if (d > maxerr) maxerr = d;
                }
            }

            const double ws_mb    = 2.0 * bytes / 1e6;     // 读 + 写
            const double gbps2    = 8.0 * n / (ms * 1e6);  // 不可再少的 2 次访问
            const double gbps4    = 16.0 * n / (ms * 1e6); // 若三遍全部 miss cache
            const double ideal_ms = 8.0 * n / (spec_bw_gbps * 1e9) * 1e3;

            char errbuf[32];
#ifdef NO_EXP
            snprintf(errbuf, sizeof(errbuf), "%s", "n/a");
#else
            snprintf(errbuf, sizeof(errbuf), "%.2e", maxerr);
#endif

            printf("%6d %6d %8.1fM %9.4f %9.1f %7.1f%% %9.1f %9.4f %9s\n", M, N, ws_mb, ms, gbps2,
                   100.0 * gbps2 / spec_bw_gbps, gbps4, ideal_ms, errbuf);

            free(hx);
            free(hy);
            free(ref);
            cudaFree(dx);
            cudaFree(dy);
        }
    }

    cudaFree(d_flush);

    printf("\n--- 现在这份表该怎么读 ---\n");
    printf("workset : 读+写的总量。远小于 L2(约 40MB)的行，即使加了 flush,\n");
    printf("          单次迭代内部仍有大量数据复用，速度会偏高\n");
    printf("GB/s(2x): 假设 kernel 只搬一遍读一遍。这个才是这张卡的实话\n");
    printf("%%spec   : 应 <= 100%%。若仍 >100%%，说明还有缓存效应没被解释\n");
    printf("GB/s(4x): 按朴素模型的 3 读 1 写算。它和 2x 的差距，就是 L2\n");
    printf("          帮你省掉的重复读——这个差距越大，说明\"砍遍数\"越没用\n");
    printf("ideal_ms: 2 次访问下的理论下限\n");
    return 0;
}
