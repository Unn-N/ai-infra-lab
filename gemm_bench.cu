// ============================================================
// gemm_bench.cu — 第 1 课：从 naive GEMM 到 shared memory tiling
//
// 目标卡: RTX 5070 (Blackwell, compute capability 12.0 -> -arch=sm_120)
// 注意: sm_120 需要 CUDA Toolkit >= 12.8，旧版本会报
//       "nvcc fatal: Unsupported gpu architecture 'compute_120'"
//
// 编译（Windows, 需在 "x64 Native Tools Command Prompt" 里）:
//     nvcc -O3 -arch=sm_120 gemm_bench.cu -o gemm_bench.exe -lcublas
// 编译（WSL / Linux）:
//     nvcc -O3 -arch=sm_120 gemm_bench.cu -o gemm_bench -lcublas
// 运行:
//     ./gemm_bench
//
// 学习目标（做完这三个你才算真的理解了，而不是抄完就跑）:
//   1. 用数字说话：GFLOP/s、相对 cuBLAS 的百分比、离 roofline 还有多远
//   2. naive 版为什么慢：每个线程从 global memory 读了多少次、浪费了多少带宽
//   3. tiling 为什么快：数据复用率、__syncthreads 的代价、+1 padding 的作用
// ============================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 32

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                        \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error: %d (%s:%d)\n", (int)_s, __FILE__,   \
                    __LINE__);                                                 \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ------------------------------------------------------------
// Kernel 1: naive
// 每个线程算 C 的一个元素，沿 K 方向做一次长度为 K 的点积。
// 问题：A 的一行被 N 个线程重复读，B 的一列被 M 个线程重复读。
//       没有任何复用，全部压力压到 global memory 上。
// ------------------------------------------------------------
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ------------------------------------------------------------
// Kernel 2: shared memory tiling
// 一个 block 负责 C 的一个 TILE x TILE 方块。
// 每轮迭代把 A 的一个 TILE x TILE 块和 B 的一个 TILE x TILE 块搬进
// shared memory，然后 block 内所有线程共享这份数据。
//
// 复用率：搬进 shared 的每个元素被 TILE 个线程用到，
//        global memory 访问量从 O(M*N*K) 降到 O((M+N)*K)。
//
// As[TILE][TILE+1] 的 +1：让同一列的元素落在不同的 bank 上，
// 避免 32 个线程同时访问 bank 冲突（改回 TILE 试试，会明显变慢）。
// ------------------------------------------------------------
__global__ __launch_bounds__(TILE * TILE)
void gemm_tiled(const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 协同加载：每个线程搬一个元素
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------
// 计时工具：CUDA event，比 host 侧计时准确得多
// ------------------------------------------------------------
template <typename F>
double time_kernel(F launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;  // 单次耗时(ms)
}

static void fill_random(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

static double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

int main() {
    // ---------- 设备信息 + roofline 参考 ----------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const double clock_ghz =
        static_cast<double>(prop.clockRate) / 1e6;  // kHz -> GHz
    // 粗估：每 SM 每周期 128 个 FP32 lane，FMA = 2 FLOP
    const double peak_fp32_tflops =
        static_cast<double>(prop.multiProcessorCount) * 128.0 * 2.0 * clock_ghz / 1000.0;
    // 显存带宽：DDR 每周期 2 次传输
    const double mem_bw_gbps =
        static_cast<double>(prop.memoryClockRate) * 2.0 * (prop.memoryBusWidth / 8.0) / 1e6;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ★ 关键：cuBLAS 在 Ampere 之后**默认允许把 FP32 降到 TF32 走 Tensor Core**。
    //   如果不关掉，你拿纯 FP32 的 kernel 去比，会觉得"我才到 cuBLAS 的 20%"，
    //   那不是你的实现差，是两边根本不在比同一个东西。
    //   CUBLAS_PEDANTIC_MATH = 关闭 TF32，保证是公平的 FP32 vs FP32。
    //   （想看 Tensor Core 的差距时，把这一行换成 CUBLAS_DEFAULT_MATH 再跑一次）
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    printf("======================================================================\n");
    printf("GPU      : %s\n", prop.name);
    printf("SM count : %d\n", prop.multiProcessorCount);
    printf("Compute  : %d.%d\n", prop.major, prop.minor);
    printf("Est. FP32 peak : %.1f TFLOP/s   (SMs x 128 x 2 x clock)\n", peak_fp32_tflops);
    printf("Est. HBM  BW   : %.1f GB/s\n", mem_bw_gbps);
    printf("======================================================================\n\n");

    printf("%6s %10s %12s %12s %12s %12s %8s\n",
           "N", "naive ms", "naive GF/s", "tiled GF/s", "cublas GF/s",
           "%cuBLAS", "maxerr");
    printf("----------------------------------------------------------------------\n");

    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; ++s) {
        const int M = sizes[s], N = sizes[s], K = sizes[s];

        float *dA, *dB, *dC, *dCref;
        CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCref, (size_t)M * N * sizeof(float)));

        fill_random(dA, (size_t)M * K);
        fill_random(dB, (size_t)K * N);

        const double flops = 2.0 * M * N * K;

        // ---- cuBLAS 参考值（行主序 -> 列主序，交换 A/B）----
        const float alpha = 1.0f, beta = 0.0f;
        auto launch_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, dB, N, dA, K, &beta,
                                     dCref, N));
        };

        // ---- tiled ----
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        auto launch_tiled = [&]() { gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K); };

        // ---- naive（大尺寸太慢，跳过）----
        bool run_naive = (M <= 1024);
        auto launch_naive = [&]() { gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K); };

        const int iters = (M <= 512) ? 50 : (M <= 1024 ? 20 : 10);

        double t_cublas = time_kernel(launch_cublas, 5, iters);
        double t_tiled  = time_kernel(launch_tiled,  5, iters);
        double t_naive  = run_naive ? time_kernel(launch_naive, 2, 5) : 0.0;

        // ---- 正确性检查：tiled vs cuBLAS ----
        launch_tiled();
        CUDA_CHECK(cudaDeviceSynchronize());
        double err = 0.0;
        {
            cudaEvent_t dummy; (void)dummy;
            // 拷回 host 逐元素比较（仅用于教学，大尺寸时略慢）
            float* hC = (float*)malloc((size_t)M * N * sizeof(float));
            float* hR = (float*)malloc((size_t)M * N * sizeof(float));
            CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            launch_cublas();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hR, dCref, (size_t)M * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            err = max_abs_diff(hC, hR, (size_t)M * N);
            free(hC);
            free(hR);
        }

        double gf_naive  = run_naive ? flops / t_naive  / 1e6 : 0.0;
        double gf_tiled  = flops / t_tiled  / 1e6;
        double gf_cublas = flops / t_cublas / 1e6;

        printf("%6d %10.2f %12.1f %12.1f %12.1f %12.1f %8.2e\n",
               N,
               run_naive ? t_naive : 0.0,
               gf_naive,
               gf_tiled,
               gf_cublas,
               gf_cublas > 0 ? 100.0 * gf_tiled / gf_cublas : 0.0,
               err);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dCref));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\n提示: %%cuBLAS 那一列就是你的「面试话术」——\n");
    printf("      「我手写的 tiled GEMM 在 N=4096 上达到 cuBLAS 的 XX%%」\n");
    return 0;
}
