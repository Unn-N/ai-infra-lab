#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 32

// ===== Codex 补充：CUDA 错误检查 =====
// CUDA API 会返回状态码。失败时打印调用位置并退出，避免带着错误继续运行。
// do/while (0) 让这个多行宏可以像普通的一条语句那样使用。
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t status = (expr);                                             \
        if (status != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                     \
                         cudaGetErrorString(status), __FILE__, __LINE__);         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)
// ===== Codex 补充结束 =====

__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    // [Codex 修正] 与下面 As[ty][tx]、Bs[ty][tx] 的布局对应：
    // y/ty 决定输出行，x/tx 决定输出列；同一 ty 的线程合作搬 A 的同一行。
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// [Codex 说明] 下列 run_gemm 的流程由你起草；我整理了格式、给已有 API
// 加上 CUDA_CHECK，并在标记处补齐执行检查、结果拷回和释放显存。
// 调用前：hA、hB 已填好，hC 有 M*N 个 float 的空间；M、N、K 为正数。
void run_gemm(const float* hA, const float* hB, float* hC, int M, int N, int K) {
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytesA = size_t(M) * K * sizeof(float);
    const size_t bytesB = size_t(K) * N * sizeof(float);
    const size_t bytesC = size_t(M) * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);

    // ===== Codex 补充：完成一次调用 =====
    // 检查启动错误（例如不合法的启动配置），再等待 GPU 完成，检查执行错误。
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // kernel 把输出写进 dC；这里把结果复制到调用者准备的 CPU 数组 hC。
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // 结果已经取回，释放本次调用申请的三块显存。hA/hB/hC 由调用者管理。
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    // ===== Codex 补充结束 =====
}

// ===== Codex 补充：最小测试入口 =====
// 只验证一次矩阵乘法；先不加入随机填值、计时或 cuBLAS 对照。
// [Codex 补充] 测量入口引用本文件时省略这个 main；单独编译仍运行小例子。
#ifndef GEMM_SELF_BENCH
int main() {
    const int M = 2, N = 2, K = 3;

    // CPU 输入按行连续存放：A 为 2*3，B 为 3*2。
    const float hA[M * K] = {1, 2, 3,
                            4, 5, 6};
    const float hB[K * N] = {7,  8,
                            9, 10,
                           11, 12};
    float hC[M * N] = {};  // 给拷回的结果预留 CPU 空间。

    // 手算参考值：例如 C[0][0] = 1*7 + 2*9 + 3*11 = 58。
    const float expected[M * N] = {58,  64,
                                 139, 154};

    // 这是 CPU 上的普通函数调用；真正的 GPU 启动语句在 run_gemm 内部。
    run_gemm(hA, hB, hC, M, N, K);

    bool passed = true;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const int i = row * N + col;
            // 浮点结果通常按容差比较；非有限数也必须判为失败。
            const bool ok = std::isfinite(hC[i]) &&
                            std::fabs(hC[i] - expected[i]) < 1e-5f;
            std::printf("C[%d][%d] = %.1f, expected %.1f: %s\n",
                        row, col, hC[i], expected[i], ok ? "OK" : "FAIL");
            passed = passed && ok;
        }
    }
    std::printf("GEMM correctness: %s\n", passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
// ===== Codex 补充结束 =====
