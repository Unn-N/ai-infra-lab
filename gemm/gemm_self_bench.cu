// Codex 编写：测量辅助代码集中在此，gemm_self_re.cu 保留你的当前学习实现。
// 在仓库根目录编译、运行：
// /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self_bench.cu -o build/gemm_self_bench
// ./build/gemm_self_bench

// 直接引用同一份 kernel，只省略学习文件自己的 main，防止两个 main 冲突。
#define GEMM_SELF_BENCH
// [Codex 调整] 原版 gemm_self.cu 保留作对照，这里测正在修改的 re 版本。
#include "gemm_self_re.cu"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

// 计时前抽查 64 个输出，与 CPU double 点积比较；不是全矩阵正确性证明。
// 前 9 个点覆盖输出的首/中/末行列，剩余点分散在矩阵内。
static bool check_samples(const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& c, int M, int N, int K,
                          double& max_error) {
    const int rows[] = {0, M / 2, M - 1};
    const int cols[] = {0, N / 2, N - 1};
    max_error = 0;
    for (int s = 0; s < 64; ++s) {
        const int row = s < 9 ? rows[s / 3] : (s * 37 + 11) % M;
        const int col = s < 9 ? cols[s % 3] : (s * 71 + 13) % N;
        double reference = 0;
        for (int k = 0; k < K; ++k) {
            reference += double(a[size_t(row) * K + k]) * b[size_t(k) * N + col];
        }
        const float actual = c[size_t(row) * N + col];
        const double error = std::fabs(double(actual) - reference);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual) || error > 1e-4 + 1e-4 * std::fabs(reference)) {
            std::fprintf(stderr, "FAIL M=%d N=%d K=%d C[%d][%d]: got=%g ref=%g\n",
                         M, N, K, row, col, actual, reference);
            return false;
        }
    }
    return true;
}

int main() {
    // 学习用的固定观察范围：小/中/大方阵、长方形、非整齐边缘。
    // 后续比较改动时保持这张表不变，不宣称代表所有 GEMM 输入。
    const int shapes[][3] = {
        {128, 128, 128},
        {512, 512, 512},
        {2048, 2048, 2048},
        {128, 1024, 512},
        {127, 513, 257},
    };
    constexpr int warmup = 5;
    constexpr int repeats = 20;

    cudaDeviceProp gpu{};
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    std::printf("GPU: %s\n", gpu.name);
    std::printf("Mode: FP32, row-major, OUTPUT_TILE=%d K_TILE=%d THREAD_TILE=%d; same buffers, no L2 flush\n",
                OUTPUT_TILE, K_TILE, THREAD_TILE);
    std::printf("Timing: %d warmups; 3 rounds x %d launches; CUDA events\n", warmup, repeats);
    std::printf("Excluded: allocations, CPU/GPU copies, correctness checks\n");
    std::printf("Check: 64 sampled outputs vs CPU double, BEFORE timing\n\n");
    std::printf("    M     N     K   round1_ms   round2_ms   round3_ms   median_ms   GFLOP/s   sampled_maxerr\n");

    // 固定种子保证重复启动程序时生成相同的输入，包含正数和负数。
    std::mt19937 random(20261107);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (const auto& shape : shapes) {
        const int M = shape[0], N = shape[1], K = shape[2];
        std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N);
        for (float& x : hA) x = value(random);
        for (float& x : hB) x = value(random);

        const size_t bytesA = hA.size() * sizeof(float);
        const size_t bytesB = hB.size() * sizeof(float);
        const size_t bytesC = hC.size() * sizeof(float);
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, bytesA));
        CUDA_CHECK(cudaMalloc(&dB, bytesB));
        CUDA_CHECK(cudaMalloc(&dC, bytesC));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

        const dim3 block(OUTPUT_TILE / THREAD_TILE, OUTPUT_TILE / THREAD_TILE);
        const dim3 grid((N + OUTPUT_TILE - 1) / OUTPUT_TILE,
                        (M + OUTPUT_TILE - 1) / OUTPUT_TILE);
        auto launch = [&]() { gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K); };

        // 把输出先填成 NaN 的字节模式，抽查时也能发现没有写出的采样位置。
        CUDA_CHECK(cudaMemset(dC, 0xff, bytesC));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        double max_error = 0;
        if (!check_samples(hA, hB, hC, M, N, K, max_error)) return EXIT_FAILURE;

        // 预热和检查都不计时；计时也不调用会分配/拷贝数据的 run_gemm。
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::array<float, 3> times{};
        for (float& ms : times) {
            // 同一默认 stream 中用 event 包住连续 repeats 次启动。
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < repeats; ++i) launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= repeats;  // 一批的 GPU 时间 / 次数 = 每次平均耗时。
            if (!std::isfinite(ms) || ms <= 0) {
                std::fprintf(stderr, "Invalid timing result\n");
                return EXIT_FAILURE;
            }
        }
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        const float median_ms = sorted[1];
        // 采用 GEMM 常用计数：一次乘加按 2 FLOP，吞吐 = 2*M*N*K / 秒数。
        const double gflops = 2.0 * M * N * K / (median_ms * 1e6);
        std::printf("%5d %5d %5d %11.6f %11.6f %11.6f %11.6f %9.1f %16.2e\n",
                    M, N, K, times[0], times[1], times[2], median_ms, gflops, max_error);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::printf("\nAll sampled checks PASS. Timings describe this repeated-buffer workload only.\n");
    return EXIT_SUCCESS;
}
