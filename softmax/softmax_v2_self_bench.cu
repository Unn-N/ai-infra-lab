// ============================================================
// softmax_v2_self_bench.cu — 给 warp 版 kernel 做性能测量
//
// 编译: nvcc -O3 -arch=sm_120 softmax_v2_self_bench.cu -o bench_v2self
// 运行:
//   ./bench_v2self        按 N 自动选 W（dispatcher）
//   ./bench_v2self 1      强制 W=1（对照）
//   ./bench_v2self 8      强制 W=8
//
// 只负责测量。被测 kernel 从 softmax_v2_self.cu 引入，避免两份代码不同步。
// ============================================================

#ifndef KERNEL_FILE
#define KERNEL_FILE "softmax_v2_self.cu"
#endif
#include KERNEL_FILE

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------- L2 冲刷 ----------------
__global__ void bench_flush_l2(int* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = 0;
}

// ---------------- 参考实现 ----------------
static void cpu_softmax_row(const float* x, float* y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i) m = std::fmax(m, x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double)x[i] - m);
    for (int i = 0; i < n; ++i) y[i] = (float)(std::exp((double)x[i] - m) / s);
}

// ---------------- 计时（每轮先冲 L2，只夹住 kernel）----------------
template <typename F>
double time_flushed(F launch, int warmup, int iters, int* d_flush, size_t flush_i, int flush_blocks) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    double total = 0.0;
    for (int i = 0; i < iters; ++i) {
        bench_flush_l2<<<flush_blocks, 256>>>(d_flush, flush_i);  // 不计时
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

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);

    const double spec_bw = 672.0;  // RTX 5070，换卡改这里
    const size_t flush_n = (size_t)64 * 1024 * 1024;  // 256MB
    int* d_flush = nullptr;
    cudaMalloc(&d_flush, flush_n * sizeof(int));

    printf("=================================================================\n");
    printf("GPU     : %s   SMs=%d\n", p.name, p.multiProcessorCount);
    printf("spec BW : %.0f GB/s\n", spec_bw);
    printf("mode    : %s\n", forceW ? "强制 W" : "自动（N<=512→1, 否则 8）");
    printf("=================================================================\n\n");

    printf("%6s %6s %3s %3s %9s %9s %9s %8s %9s\n", "M", "N", "W", "R", "time(ms)", "GB/s", "%spec",
           "ideal", "maxerr");
    printf("-----------------------------------------------------------------------------\n");

    const int Ms[] = {512, 2048, 8192};
    const int Ns[] = {128, 512, 2048, 8192};

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 4; ++b) {
            const int M = Ms[a], N = Ns[b];
            const size_t n = (size_t)M * N;
            const size_t bytes = n * sizeof(float);

            const int W = forceW ? forceW : softmax_pick_w(N);
            const int grid = softmax_grid(M, W);

            float *dx = nullptr, *dy = nullptr;
            cudaMalloc(&dx, bytes);
            cudaMalloc(&dy, bytes);

            float* hx = (float*)malloc(bytes);
            for (size_t i = 0; i < n; ++i) hx[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;
            cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

            auto launch = [&]() { softmax_rows<<<grid, BLOCK>>>(dx, dy, M, N, W); };

            const int iters = (n <= (1u << 22)) ? 20 : 8;
            const double ms = time_flushed(launch, 3, iters, d_flush, flush_n, 2048);

            launch();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                printf("[FAIL] M=%d N=%d : %s\n", M, N, cudaGetErrorString(e));
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

            const double gbps = 8.0 * n / (ms * 1e6);
            const double ideal = 8.0 * n / (spec_bw * 1e9) * 1e3;

            printf("%6d %6d %3d %3d %9.4f %9.1f %8.1f%% %8.4f %9.2e\n", M, N, W, WARPS_PER_BLOCK / W,
                   ms, gbps, 100.0 * gbps / spec_bw, ideal, maxerr);

            free(hx);
            free(hy);
            free(ref);
            cudaFree(dx);
            cudaFree(dy);
        }
    }

    cudaFree(d_flush);
    printf("\nGB/s 按“读一遍 + 写一遍”算（L2 吸收重复读）\n");
    return 0;
}
