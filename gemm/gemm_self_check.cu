// Codex 整理：独立正确性入口，逐个输出与 CPU double 点积比较。
// nvcc -O3 -arch=sm_120 gemm/gemm_self_check.cu -o build/gemm_self_check
// ./build/gemm_self_check
// 覆盖当前已测的小尺寸、非方阵、4×4 小块边缘和跨输出 tile 的情况。
#define GEMM_SELF_BENCH
// [Codex 测量辅助] 默认测 re 版本；编译时可指定独立实验文件。
#ifndef GEMM_SELF_SOURCE
#define GEMM_SELF_SOURCE "gemm_self_re.cu"
#endif
#include GEMM_SELF_SOURCE
#include <vector>

int main() {
    const int cases[][3] = {{1, 1, 1}, {32, 32, 32}, {37, 45, 67}, {64, 33, 96}, {63, 65, 33}, {65, 67, 31}, {129, 130, 35}};
    for (const auto& shape : cases) {
        const int M = shape[0], N = shape[1], K = shape[2];
        std::vector<float> a(size_t(M) * K), b(size_t(K) * N), c(size_t(M) * N);
        for (size_t i = 0; i < a.size(); ++i) a[i] = (int(i % 17) - 8) * 0.125f;
        for (size_t i = 0; i < b.size(); ++i) b[i] = (int(i % 13) - 6) * 0.25f;
        run_gemm(a.data(), b.data(), c.data(), M, N, K);
        double max_error = 0;
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                double reference = 0;
                for (int k = 0; k < K; ++k)
                    reference += double(a[size_t(row) * K + k]) * b[size_t(k) * N + col];
                const float actual = c[size_t(row) * N + col];
                const double error = std::fabs(double(actual) - reference);
                if (!std::isfinite(actual) || error > 1e-5 + 1e-5 * std::fabs(reference)) {
                    std::printf("FAIL M=%d N=%d K=%d at (%d,%d): got=%g ref=%g\n",
                                M, N, K, row, col, actual, reference);
                    return 1;
                }
                if (error > max_error) max_error = error;
            }
        }
        std::printf("PASS M=%d N=%d K=%d max_abs_error=%g\n", M, N, K, max_error);
    }
    return 0;
}
