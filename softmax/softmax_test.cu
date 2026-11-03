// ============================================================
// softmax_test.cu — 逐行 softmax 的正确性测试
//
// 编译: nvcc -O3 -arch=sm_120 softmax_test.cu -o softmax_test
// 运行: ./softmax_test
//
// ------------------------------------------------------------
// 一个好的算子测试，不是"跑通就行"，而是用不同的输入去**逼出错误**。
// 这里四个用例，各打一个薄弱点：
//
//   1. 小规模随机   —— 基础对拍，错了最容易定位
//   2. 非整齐 N     —— N 不是 32/256 的倍数，边界处理容易错
//   3. 极端数值     —— 含 1000 和 -1000，不减最大值会溢出成 inf/nan
//   4. 整行全相等   —— 所有 exp 相同，检查归一化是否退化
//
// 除了逐元素对拍，还独立检查两条性质：
//   · 每行的和必须 = 1（softmax 的定义，与参考实现无关）
//   · 输出里不允许出现 NaN 或 Inf
//
// 后两条比"和参考实现一致"更强：参考实现也可能一起错。
// ============================================================

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256

// ---------------- 被测代码 ----------------
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

__global__ void softmax_rows_naive(const float* __restrict__ x, float* __restrict__ y, int M,
                                   int N) {
    const int row = blockIdx.x;

    const float* xr  = x + (long long)row * N;
    float* yr        = y + (long long)row * N;
    const int stride = static_cast<int>(blockDim.x);

    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += stride) {
        m = fmaxf(m, xr[i]);
    }
    m = reduce_max(m);

    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += stride) {
        s += expf(xr[i] - m);
    }
    s = reduce_sum(s);

    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < N; i += stride) {
        yr[i] = expf(xr[i] - m) * inv;
    }
}

// ---------------- 参考实现（主机侧，用 double）----------------
static void cpu_softmax_row(const float* x, float* y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i) m = std::fmax(m, x[i]);

    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double)x[i] - m);

    for (int i = 0; i < n; ++i) y[i] = (float)(std::exp((double)x[i] - m) / s);
}

// ---------------- 三种造数据的方式 ----------------
static void fill_random(float* p, size_t n, unsigned seed) {
    std::srand(seed);
    for (size_t i = 0; i < n; ++i) p[i] = (float)std::rand() / RAND_MAX * 8.0f - 4.0f;
}

static void fill_extreme(float* p, size_t n, int N) {
    // 每行的第一个元素给 1000，第二个给 -1000，其余随机小值。
    // 不减最大值的话，exp(1000) 直接溢出成 inf。
    std::srand(12345);
    for (size_t i = 0; i < n; ++i) {
        switch (i % N) {
            case 0: p[i] = 1000.0f; break;
            case 1: p[i] = -1000.0f; break;
            default: p[i] = (float)std::rand() / RAND_MAX * 2.0f - 1.0f; break;
        }
    }
}

static void fill_equal(float* p, size_t n, float v) {
    for (size_t i = 0; i < n; ++i) p[i] = v;
}

// ---------------- 一个用例 ----------------
static int run_case(const char* name, int M, int N, int mode) {
    const size_t n     = (size_t)M * N;
    const size_t bytes = n * sizeof(float);

    float* hx = (float*)malloc(bytes);
    float* href = (float*)malloc(bytes);
    float* hgot = (float*)malloc(bytes);

    switch (mode) {
        case 0: fill_random(hx, n, 42); break;
        case 1: fill_extreme(hx, n, N); break;
        default: fill_equal(hx, n, 0.5f); break;
    }

    // 参考结果
    for (int r = 0; r < M; ++r) cpu_softmax_row(hx + (size_t)r * N, href + (size_t)r * N, N);

    float *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

    softmax_rows_naive<<<M, BLOCK>>>(dx, dy, M, N);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("%-14s FAIL  启动/执行出错: %s\n", name, cudaGetErrorString(e));
        cudaFree(dx); cudaFree(dy);
        free(hx); free(href); free(hgot);
        return 1;
    }

    cudaMemcpy(hgot, dy, bytes, cudaMemcpyDeviceToHost);

    // 检查 1：与参考实现的最大绝对误差
    double max_abs_err = 0.0;
    int    nan_count   = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(hgot[i])) { ++nan_count; continue; }
        double d = std::fabs((double)hgot[i] - href[i]);
        if (d > max_abs_err) max_abs_err = d;
    }

    // 检查 2：每行的和必须等于 1（这条不依赖参考实现）
    double max_sum_err = 0.0;
    for (int r = 0; r < M; ++r) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += hgot[(size_t)r * N + i];
        double d = std::fabs(sum - 1.0);
        if (d > max_sum_err) max_sum_err = d;
    }

    // 容差：float 有 7 位有效数字，1e-5 是这类算子的常规标准
    const double tol = 1e-5;
    const bool ok = (nan_count == 0) && (max_abs_err < tol) && (max_sum_err < tol);

    printf("%-14s M=%-6d N=%-7d maxerr=%-10.2e rowsum_err=%-10.2e nan=%-4d %s\n", name, M, N,
           max_abs_err, max_sum_err, nan_count, ok ? "PASS" : "FAIL");

    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(href);
    free(hgot);
    return ok ? 0 : 1;
}

int main() {
    printf("=== softmax_rows_naive 正确性测试 ===\n\n");

    int fails = 0;
    fails += run_case("小规模随机", 2, 8, 0);          // 能手算的最小场景
    fails += run_case("非整齐N", 17, 1000, 0);         // 1000 不能被 256 整除
    fails += run_case("极端数值", 8, 64, 1);           // 含 ±1000，测数值稳定
    fails += run_case("全相等行", 4, 32, 2);           // 退化情况
    fails += run_case("单元素行", 3, 1, 0);            // N=1 的边界

    printf("\n%s\n", fails == 0 ? "全部通过" : "有失败用例");
    return fails;
}
