

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void flush_l2(int* p, size_t n) {
    size_t i            = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        p[i] = 0;
}

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

__global__ void test(float* out) {
    float v = (float)threadIdx.x;
    float r = reduce_sum(v);
    if (threadIdx.x == 0) out[0] = r;
}

static void cpu_softmax_row(const float* x, float* y, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; ++i)
        m = fmaxf(m, x[i]);

    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += exp((double)x[i] - m);

    for (int i = 0; i < n; ++i)
        y[i] = (float)(exp((double)x[i] - m) / s);
}

int main() {
    const int M = 2, N = 3;

    // ① 准备输入（自己定值，方便手算）
    float hx[M * N] = {1, 2, 3, 4, 5, 6};

    // ② 在主机上算一遍"标准答案"
    float href[M * N];
    for (int r = 0; r < M; ++r)
        cpu_softmax_row(hx + r * N, href + r * N, N);

    // ③ 搬到显存
    float *dx, *dy;
    cudaMalloc(&dx, M * N * sizeof(float));
    cudaMalloc(&dy, M * N * sizeof(float));
    cudaMemcpy(dx, hx, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // ④ 跑
    softmax_rows_naive<<<M, BLOCK>>>(dx, dy, M, N);
    cudaDeviceSynchronize();

    // ⑤ 搬回来
    float hgot[M * N];
    cudaMemcpy(hgot, dy, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ⑥ 逐个对比
    for (int i = 0; i < M * N; ++i)
        printf("i=%d  got=%.6f  ref=%.6f  %s\n", i, hgot[i], href[i],
               fabsf(hgot[i] - href[i]) < 1e-6f ? "ok" : "MISMATCH");

    // ⑦ 收尾
    cudaFree(dx);
    cudaFree(dy);
    return 0;
}