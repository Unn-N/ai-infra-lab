# 07 · GEMM 的简单测量入口

2026-11-07，按用户要求由 Codex 编写。测量辅助代码集中在 `gemm/gemm_self_bench.cu`，直接引用 `gemm_self.cu` 的同一份 kernel；学习文件仅给原有 main 增加条件编译开关，没有改动计算逻辑、host 调用流程或小例子。

## 使用

在仓库根目录运行：

```bash
mkdir -p build
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self_bench.cu -o build/gemm_self_bench
./build/gemm_self_bench
```

原来的 `gemm_self.cu` 仍可单独编译运行其手算小例子。更改被测 kernel 后需要重新编译 bench。

## 固定的观察范围和计时口径

- FP32，按行连续存储，`C = A * B`；初始版本 `TILE=32`。
- 五组 `(M,N,K)`：`(128,128,128)`、`(512,512,512)`、`(2048,2048,2048)`、`(128,1024,512)`、`(127,513,257)`。它们用于学习中观察规模、形状和边缘差异，不代表所有输入。
- 固定随机种子生成含正负数的输入，每次运行程序保持相同输入。
- 每组检查后再预热 5 次；随后测 3 轮，每轮用 CUDA events 包住连续 20 次 kernel 启动，以整轮耗时除以 20。三轮平均耗时的中位数作为本次代表值，原始三轮数据也打印出来。
- 分配、CPU/GPU 拷贝、CPU 参考检查和预热均在计时之外；计时直接启动 kernel，不计整个 `run_gemm`。
- 同一组 GPU 缓冲区重复使用，**不冲刷 L2，允许缓存复用**。这是重复调用下的观察，不能当作冷缓存性能，也不能直接与旧 Softmax 的冲缓存数字比较。
- event 测的是同一 stream 中一批启动的经过时间。小 kernel 的批次平均值可能包含启动间隙，不应理解为完全剥离所有调度开销的纯指令执行时间。
- 每组计时前抽查 64 个输出，含首/中/末行列和内部位置，以 CPU double 点积作参考；容差 `1e-4 + 1e-4*abs(reference)`，拒绝 NaN/Inf。**抽查通过不是全矩阵或全部输入的正确性证明。**
- 吞吐量按 `2*M*N*K` FLOP 除以耗时计算。不估算“距离硬件上限百分比”，不预先断言瓶颈。

后续先固定这张尺寸表与计时口径，只比较同一行输入的改动前后结果；观察改善和退步，再决定是否需要不同实现或配置。桌面 GPU 的三轮结果也可能波动，不把小差异直接解释为优化收益。

计时依据：[NVIDIA CUDA events 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams)。

## 首次基线与验证

学习文件 SHA-256：`0306184a4d13de625b8262c9ed1bf1d0b4702025a8764f9e4707f64c3207eb39`。本次在未提交的工作区编译运行；原始完整输出也保存在忽略的 `build/gemm_self_baseline_2026-11-07.txt`。

```text
GPU: NVIDIA GeForce RTX 5070
Mode: FP32, row-major, TILE=32; same buffers, no L2 flush
Timing: 5 warmups; 3 rounds x 20 launches; CUDA events
Excluded: allocations, CPU/GPU copies, correctness checks
Check: 64 sampled outputs vs CPU double, BEFORE timing

    M     N     K   round1_ms   round2_ms   round3_ms   median_ms   GFLOP/s   sampled_maxerr
  128   128   128    0.008318    0.010187    0.007930    0.008318     504.2         2.93e-06
  512   512   512    0.137587    0.113054    0.112840    0.113054    2374.4         7.11e-06
 2048  2048  2048    6.769493    6.700034    6.682069    6.700034    2564.1         3.19e-05
  128  1024   512    0.058533    0.058398    0.058318    0.058398    2298.3         9.63e-06
  127   513   257    0.024261    0.024730    0.024077    0.024261    1380.3         4.76e-06

All sampled checks PASS. Timings describe this repeated-buffer workload only.
```

原有手算小例子仍通过。另用临时副本故意把 kernel 输出改成 0，确认测量程序在首个尺寸的 CPU 对照处报错，以非零状态退出且不打印性能行；临时验证文件位于忽略的 `build/`，没有修改真实 kernel。

以上数据是一次本机测量基线，不是性能优化结论，也没有证明当前版本的瓶颈。下一步一次研究一项改动，并保留逐尺寸结果。


## 当前工作副本变更

2026-11-07 下午，用户新建 `gemm_self_re.cu` 并要求先拆分尺寸定义。测量入口目前引用该 re 文件，打印 `OUTPUT_TILE`、`K_TILE`、`THREAD_TILE`；初始配置为 32、32、1，计算流程仍与此前基线一致。本文上方的首次输出、哈希和 TILE 实验数据均保留为当时版本的历史记录。当前配置限制和下一步见 [学习续记](06-gemm-tiling.md)。
