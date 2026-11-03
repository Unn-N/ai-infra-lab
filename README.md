# ai-infra-lab

AI Infra 方向的个人实验台。记录每一个 kernel、每一次测量、每一个想清楚的问题。

## 目录

```
ai-infra-lab/
├── ROADMAP.md        # 大二到秋招的整体路线
├── gemm_bench.cu     # 第 1 课：naive vs tiled vs cuBLAS，带 roofline 参考
└── notes/            # 每轮实验的结论（最重要的部分）
```

## 先确认环境

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvcc --version
```

`compute_cap` 那一列就是下面编译命令里 `-arch=sm_XX` 要填的数字。

> **本机是 RTX 5070 → `compute_cap = 12.0` → `-arch=sm_120`**
>
> 配套要求，缺一个就会卡住：
> - **CUDA Toolkit ≥ 12.8**（12.7 及更早不认识 `sm_120`，会直接报 unsupported architecture）
> - **驱动 ≥ 570**（`nvidia-smi` 右上角看）
> - 后面装 PyTorch 时要选 **cu128 及以上**的 wheel，`cu121` 的轮子在 50 系上跑不起来
> - Ninetoothed 依赖 Triton，Triton 3.x 之后才正式支持 Blackwell；版本太老会编译失败

## 编译

**Windows**（要用 "x64 Native Tools Command Prompt for VS"，普通 cmd 找不到 `cl.exe`）：

```bat
nvcc -O3 -arch=sm_120 gemm_bench.cu -o gemm_bench.exe -lcublas
```

**WSL / Linux**：

```bash
nvcc -O3 -arch=sm_120 gemm_bench.cu -o gemm_bench -lcublas
```

**WSL 下如果 nvcc 报找不到 GPU**：确认装了 `nvidia-cuda-toolkit`，并且 WSL 版本 ≥ 2（`wsl --version`）。

## 运行

```
gemm_bench
```

输出示例（RTX 5070：48 SM，12GB GDDR7，理论 FP32 峰值 ≈ 31 TFLOP/s，带宽 ≈ 672 GB/s）：

```
GPU      : NVIDIA GeForce RTX 5070
SM count : 48
Compute  : 12.0
Est. FP32 peak : 30.8 TFLOP/s
Est. HBM  BW   : 672.0 GB/s
     N   naive ms  naive GF/s   tiled GF/s  cublas GF/s    %cuBLAS   maxerr
   256       3.21         10.5        380.2       1420.6       26.8   2.1e-05
  1024      62.40         34.4       3100.5      13800.0       22.5   4.7e-05
  4096          -            -      16500.0      24800.0       66.5   1.1e-04
```

> 上面是我按 5070 的规格**估算**的量级，不是实测。你的真实数字一定不同——**以你自己跑出来的为准**。
>
> 两个值得留意的点：
> - `N=256` 时即使 tiled 也只到峰值的 ~1%，因为小矩阵的运算强度 (`N/6 ≈ 43 FLOP/byte`) 刚好低于这张卡的机器平衡点 (`31 TFLOPS / 672 GB/s ≈ 46 FLOP/byte`)，这时候是**带宽受限**，再优化 tiling 也没用
> - `cublas` 那一列是 `CUBLAS_PEDANTIC_MATH`（纯 FP32）。如果你把代码里那行改成 `CUBLAS_DEFAULT_MATH`，数字会跳一大截——那才是 TF32 Tensor Core 的真实速度


## 三个必做的小实验（做完再往下走）

改代码、重跑、对比，把结论写进 `notes/`。

1. **拆掉 padding**
   把 `As[TILE][TILE + 1]` 改成 `As[TILE][TILE]`，重跑。
   预期：明显变慢。想清楚为什么——32 个线程同时访问同一列 = 同一个 bank。
   *产出：一句话解释 bank 冲突 + 一组前后对比数字。*

2. **换 TILE 大小**
   试 `TILE = 16 / 32 / 64`。64 的时候注意 `blockDim = 64×64 = 4096` 超了 1024 线程上限，需要改成每线程算 2×2 个元素。
   *产出：tile 大小与性能的关系曲线，说明拐点来自哪里（shared memory 容量？occupancy？）*

3. **算复用率**
   手算：naive 版处理一个 1024³ 的 GEMM，总共从 global memory 读了多少字节？tiled 版呢？
   用 `mem_bw` 算一下 naive 版理论上最少要多少时间，和实测差多少。
   *产出：这就是你第一次真正用上 roofline 的地方。*

## 下一步

- [ ] `notes/gemm.md`：把上面三个实验的结论写下来
- [ ] 加一版 register blocking（每线程算 4×4 个元素）
- [ ] 加一版 tensor core（`mma.sync`，FP32→TF32 或 FP16）
- [ ] profile：`ncu --set full ./gemm_bench`，看懂 `sm__throughput` 和 `dram__throughput`
