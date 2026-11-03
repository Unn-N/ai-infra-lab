# ai-infra-lab

一个 GPU 算子的学习与优化实验台。目标不是"把算子写出来"，而是**知道自己离硬件上限还有多远、差在哪**。

---

## 目前做了什么

| 工作 | 状态 | 结果 |
|---|---|---|
| 逐行 softmax（朴素版） | ✅ | `8192×8192` 达 **85.5%** 显存带宽 |
| 逐行 softmax（warp 版） | ✅ | `N=128` 从 **19% → 93%**，**4.96×** |
| 正确性测试 | ✅ | 5 个用例，含非整齐 N / ±1000 极端值 / 退化行 |
| GEMM 基准 | 🚧 | 代码已写，未实测 |

### v2 的 4.96× 是怎么来的

拆成两个正交的改动，分别测：

| 改动 | 针对的瓶颈 | 收益 |
|---|---|---|
| 归约：shared 树形 → `__shfl_down_sync` | 屏障开销（9 次 → 0~2 次） | **1.63×** |
| 结构：一行 256 线程 → 一行 32 线程（多行共享 block） | 短行时一半 warp 空转 + 在飞请求不足 | **3.04×** |

`1.63 × 3.04 = 4.96`，和实测对得上。

---

## 三条结论（比代码值钱）

**1. "读三遍"不等于三倍显存流量。**
朴素版扫三遍，但一行只有 32KB，L2 全吸收了。真实流量就是读一遍写一遍——所以靠减少遍数省带宽的思路是错的。

**2. 短行和长行是两个不同的问题。**
长行撞带宽墙（85%，接近顶）；短行撞**延迟 + 协调**墙（19%，8 个 warp 有一半在屏障前空转）。同一个 kernel，两种病，药不一样。

**3. 小工作集的 `%spec` 会系统性偏高。**
写回被 L2 接住，直到下一次 flush 才落到显存——而 flush 不计时。所以小张量的带宽数字偏乐观，**只能同尺寸横比，不能跨尺寸纵比**。

细节见 [`notes/`](notes/)。

---

## 目录

```
softmax/
  softmax_v1.cu        朴素版：block 一行、shared 归约、带 L2 冲刷的计时框架
  softmax_v2.cu        优化版：多行/warp 归约，W=1/2/4/8 可切
  softmax_self.cu      自己重写的 v1
  softmax_v2_self.cu   自己重写的 v2（一个 warp 一行，零屏障）
  softmax_test.cu      正确性测试（5 个用例）
  hello_cuda.cu        环境自检
gemm_bench.cu          GEMM 基准（朴素 / tiled / cuBLAS 对照）
notes/                 实验记录
```

---

## 运行

需要 CUDA Toolkit ≥ 12.8（RTX 50 系需要 `sm_120`）。

```bash
cd softmax

nvcc -O3 -arch=sm_120 softmax_test.cu  -o softmax_test  && ./softmax_test
nvcc -O3 -arch=sm_120 softmax_v1.cu    -o softmax_v1    && ./softmax_v1
nvcc -O3 -arch=sm_120 softmax_v2.cu    -o softmax_v2    && ./softmax_v2 1
```

硬件：RTX 5070（48 SM / 12 GB GDDR7 / 峰值带宽 ≈ 672 GB/s）

---

## 待办

- [ ] `float4` 向量化：长行能不能从 85.5% 再往上走
- [ ] 拿到真实 DRAM 流量，修正 `%spec` 口径（WSL 下 profiler 受限）
- [ ] 把 v2 的两层归约拆成独立可测的中间版本
- [ ] GEMM 三版跑通并出对比
