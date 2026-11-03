# AI Infra 路线图 · 大二 → 2029 届秋招

> 目标岗位：高性能算子 / 通信 / 编译器工程师（DeepSeek 及同类 AI Infra 团队）
> 起点（2026.10，大二上）：C++ 算法基础、已写过 CUDA、在读 Ninetoothed(Triton DSL) 源码
> 终点（2028.9 秋招）：3 个硬通货 + 一次大厂 infra 实习

---

## 核心判断

**缺的不是知识量，是「可量化的产出」。**

JD 里最难的从来不是 C++，是那句「只关心当前设计和实现距离理论上限还有多远」——这要求你能拿出**自己的数字**。所以整条路线的组织原则是：

> 每一个阶段都必须产出一个「可复现、有 benchmark、能讲 30 分钟」的东西。

---

## 里程碑

### M1 · 建立"写→测→优化"闭环（2026.10 – 2027.1，大二上 + 寒假）

| 产出 | 验收标准 |
|---|---|
| GEMM 三版：naive / shared-memory tiled / tensor-core(mma) | 在 N=4096 上达到 cuBLAS 的 **60%+** |
| 一份 benchmark 报告（Markdown + 曲线图） | 有 GFLOP/s、相对 roofline 的差距分析、bank conflict 前后对比 |
| 硬件体系笔记 | 能白板讲清 memory hierarchy / occupancy / roofline / bank conflict |

配套阅读：H&P 第 4–6 章；CUDA C Programming Guide 的 *Performance Guidelines* 全章。

**这一步的价值**：简历上出现「手写 CUDA GEMM，达 cuBLAS 的 X%」，你就从"学过"跨到"做过"。

---

### M2 · 进入核心算子（2027.2 – 2027.6，大二下）

| 产出 | 验收标准 |
|---|---|
| Flash Attention 前向实现 | 对比 PyTorch SDPA，达到 **50%+**；讲清 online softmax 的数值稳定性 |
| Softmax / RMSNorm / LayerNorm 优化版 | 达到 memory-bound roofline 的 **80%+**（这类算子是纯带宽瓶颈，好测好讲） |
| **Ninetoothed 一个 merged PR** | 真实合并，不是草稿；附生成 kernel 的性能对比 |

**关键**：这一阶段开始"两条腿走路"——一条手写 CUDA，一条 DSL 编译器。前者证明你懂硬件，后者证明你懂抽象。**同时具备这两条的人极少。**

---

### M3 · 第一个实习（2027.7 – 2027.9，大二暑假）

目标不必是大厂核心组，能碰到真实算子就行：AI 芯片公司（寒武纪/壁仞/摩尔线程）、推理框架团队、InfiniTensor 生态内的公司。

**投递前自检**：简历上有没有一个能被追问 30 分钟的 kernel 项目？没有就先别投，去补。

---

### M4 · 补第二层：通信 + 编译器（2027.9 – 2028.3，大三上）

| 方向 | 到什么程度 | 为什么 |
|---|---|---|
| 集合通信 | 讲清 AllReduce/Ring、NCCL 在做什么、DeepEP 的低延迟 dispatch/combine 思路 | 不要求你有 8 卡环境，要求你能在面试里画出数据流 |
| RDMA | 知道 verbs 接口、RoCE vs IB 的差别 | 够用即可，深水区 |
| 编译器 | 读 TileLang / Triton 的 lowering 流程，能改一个 pass | 你有 Ninetoothed 底子，这条性价比最高 |

---

### M5 · 暑期实习 + 秋招（2028.3 – 2028.11）

- 2028.3–2028.6：投**暑期实习**（这才是真正决定秋招的那一枪）
- 2028.9–2028.11：秋招。此时你手里应该有：
  1. 量化过的 kernel 项目（M1/M2）
  2. 知名开源 infra 项目的 merged PR
  3. 一段真实实习经历

---

## 简历三件套（到 2028 年必须齐）

1. **「手写 CUDA GEMM / FlashAttention，达到 cuBLAS 的 X%」** — 证明动手能力
2. **「Ninetoothed / TileLang 贡献者，PR #xxx merged」** — 证明抽象能力，且直接对口 JD 第三条
3. **「某公司 AI Infra 实习，负责某算子」** — 证明工程能力

这三样里，**第 2 样是你最独特、别人最难复制的**。

---

## 明确不做的事

- ❌ 现在碰 RDMA / 多机通信（没环境，投入产出比极低）
- ❌ 从头啃 MLIR 完整后端（太重，等 M4 再说）
- ❌ 同时学三门语言 / 追新框架
- ❌ 为了"简历好看"去刷水题式的项目

---

## 每周最低配速（保证不被课程冲垮）

| 项目 | 时间 |
|---|---|
| 写 kernel / 读源码 | 6–8 h |
| 体系结构 / 数学 | 2–3 h |
| 记录（benchmark 报告、笔记） | 1 h |

**记录那 1 小时最重要。** 没有记录的优化，三个月后你自己都想不起来当时为什么快。
