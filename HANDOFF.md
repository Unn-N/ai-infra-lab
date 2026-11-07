# 学习项目接手记录

核对日期：2026-11-06（Asia/Hong_Kong）。此记录由 Codex 根据实际文件、Git 状态和 DSH「问候」中的相关原文整理。

## 项目位置与目录

当前源码、Git 历史、笔记与编译运行均以 **WSL Ubuntu 的 `/home/unn/ai-infra-lab`** 为准，用户为 `unn`。

Windows 可通过 `\\wsl.localhost\Ubuntu\home\unn\ai-infra-lab` 访问同一目录。

`C:\Users\Administrator\Documents\deepseek-harness\default-workspace\ai-infra-lab` 是早期副本：其中 GEMM 仍位于根目录，还缺少 WSL 中的部分自写文件；不能直接拿它覆盖当前仓库。

```text
ai-infra-lab/
  README.md                    使用入口、当前状态与编译命令
  HANDOFF.md                   本记录：目录、Git 基线、验证范围与接续位置
  ROADMAP.md                   早期助手提出的路线图，非用户确认的计划
  .clangd / .clang-format       CUDA 编辑与格式设置
  softmax/
    hello_cuda.cu              环境自检
    softmax_v1.cu              初始示例与测量框架
    softmax_v2.cu              多行/warp 归约示例
    softmax_self.cu            用户重写的 v1
    softmax_v2_self.cu         用户重写后继续完善的 v2，含 float4
    softmax_v2_self_bench.cu   自写版本的测量入口
    softmax_test.cu            内嵌 naive 实现的五用例正确性程序
  gemm/
    gemm_bench.cu              naive / shared tiled / cuBLAS 对照
    gemm_self.cu               首次自写 tiled kernel 草稿（2026-11-07，待验证）
  notes/
    01-bottleneck.md           早期瓶颈分析与实验
    02-execution-model.md      早期执行模型笔记
    03-v2-multrow.md           多行与 warp 归约实验
    04-measurement-protocol.md 测量协议与历史噪声观察
    05-float4.md               float4 实验记录
    06-gemm-tiling.md          已讨论清楚的 GEMM 分块流程
  build/                      编译产物与临时实验文件，Git 已忽略
```

`build/` 还留有标量/float4 对比的临时源码与程序。此次未清理；若需复现实验，应先确认临时源码是否需要正式保存。

## Git 基线（2026-11-06 接手时）

- 仓库：[Unn-N/ai-infra-lab](https://github.com/Unn-N/ai-infra-lab)。
- 当前分支：`main`，跟踪 `origin/main`；只有一个已登记的工作目录。
- 接手前本地 HEAD、已有 `origin/main` 引用及本次 `git ls-remote` 查到的远端 main 均为 `c2a8f31d3449aab13df367b06fdbe49812bd224b`。
- 已有 7 个提交；最近两个是 `73d985e`（float4 实验与测量记录）、`c2a8f31`（GEMM 移入子目录并修复 host 向 device 指针写入的问题）。
- **接手前工作区不是干净状态**：`gemm/gemm_bench.cu` 有未提交改动。改动包含非有限数检查、大尺寸 naive 不再跳过，以及大尺寸结果差异的调试输出。这些不是本次接手新增的代码。
- 本次只更新文档，没有提交或推送，没有更改远端、分支或已有源码。

原始 GEMM diff、README/ROADMAP 原文、源码 SHA-256 与 Git 基线已保存至本聊天工作目录的 `handoff/2026-11-06/`。原始 patch 为 `gemm-before-takeover.patch`。

常用只读检查：

```bash
cd /home/unn/ai-infra-lab
git status --short --branch
git diff --stat
git diff -- gemm/gemm_bench.cu
git log -7 --oneline
```

## 环境与验证范围

- 实际运行环境：WSL 发行版 `Ubuntu`，用户 `unn`；另一个 `Ubuntu-24.04` 不是此项目所在的环境。
- 已观察到 GPU 为 RTX 5070；`/usr/local/cuda/bin/nvcc --version` 为 CUDA 13.3，V13.3.73。
- 当前非交互命令的 PATH 未包含 CUDA 编译器目录，命令使用完整路径以避免依赖交互终端设置。
- 本次从源码重新编译并运行 `softmax/softmax_test.cu`，5/5 PASS：随机、非整齐 N、极端数值、全相等行、单元素行。日志保存在聊天目录的 `handoff/2026-11-06/softmax-correctness.txt`。
- **该程序内嵌了 naive Softmax 实现**；通过它不能证明独立的 v1、v2、自写 float4 版本都已经通过同样测试。
- 本次没有重新运行 GEMM 性能测量，也没有用 profiler 证明瓶颈。DSH 原文有 GEMM 运行输出，属于历史观测；“已经测过”和“原因已经证实”必须分开。

```bash
cd /home/unn/ai-infra-lab
mkdir -p build
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 softmax/softmax_test.cu -o build/softmax_test_takeover
./build/softmax_test_takeover

# 需要继续 GEMM 实验时再编译和运行；此次未执行以下两行。
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_bench.cu -o build/gemm_bench -lcublas
./build/gemm_bench
```

## 学习进度与协作方式

Softmax 已经历示例阅读、用户重写、shared 归约到 warp 归约、多行安排、W 选择与 float4 实验。当前主线是 GEMM。

截至接手，用户已经推清楚：

1. C 可划为输出块，并分配给 block。
2. 一个 C 输出块需要 A 的对应横带和 B 的对应竖带。
3. 沿 K 方向逐轮搬入匹配的 A、B 输入块到 shared，线程将部分和保存在各自累加器中。
4. 沿 K 累加结束后才写回这一块 C。
5. 一个 block 处理几个输出块是实现选择。可以给更多输出块安排更多 block，也可以让同一个 block 循环处理多个输出块；所有 block 无需同时驻留。

当前笔记见 [06-gemm-tiling.md](notes/06-gemm-tiling.md)。**register blocking 尚未完成理解与实现，不把它标为已掌握。**

交流按用户自己的推演推进：先听清问题与假设，先回答它是否成立以及成立条件；明确区分输入块、输出块、block 和实际执行资源。用户讨论一般可行方案时，不把当前代码的安排当作硬件限制。一次讲清一个问题，不自行替用户安排长期计划。

## 历史文档中待复核的说法

原有笔记保留历史过程；保留不代表其中每条解释已经被验证。后续碰到对应主题时再逐项核实，不为接手而抢跑学习进度。

- Softmax 的 `%spec` 是按假设字节数与标称带宽估出的指标，不能直接当作测得的真实 DRAM 利用率。缓存与噪声的具体成因尚未全面验证。
- `notes/01` 的“效率”存在比值方向表述问题；“四堵墙互斥”等绝对结论也需修订。
- `notes/02` 的固定驻留线程数、固定延迟、warp 天然同步与写入 L1 的描述需要按架构和代码重新核对，不作为当前教学前提。
- GEMM 注释中的全局访存复杂度、padding 必然加速、纯 FP32 对照与性能上限等解释尚未系统审计。
- “tiled 已经证实卡在 shared 带宽”“读次数下降多少，耗时必然同比下降”不能从当前运行表直接推出。
- 单线程处理 8×8 输出时，每个 k 步若读 8 个 A 标量、8 个 B 标量并更新 64 个输出，是 **16 次标量读取对应 64 次乘加**；早期聊天里的“2 次读取对应 64 次乘加”不能沿用。
- `ROADMAP.md` 是 DeepSeek 早期自行提出的建议。DSH 第 68 轮明确承认它不是商定计划，不据此设定期限、验收标准或催促学习。

## 接下来从哪里继续

2026-11-07 已读完基础 tiled kernel 的核心计算流程，保留了 `gemm/gemm_self.cu` 的第一次自写草稿。用户明确表示“知道怎么回事，但从空白重新写还需要推导”，继续以独立尝试和按需提示推进。

下一步先处理自写文件的 Clang 专用头文件编译错误，再让用户核对输出坐标与 shared 布局的对应关系；随后补 host 入口并做小尺寸正确性验证。详细过程与实际编译错误见 [当日学习记录](notes/06-gemm-tiling.md)。尚未有自写 GEMM 的通过结果或新性能数据。

本次按用户要求将接手文档、学习记录、自写草稿以及原有 `gemm_bench.cu` 未提交改动一起保存为 Git 检查点。2026-11-06 的 Git 基线与验证记录保留为历史事实；最终提交与远端同步状态以实际 Git 结果为准。
