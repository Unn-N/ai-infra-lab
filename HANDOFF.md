# 学习项目接手记录

> 2026-11-07 最新收尾状态见 [notes/22-gemm-day-end.md](notes/22-gemm-day-end.md)。默认保留64/4单缓冲，双缓冲文件当前32/2；下文逐时记录保留历史状态。

核对日期：2026-11-06（Asia/Hong_Kong）。此记录由 Codex 根据实际文件、Git 状态和 DSH「问候」中的相关原文整理。

**当前状态（2026-11-07 里程碑）：** 当前 re 配置为 OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4；原 self 为对照，独立检查与测量入口均已保存。最新状态与复现见 [12-gemm-milestone.md](notes/12-gemm-milestone.md)。下文接手基线及按时间追加的进度保留为历史记录，以最新里程碑为准。

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
    gemm_self.cu               自写 tiled kernel、host 调用与手算小例子
    gemm_self_re.cu            当前 64/32/4 寄存器分块版本，带 Codex 实现注释
    gemm_self_check.cu         七组 CPU 全输出正确性检查
    gemm_self_bench.cu         Codex 编写的独立测量入口，目前引用 re 版本
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

同日上午，用户继续完成 host 调用骨架，并要求 Codex 补齐且标明作者。当前 `gemm_self.cu` 已有带 Codex 注释的错误检查、输出坐标修正、调用收尾和最小 main；编译、手算例子与 4 组 CPU 参考对照均通过。前述初稿失败记录保留为历史过程，详细验证范围见 [当日学习记录](notes/06-gemm-tiling.md)。下一步由用户理解补充代码、解释索引并继续写自己的注释；没有新增性能数据。

本次按用户要求将接手文档、学习记录、自写草稿以及原有 `gemm_bench.cu` 未提交改动一起保存为 Git 检查点。2026-11-06 的 Git 基线与验证记录保留为历史事实；最终提交与远端同步状态以实际 Git 结果为准。


2026-11-07 下午补充：用户希望沿用 Softmax 的小步迭代路径，并明确由 Codex 负责测量辅助代码、保持其余代码简单。已新增独立 `gemm_self_bench.cu`，原计算代码仅增加省略 main 的开关；五组输入抽查通过，首轮基线与完整口径见 [07-gemm-baseline.md](notes/07-gemm-baseline.md)。此阶段没有改动 GEMM 算法；后续逐项提出假设、实现和比较，不要求先完成通用自动选型。


2026-11-07 下午 TILE 实验：按用户要求仅改常量测试 16/64，并与 32 交错对照。16 的五组抽查通过，四组非最小输入在两次对照中均更快；128³ 波动大，暂不判断。64 编译成功但 4096 线程/block 超过设备 1024 上限，启动失败，没有性能数字。实验后源码恢复 TILE=32。详细结果见 [08-gemm-tile-size.md](notes/08-gemm-tile-size.md)。


随后用户要求增加 naive 与 TILE=8/4 对照。已使用同一测量入口在临时副本中完成 5 个版本、3 次交错进程的实验，学习源码与测量入口均未改动。四组非最小输入的汇总结果在 16 最快，继续缩到 8/4 回退，4 慢于固定 block=32×32 的 naive；128³ 波动大，不做稳定排名。表格、条件和原始数据见 [09-gemm-tile-sweep.md](notes/09-gemm-tile-sweep.md)。


2026-11-07 15:30 后：按用户要求，仅在其新建的 `gemm_self_re.cu` 中将 TILE 拆为 OUTPUT_TILE=32、K_TILE=32、THREAD_TILE=1，并同步测量入口。原 `gemm_self.cu` 未改动。四组 CPU 全输出对照、小例子和五组测量抽查通过；当前 static_assert 保留单输出与原加载布局的限制。用户明确要求后续优化步骤自己考虑，不替其实现寄存器分块。


2026-11-07 15:39 后，用户确认执行每线程 2×2 的首项实验：当前 re 版本是 OUTPUT_TILE=32、K_TILE=32、THREAD_TILE=2，block=16×16。Codex 实现并验证了新加载/累加/写回流程，原版 self 保留。四组 CPU 全输出对照与全部性能输入抽查通过；同期三次交错对照中四组非最小输入约 1.96—2.94× 加速，128³ 汇总退步且波动大。资源查询为 42 寄存器/线程、8192 字节 shared/block、理论最多 5 block/SM。详见 [10-gemm-register-2x2.md](notes/10-gemm-register-2x2.md)。


2026-11-07 15:58 后按用户要求将当前 re 配置改为 OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4，block 保持 16×16。小例子、七组 CPU 全输出对照和全部性能抽查通过。同期交错对照中 2048³ 约 1.57× 加速，512³ 约 1.04×，另外三组退步；资源为 62 寄存器/线程、16 KiB shared/block、理论最多 4 block/SM。当前源码保留新配置，旧版快照保留于 build/register_4x4_2026-11-07。详见 [11-gemm-register-4x4.md](notes/11-gemm-register-4x4.md)。


2026-11-07 16:06，用户明确将当前位置确认为里程碑并要求提交。整理当前实现、检查/测量入口、实验表格和原始输出，补充 notes/12；将已使用的七组 CPU 全输出对照正式保存为 gemm_self_check.cu。K_TILE 后续对照尚未开始。


2026-11-07 16:25，按用户要求将其对“扩大输出块提高复用，但小/窄输出可能缺少并行任务”的思考整理为 [13-reuse-and-parallelism.md](notes/13-reuse-and-parallelism.md)。明确区分输出规模、输入数据量与计算量，区分增加块内线程和增加独立 block；按形状选择配置是候选方向，自动分派与下一轮 K_TILE 实验均未执行。


2026-11-07，用户追问当前 GEMM 的真实带宽与等待。初次 ncu 权限检查受阻，随后用户在 Windows 开启性能计数器权限，已完成两次 2048³ 采集。DRAM 平均活跃约 6.25%—11.44%，L2 命中约 90.10%—95.46%，支持当前预热重复输入的 DRAM 带宽未饱和；计时/流量存在波动，不把 warp 停顿比例当作全卡墙钟时间。见 notes/14、15 及 notes/data/2026-11-07-gemm-profile；未修改 kernel，未实现双缓冲，未提交此次文档。


2026-11-07 17:24 后，用户同意尝试双缓冲。新增独立 gemm_self_double_buffer.cu，默认 re 版保留；测量/检查入口通过编译选项选择实验版本。实际 70 寄存器/线程、32 KiB shared、理论 3 block/SM；七组全输出检查及全部性能抽查通过。同期三次交错实验中 2048³ 耗时约增加 13.3%，两组长方形/带边缘输入约 1.22—1.25× 加速，512³ 差别小，128³ 波动。Compute Sanitizer 被主机 WDDM 调试接口阻挡，未通过此项验证；详见 notes/16 及原始数据。尚未提交，未继续调参。


2026-11-07 17:31 后，按用户要求对单/双缓冲做同期带宽采集，选 2048³ 与 128×1024×512。少指标定点采集各两轮：长方形 DRAM 约 64—67→90—93 GB/s、字节带宽占持续峰值约 9.6%—10.0%→13.4%—13.9%；大方阵约 42—46→31—34 GB/s，占比约 6.2%—6.9%→4.6%—5.0%。长方形 SM elapsed 指令发射比例约 17%→30%，活跃周期比例仍约 64%，不解释为使用了更多 SM。宽指标采集中两次大方阵双缓冲 L2 hit 超 100%，保留异常并另做少指标验证；详见 notes/17 和原始数据。未修改 kernel，未提交。


2026-11-07 17:43 后按用户要求，将独立双缓冲版输出块改 32、线程块内每线程输出改 2×2，K_TILE 仍 32；旧 64/4 源码快照保留。四版本同期对照，2048 输入中 64/4 单缓冲 1.602622 ms、双缓冲 1.758870 ms，32/2 单缓冲 2.491141 ms、双缓冲 3.437003 ms。新双缓冲实际 56 寄存器/线程、16 KiB shared、理论 4 block/SM；七组完整检查及全部测量抽查通过。该输入排名在三次进程中一致，短任务仍波动；仅记录结构线索，未单独 profile 新版。当前 double_buffer 文件保持新配置，默认 re 文件未改，详见 notes/18。尚未提交。


2026-11-07 18:40 后，用户重新追问瓶颈与是否全任务同时驻留。核实 64/4 单缓冲 2048 输出需 1024 block，全卡理论同时 192；4 block/SM 上限来自寄存器（62 编译寄存器、实际分配 64/线程），而非静态 shared 填满。补采 FMA 管线指令吞吐约 39%、发射约 72%、active warp 占用约 63%、L1TEX 汇总约 78%，存在 shared bank conflicts。仅定位到容量约束和进一步线索，没有证明唯一主瓶颈或计算算力已满；见 notes/19。未改 kernel，未提交。


2026-11-07 18:54 后按用户要求，以单缓冲测32×32=1024线程，分别每线程2×2（输出64）和4×4（输出128），K_TILE=32。两版均能启动，七组完整检查与全部性能抽查通过；实际寄存器42/62，shared16/32 KiB，都最多1 block/SM。三版同期2048耗时：原256线程/4×4 1.511902 ms，新1024线程/2×2 3.052032 ms，新1024线程/4×4 1.583077 ms；三次进程排名一致。理论驻留线程都1024/SM，增加单块线程没有提高该上限。保存notes/20及三版快照，默认学习源码与双缓冲源码均未改，尚未提交。


2026-11-07 19:03 开始、19:40 中断恢复后完成As-only padding实验：最快64/4单缓冲仅As行宽32→33，静态shared+256B，寄存器62→56，理论仍4 block/SM。七组完整检查及全部性能抽查通过。2048普通同期计时1.566336→1.718776 ms，约慢9.7%；shared读取bank冲突3360万→约3万（减少99.9%），但动态shared加载指令50331648→67108864（增加33.3%）。SASS中4条LDS.64变为2条LDS.64+4条LDS，奇数行132B步长破坏8B对齐。证据表明冲突减少同时带来指令增多，尚未隔离全部代价；默认源码保持未padding，候选快照/原始数据见notes/21。未继续+2或Bs布局实验，未提交。


2026-11-07 23:25，用户要求收尾提交。新增notes/22作为当前入口，汇总notes/13—21与原始数据、实验源码快照；保留最快默认基线和独立双缓冲学习文件。只整理已授权的本日改动，构建二进制和.ncu-rep仍忽略。本条随收尾检查点保存，实际提交与远端同步以Git为准。
