# 24 · 默认 GEMM 的调度、等待与指令构成诊断

2026-11-09。用户授权检查“数据已驻留但一轮处理很慢”的原因，由Codex设计和执行采集。本轮固定默认64/32/4单缓冲、无padding，输入(M,N,K)=(2048,2048,2048)，只观察，不修改kernel。默认源码SHA256与此前一致：e0746da45b8a956068ca57632600a1b1c7ac4a3499ee2e2dc0b35218d20f5788。

## 采集口径

沿用相同profile入口、固定随机输入、六次预热调用后采第七次；-O3 -lineinfo -arch=sm_120。Nsight Compute 2026.2.1，application replay、cache-control none、clock-control none。

SchedulerStats、WarpStateStats、ComputeWorkloadAnalysis和InstructionStats共采两轮；每轮涉及20次application replay。另采两轮SourceCounters，关联源码/SASS及PC采样，读取其中Not Issued样本。Opcode分布、source表和详细表从已有报告导出，不重复运行kernel。

本轮没有锁定GPU频率，多pass计数不能视为同一条完整时间线；不以被采集/插桩后的耗时替代普通CUDA events计时。首次CSV解析遗漏单位行，修正为选ID=0数据行，复用已有报告，无需丢弃或重跑第一轮采集。源码行关联是编译器调试映射，优化后的一条源码行可能承载多个操作，不能直接等同某种唯一指令。

## 调度器是否一直没有工作可做

| 项目 | 两轮范围 |
|---|---:|
| 调度器发射活跃，active周期 % | 73.77—74.22 |
| 驻留warp/调度器，active周期平均 | 7.55—7.55 |
| 可执行warp/调度器，active周期平均 | 3.31—3.33 |
| FMA指令吞吐/持续峰值，elapsed % | 36.69—38.75 |
| LSU指令吞吐/持续峰值，elapsed % | 65.13—68.79 |

调度器约74%的active周期存在可发射工作并发射，约26%的active周期无eligible warp。平均驻留7.55个warp，可执行约3.3个warp，说明有相当多周期已有候选工作，同时也存在未被隐藏的等待；平均可执行warp数不能证明每周期都有工作可选。

not_selected约3.48—3.50 warp-cycles/issue，是已经可执行但本周期调度器选择别的warp。工具自动规则显示的约34%“Local Speedup”源于该类别的局部比例；它不是整个kernel可兑现的加速预测，本记录不按它声称34%时间浪费或34%提速空间。

FMA指令吞吐约37%—39%，没有乘加管线已经满载的证据。LSU指令吞吐约65%—69%比FMA更高，但口径不是shared字节带宽，也不足以单独证明唯一瓶颈。没有在较大汇总吞吐数字中选一个当作硬件满载结论。

## 发射空缺时，线程在等什么

| 发射空缺时的采样类别 | 两轮样本数范围 | 主要源码位置 |
|---|---:|---|
| MIO队列满 | 6,305—6,501 | 75/76行：shared→寄存器读取 |
| 等待L1TEX操作结果 | 5,195—5,249 | 58/65行：global→shared的消费侧 |
| 等待MIO操作结果 | 3,680—3,763 | 79行：乘加消费shared加载的结果 |
| 固定延迟执行依赖 | 3,094—3,100 | 分散在读取、地址/索引等代码 |
| 块内同步 | 2,061—2,147 | 75/83行及屏障后续指令 |
| 数学执行管线不可用 | 353—404 | 数值较小，分散在若干源码行 |

MIO队列压力、global读取依赖、shared读取依赖和固定延迟依赖均存在；数学管线排队样本较少。没有一个单独原因足以解释全部发射空缺。MIO还包括特殊数学/动态分支等操作，结合本例75/76行源码和LDS指令才把这里的热点关联为shared访问。

重要边界：上述是warp PC采样次数，多个warp的等待可重叠，不能换算为可相加的kernel墙钟时间百分比。即使仅看Not Issued，也需要用针对性代码改动和普通计时验证实际代价。消费指令承接依赖等待样本：FFMA行出现short scoreboard不表示FFMA算得慢；它可能在等此前LDS返回。long scoreboard表示等待L1TEX相关操作，不能仅凭此区分L1/L2/DRAM，不能称为“等待显存总线”。

## 真实指令构成

两轮warp级总指令数均为562,184,192。按opcode与category导出的总数相互核对一致。FFMA为268,435,456，乘32线程等于8,589,934,592次线程乘加，符合2048×2048×2048。

| 动态指令 | 次数（warp级） | 占全部指令 |
|---|---:|---:|
| FFMA，GEMM浮点乘加 | 268,435,456 | 47.75% |
| 整数指令类别 | 134,414,336 | 23.91% |
| Load/Store类别 | 75,644,928 | 13.46% |
| Uniform Datapath类别 | 44,728,320 | 7.96% |

FFMA约占47.75%的指令，不能把其余所有指令称为无用开销；读取、地址计算和控制是执行任务必需的一部分，也不能将指令比例视为时间比例。

按源码行55—66（A/B两段global→shared搬运及索引、边界判断等）的调试映射合计177,684,480条，占31.61%；两轮相同。这一区域仍有global读取结果等待样本。因此“数据搬运不是瓶颈”只能收窄为此前未观察到DRAM持续字节带宽饱和，不能排除搬运相关的指令开销与延迟。

内层shared读取75/76行分别映射61,341,696与25,165,824条；72行K内循环控制16,777,216条；79行乘加映射276,824,064条，其中包含FFMA以外的循环生成操作。不要将该行总指令数直接当作纯FFMA数。

## 当前可操作的判断

证据支持将调查重点放到指令组织、片上数据供应以及输入搬运辅助逻辑；暂不支持“DRAM带宽跑满”或“FP32乘加已达峰值”，也没有唯一根因。数据驻留较久是这些活动和等待共同形成的生命周期。

推荐下一个单项实验：将本版固定block分工的线程步长以编译期常量表达，让编译器明确每block256线程、每线程搬运8个A/8个B，观察搬运循环的地址/控制指令是否减少。当前代码使用blockDim动态求threads；不要用一个数字256散落代码，应该从OUTPUT_TILE/THREAD_TILE推导。保留边界正确性、原输出分工和计算逻辑。此建议尚未实施，不能保证加速；验证须包括源码/SASS动态指令变化、资源和同期计时。

## 文件与复现

原始命令、报告CSV/JSON、源码关联表、两轮等待样本、按opcode分布、实际source/profile快照及编排脚本保存于notes/data/2026-11-09-gemm-scheduler。build/scheduler_2026-11-09中可执行文件和.ncu-rep二进制忽略。

本轮未改kernel，未新增正确性测试；对象与notes/23中已通过检查的baseline源码相同。+2候选仍独立保存。未实施下一个优化，尚未提交或push。

参考：[Nsight Compute章节定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)、[等待原因与诊断边界](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-warp-stall-breakdown-dominated-by-specific-stall-reasons)、[指令分类指标](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#instructions-per-opcode-metrics)。
