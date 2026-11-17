# 31 · 最新float4与严格FP32 cuBLAS：差距与结构

2026-11-10。用户认为降低输入精度的Tensor Core路线改变了比较条件，希望先弄清当前普通FP32学习版与cuBLAS的差距，以及是否只差几个小优化。本轮仅补测和诊断，不改计算kernel，不开始Tensor Core或新优化。

## 对照与验证

- 学习版：gemm/gemm_self_float4.cu；SHA256=4ee9339437cf7c087a23a7935e0f199aa0deb7092a22b750be72ac6a4a280801。64×64输出、K32、每线程4×4，256线程，A局部转置，float4搬运，单缓冲。
- cuBLAS：沿用gemm_reference_bench.cu的GemmEx，FP32 A/B/C、32F_PEDANTIC、PEDANTIC_MATH，alpha=1/beta=0。交换A/B和M/N解释row-major，不做物理全矩阵转置。库版本130600。
- 独立实验目录把参考入口的默认include改为最新float4快照，修正显示名称；仓库公开的参考入口及所有学习kernel未改。
- 两版均对同一十组小尺寸/非方阵/边缘用例逐输出CPU double检查，全部PASS；全部普通计时的64点随机输入检查PASS。检查范围有限，不等于穷尽全部输入。未重试此前WDDM阻挡的Compute Sanitizer，不声称通过race/sync/memcheck。

## 同期普通计时

沿用五形状、固定种子20261107。一次检查后20次预热；每进程3轮，每轮100次。self/cuBLAS/cuBLAS/self/self/cuBLAS交错运行，各三进程，取九轮批量平均耗时的中位数。CUDA event包住调用批次；同缓冲、不清L2，不含分配/拷贝/检查/handle创建。短任务可能包含CPU派发空隙和WDDM环境因素。

| M,N,K | 最新转置+float4 ms | cuBLAS严格FP32 ms | 库加速比 |
|---|---:|---:|---:|
| 128, 128, 128 | 0.011003 | 0.015395 | 0.715× |
| 512, 512, 512 | 0.040896 | 0.026974 | 1.516× |
| 2048, 2048, 2048 | 1.245332 | 0.786424 | 1.584× |
| 128, 1024, 512 | 0.029038 | 0.019987 | 1.453× |
| 127, 513, 257 | 0.018946 | 0.036146 | 0.524× |

2048³本轮学习版1.245332ms、cuBLAS 0.786424ms；库快1.584倍。若要追平，学习版耗时需下降36.85%，不是“减少58%耗时”。常规GEMM有效FLOP口径为17,179,869,184，分别约13.80/21.85TFLOP/s。三进程中位数学习版1.241589/1.245332/1.260149ms、库0.776635/0.788332/0.786424ms；九轮范围不重叠。

512和长方形本轮汇总库更快；128和非整齐输入学习版更快，但短任务尤其库非整齐数据有明显进程漂移（0.066299/0.036146/0.022627ms），不据此宣称普遍优胜或稳定算法排名。全部数据保留，不只报告最快值。

## 2048指令：乘加相同，配套工作不同

Nsight Compute application replay、cache-control none、clock-control none。与普通计时先行调用数对齐：21次未采集调用后只采第22次。两版正反顺序各采一轮；完整GEMM调用均只看到一个kernel。profile入口每次调用后同步，与普通批量启动不同；profiler耗时不替代上表。

补充InstructionStats各一轮，通过工具自带Python报告接口提取sass__inst_executed_per_opcode数组，而不是只读CSV中该数组的总和。以下统一为warp执行次数，包含谓词关闭的指令；不是线程数、字节量或耗时比例。

| 指标，warp执行次数 | 学习版 | cuBLAS |
|---|---:|---:|
| 全部指令 | 436,436,992 | 303,979,904 |
| FFMA指令 | 268,435,456 | 268,697,600 |
| 非FFMA指令 | 168,001,536 | 35,282,304 |
| shared读取LDS | 33,554,432 | 13,791,744 |
| 普通shared写入STS | 5,242,880 | 393,216 |
| 直接搬运LDGSTS | 0 | 3,256,320 |
| 同步BAR | 1,048,576 | 378,880 |

FFMA占全部warp指令：学习版61.51%，库88.39%。学习版非FFMA约为库4.76倍。这些指令包含必要的地址、控制、搬运、同步和输出操作，不能都称为可删除的浪费，也不能按数量比例推算耗时收益。

两版pred-on线程FFMA为8,589,934,592和8,598,323,200，仅差约0.098%；库含额外输出/归并等操作，不能把微小差额归因于已经证明的单一具体步骤。主体计算均以FP32 FFMA完成，未观察到Tensor MMA opcode。SASS出现HFMA2 R,-RZ,RZ,0,0等零值初始化写法，不代表输入矩阵转成half参与主乘法。

普通shared读hardware指标记录学习版33,554,432、库12,994,560，而InstructionStats的库LDS为13,791,744；前者分类及谓词口径不同，不混拼。本表使用InstructionStats一致口径。

## 资源和执行组织

| 项目 | 学习版 | cuBLAS本轮kernel |
|---|---:|---:|
| 线程/block | 256 | 256 |
| 寄存器/线程，编译值 | 39 | 210 |
| 寄存器实际分配/线程 | 40 | 216 |
| kernel shared/block，不含driver | 16.5KiB | 48KiB |
| 含driver的shared/block | 17.5KiB | 49KiB |
| 理论block/SM | 5 | 1 |
| grid | 32×32×1=1024 | 128×1×3=384 |
| 本次InstructionStats溢出请求 | local/shared均0 | local/shared均0 |

库kernel名称为Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>。名称提示256×128分块，实际读取了SR_CTAID.Z，grid.z=3；与K切分/归并的结构相符，但尚未逐段还原参数和完整索引，不能把名称当作已完成的源码级证明，也不能简单说库只有128个block。每线程具体输出坐标和warp映射尚未还原。

库实际执行LDGSTS 3,256,320条，并执行LDGDEPBAR/DEPBAR；学习版没有这些指令。这证明库使用global→shared直接异步搬运路径，不能仅以两边普通LDG/STS计数比较总搬运成本。SASS布局支持进一步研究流水线，但当前尚未还原缓冲级数或量化搬运与计算重叠收益；不能把48KiB都解释成“更多当前输入数据”，也不能把以往一个双缓冲版本回退当作整个方向无效。

| 指标 | 学习版两轮 | cuBLAS两轮 |
|---|---:|---:|
| FMA管线elapsed吞吐 % | 43.73—43.81 | 69.35—69.45 |
| 调度器active周期发射 % | 71.85—72.01 | 78.91—78.94 |
| active warp占用 % | 77.85—77.91 | 16.62—16.62 |
| shared读取bank冲突 | 68,846.00—72,614.00 | 0.00—0.00 |
| shared写入bank冲突 | 13,763,156.00—13,770,394.00 | 0.00—0.00 |

两轮采集学习版SM时钟约2.796—2.798GHz、库2.777—2.784GHz，已不存在notes/27的2.19 vs 2.78GHz明显差异。仍未锁频，普通计时结果是时间比较依据。

## 可以判断什么，下一步研究什么

1. 在当前2048和重复缓冲负载上，差距是约1.58倍，已达到同一数量级，但追平所需约37%时间下降仍有实质工作。
2. 有明确的结构差异：高寄存器投入/较低驻留、不同输出分块/任务分配、少得多的辅助和shared读取指令、直接异步搬运。不能断言只需几个常量，也没有证明参数空间耗尽或所有收益来自某一项。
3. 优先学习普通FP32下warp/线程输出分工和寄存器复用的组织，再研究与之匹配的搬运及shared→register流水线。具体方案由用户理解后选择，本轮不实现。不同时把库所有技巧移植过来。
4. 用户2026-11-10明确当前重在积累优化工具箱和负责设计/实现/验证；AI可承担机械实现与测量，不设“必须先手写完某段才能继续”的门槛。notes/30原手写计划属于前一日状态，不作为当前强制任务。

证据尚不能把总差距拆成各技巧的毫秒贡献；需要受控实现或更细还原才能归因。cuBLAS是一个可达到的库参照，不是数学性能上限，也未搜索cuBLASLt算法。

## 复现与文件

notes/data/2026-11-10-gemm-reference-gap保存实际bench/check/profile源码快照、编译命令、全部计时、原始指标、SASS、InstructionStats数组和experiment.py.txt；build/reference_gap_2026-11-10保存忽略的可执行文件和报告。脚本按prepare/measure/profile/disasm/instructions/opcode_arrays/record阶段运行，prepare和record要求新目录。先运行Python报告接口提取数组再record。

初次合并请求dram__bytes_read/write未返回指标，解析检测到缺项后移除；本轮没有DRAM字节流量结论。opcode_raw_summary为数组总和的探索记录，正式分类依据opcode_arrays.json。

参考：[CUTLASS层级分块、流水线与K切分](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)、[Nsight Compute指标与采集口径](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

本轮仅诊断和记录；不修改kernel、不提交或push。
