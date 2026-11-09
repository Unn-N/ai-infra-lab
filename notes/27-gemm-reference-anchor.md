# 27 · 建立理想计算尺度与严格FP32 cuBLAS参照

2026-11-09晚间。用户指出缺少整体瓶颈与理论上限，感觉参数实验方向不清。Codex补充两个参照：标称FP32乘加峰值对应的理想计算时间，以及同条件严格FP32 cuBLAS的实测性能。本轮不修改学习kernel。

## 测量对象与精度

学习版使用最新A局部转置K32版本gemm_self_transpose_a.cu，SHA256=8a1cdb9617227d1b81f858df4f90607b0ec4657fc663f2cab84383207309be14。参考调用cublasGemmEx，A/B/C与compute均为FP32，computeType=CUBLAS_COMPUTE_32F_PEDANTIC，mathMode=CUBLAS_PEDANTIC_MATH；alpha=1、beta=0。pedantic模式禁用Tensor Core，避免使用TF32等不同精度路径来比较当前SIMT学习版。

按列存储接口使用C^T=B^T A^T解释row-major数组，调用中交换B/A与N/M，无额外物理转置。独立七组逐输出CPU double检查通过，包含非方阵以验证维度和leading dimension映射。所有普通计时的64点CPU double抽查通过。参考库检查不意味着穷尽全部输入。

新增gemm/gemm_reference_bench.cu把库调用与测量放在辅助入口中，保留学习源码简单。默认REFERENCE_CUBLAS=1测库，编译-DREFERENCE_CUBLAS=0测相同入口下的学习版，均链接-lcublas。cuBLAS创建、设置、销毁及分配/拷贝/校验均不计时。实际库版本见原始运行日志。

## 同期普通计时

相同五形状和随机输入序列，20次预热，每进程3轮、每轮100次调用；每版三进程按self/cublas/cublas/self/self/cublas交错，取九轮批量平均耗时中位数。CUDA events、默认stream、缓冲复用、不清L2。此处20/100不同于此前5/20的参数实验，所有比较均以本轮同期对照为准。

| M,N,K | 学习版 A转置K32 ms | cuBLAS严格FP32 ms | cuBLAS加速比 |
|---|---:|---:|---:|
| 128, 128, 128 | 0.012788 | 0.032658 | 0.392× |
| 512, 512, 512 | 0.043151 | 0.037710 | 1.144× |
| 2048, 2048, 2048 | 1.440502 | 0.809005 | 1.781× |
| 128, 1024, 512 | 0.031508 | 0.023547 | 1.338× |
| 127, 513, 257 | 0.019630 | 0.022021 | 0.891× |

2048学习版1.440502ms、cuBLAS 0.809005ms，实测库参照约快1.781倍。三次进程中位数分别为学习版1.437268/1.440502/1.460316ms，库0.798124/0.811780/0.836409ms。本轮范围分离。

短任务的cuBLAS进程与轮次波动很大，例如128的进程中位数0.068804→0.032658→0.013934ms，包含库API派发和WDDM等运行环境因素，尚未分离。它们不作为谁普遍更快的证据。cuBLAS实测是可达到的外部参考，不是数学上限，也未做库算法或Lt参数搜索。

## 标称理想计算尺度

NVIDIA产品页列出RTX5070为6144 CUDA核心、标称boost 2.51GHz。按每核心每周期一次FP32乘加、每乘加2FLOP估算，标称峰值为6144×2.51GHz×2=30.84288TFLOP/s。

2048方阵的通常GEMM计数为2×2048×2048×2048=17,179,869,184FLOP，约17.18GFLOP。若这些计算始终按上述峰值执行，纯计算时间尺度为0.557012ms，约0.56ms。

这只是标称频率下的理想算术参考：没有计入加载、地址/控制、同步和调度成本；真实GPU可能在高于或低于标称boost的频率运行。因此它不是不可突破的物理严格下界，也不是承诺当前代码能达到的耗时。没有使用Tensor/AI TOPS作为当前FP32标尺。

重复缓冲区输入可能命中缓存，本次不以A+B+C共48MiB除以DRAM峰值的数值当作热缓存测量的必需DRAM时间。

## 指令与管线对照

为确认库路径和结构差异，另编译独立2048 profile入口，先运行六次、再用cudaProfilerStart/Stop只采第七次完整GEMM调用。每次调用在报告中均只有一个kernel；库名称为Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>，与SIMT参考设置相符。Nsight Compute application replay、cache-control none、clock-control none，正反顺序各一次。

| 指标 | 学习版 | cuBLAS严格FP32 |
|---|---:|---:|
| 动态warp指令总数 | 532,800,777—532,801,176 | 303,979,904—303,979,904 |
| 调度器发射活跃，active周期 % | 77.16—77.41 | 78.87—78.89 |
| FMA指令吞吐，elapsed % | 40.53—40.76 | 69.33—69.39 |
| 采集时SM时钟GHz | 2.188—2.202 | 2.772—2.781 |

此处profiling沿用六次先行调用，与普通计时的21次先行调用（校验+20次预热）不同；GPU频率未锁定，采集时学习版约2.19—2.20GHz、库约2.77—2.78GHz。不能用profiling耗时比替代上方普通计时，也不能将全部加速唯一归因为指令数。

仍可得到较强的结构线索：同一2048乘法，学习版约5.33亿warp指令，库约3.04亿；学习版约为其1.75倍。二者发射活跃都在约77%—79%，但库FMA吞吐更高。减少辅助指令、让更多发射用于有效乘加，成为有外部对照支撑的方向；仍未证明唯一主瓶颈。两种kernel的具体分块、地址计算、循环和访存方式不同，不将全部差异称为可无代价消除的冗余。

## 用户提出的后续两项，尚未执行

1. 将固定block边长和线程步长以编译期常量表达，先单独观察搬运循环是否简化。
2. 再在通过验证的版本上做global→shared float4搬运：A可沿原global行一次读四个float，再拆分写入Ats[k+j][r]；B可连续读并尝试连续写。必须检查16B起点对齐与四元素有效性，非整齐K/N和边界用安全标量路径，保留现有输入支持。用SASS/动态指令确认宽加载是否出现，不能把显式float4当作自动加速保证。

两项都指向搬运辅助指令成本。float4不减少有效搬运字节数，主要假设是降低加载/地址/控制等指令开销。此处只记录方案，不修改或测试这两项优化。

## 文件与复现

notes/data/2026-11-09-gemm-reference-anchor保存实际测量/check/profile源码、编译及采集命令、日志、完整检查、计时、计数器和theory.json；build/reference_anchor_2026-11-09保留忽略的二进制。experiment.py.txt为实际编排脚本，prepare/measure/profile/record分阶段执行；prepare要求新目录。

本轮默认kernel、A转置K32及K64候选均未修改。新增参考入口及记录尚未提交/push。

参考：[NVIDIA RTX5070规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/)、[cuBLAS精度与Tensor Core模式](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage)。
