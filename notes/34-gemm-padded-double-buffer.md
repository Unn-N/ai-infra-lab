# 34 · padding版128/8的双缓冲：A寄存器预取、B异步搬运

2026-11-10。用户要求在notes/33行内padding版本上继续加双缓冲，使用空闲shared尝试重叠搬运和计算。上午开始实现，14:52用户要求继续后完成最终测量。本轮不改精度、不改输出/线程分工及K_TILE，不把旧64/4实验的双缓冲收益外推到这里。

## 实现及启动

独立gemm/gemm_self_float4_reg8_bpad_double_buffer.cu，基于128×128输出、每线程8×8、16×16线程、K32、A转置+4、B每8有效数后留4空位。两个shared stage均含Ats[32][132]、Bs[32][192]，单stage41,472B，两份82,944B=81KiB；含driver为82KiB/block。

超过默认48KiB静态shared限额，采用extern __shared__ __align__(16)动态区域，分成两个stage，每个stage再分A/B。host用cudaFuncSetAttribute(...MaxDynamicSharedMemorySize,82944)显式申请，启动第三参数传入GEMM_DYNAMIC_SHARED_BYTES。cudaMalloc起点及所有stage、B基址/行跨度/四元素组均满足16B对齐。

混合流水线保留A向量快路径：A下一轮四组float4先进入每线程寄存器，完成当前轮计算后转置散写到另一shared stage；B对齐且四元素有效时用__pipeline_memcpy_async直接搬到下一stage的padding位置。K/N非4倍数及边界仍按原规则安全标量读取/补零，不对无效源地址发起异步拷贝。B commit前__syncwarp，wait_prior(0)只等本线程的批次，之后的__syncthreads保证全block数据到齐。

先装stage0并等待/同步。每轮预取下一组A、发起下一组B，再算当前stage；等待B、写入预取A后，统一block同步再交换stage。该同步同时保证旧stage已被所有线程用完，下一轮可以安全覆盖。2048同步轮次由原128次变为初始1次+轮间64次，共65次。这是包含预取、B异步路径及同步重排的完整实现，不把全部收益唯一归因为某一种重叠。

候选SHA256=6c5f098bac21fd40eeb8beb3afe0a8ee7367e425c011c9d02c629849b0077f46。原单缓冲及64/4源码均保留，record再次核对基线字节一致。

## 动态shared引起的代码生成问题及修正

初版照搬原b[j]逐项读取表达式，编译器在动态指针下生成A侧2条LDS.128、B侧8条标量LDS。2048 shared有效读16,777,216→41,943,040、读bank冲突重新升到33,554,432。不能把这种额外变化混入最终双缓冲结论。

修正为：每线程8列恰好位于一个B_GROUP内，先算bc=b_shared_col(local_col)，再显式float4读取Bs[k][bc]和Bs[k][bc+4]，将8分量交给原8×8乘加。所有地址可证明16B对齐、无跨组/越界读取；增加static_assert固定当前8×8/8元素分组前提。最终SASS保留A/B各两条LDS.128，读bank冲突两轮均为0。该修改只是保留单缓冲已有宽访问，不改变逻辑分工。

初版源、编译、校验、计时、SASS和profile原始文本保存在数据目录initial_scalar_B；主目录最终结果全部在修正后重新采集。初版128regs、修正版126regs，不混用资源/耗时。

## 正确性与资源

三版重新-O3 -lineinfo -arch=sm_120、ptxas -v编译；十六组逐输出CPU double检查全部PASS：notes/32十二组，再加(129,132,64)、(129,132,96)、(129,132,97)、(131,133,65)，覆盖一轮、奇偶多轮、跨stage复用、K尾部、输出边界和向量/标量分支。全部普通/长批次计时前64点随机输入检查PASS。修正宽读取后重新编译并重跑完整检查。

| 资源 | padding单缓冲 | padding双缓冲 |
|---|---:|---:|
| 线程/block | 256 | 256 |
| 寄存器/线程 | 93 | 126 |
| static shared B | 41472 | 0 |
| dynamic shared B | 0 | 82944 |
| kernel总shared B | 41472 | 82944 |
| local/线程 B | 0 | 0 |
| 理论block/SM | 2 | 1 |

最终126寄存器实际按128分配；双缓冲1block/SM的限制来自shared用量，寄存器理论可支持2block/SM。实际local/shared spilling请求均为0。资源查询、公共计时和profile都传实际动态shared大小，不按零字节计算occupancy。

重新尝试Compute Sanitizer racecheck，返回99：Failed to initialize WDDM debugger interface、Device not supported。检测工具没有成功启动有效目标检测；其summary的2 errors来自初始化失败，不能记作2处已定位kernel竞争，也不能声称racecheck通过。完整日志及命令保存。功能检查通过不等于工具证明不存在所有竞争。

## 同期五形状计时

相同随机数据序列，20预热，3轮×100启动，每版3进程九轮；顺序reg4/single/double、double/single/reg4、single/reg4/double，取批量平均耗时中位数。CUDA events、重复缓冲、不清L2，不含分配、传输、检查及动态shared属性设置。

| M,N,K | 原64/4 ms | padding单缓冲128/8 ms | padding双缓冲128/8 ms | 双缓冲相对单缓冲加速比 |
|---|---:|---:|---:|---:|
| 128, 128, 128 | 0.012072 | 0.024424 | 0.021774 | 1.122× |
| 512, 512, 512 | 0.039357 | 0.078686 | 0.065658 | 1.198× |
| 2048, 2048, 2048 | 1.288975 | 1.426502 | 1.328663 | 1.074× |
| 128, 1024, 512 | 0.030669 | 0.079087 | 0.067325 | 1.175× |
| 127, 513, 257 | 0.019013 | 0.053541 | 0.046164 | 1.160× |

512、长方形和非整齐输入，三次进程中双缓冲均比同结构单缓冲快；128波动明显，不据此外推稳定收益。非整齐K257/N513时，A/B均不走global向量快路径，B也不发起16B异步拷贝；该行改善包含寄存器预取、同步变化等，不能单独归因B硬件异步。

## 2048长批次复测

同输入，20预热、3轮×1000启动，每版3进程九轮，交错顺序同上。跳过前两种形状的随机数，确保与五形状2048输入一致。

| 版本 | 九轮中位数 ms | 三次进程中位数 ms |
|---|---:|---|
| reg4 | 1.321422 | 1.319617 / 1.321223 / 1.325072 |
| single | 1.451807 | 1.44695 / 1.451864 / 1.457966 |
| double | 1.345299 | 1.341216 / 1.345505 / 1.352166 |

双缓冲相对padding单缓冲加速1.0792倍，耗时减少7.34%；三次进程方向一致，两版九轮范围不重叠。相对原64/4耗时增加1.81%，进程中位数仍略慢，部分原始轮次重叠，未证实超过64/4。短批次和长批次分开保存，不与上午不同轮次或旧cuBLAS数据拼接成严格差距。

## 实际指令和执行线索

Nsight Compute application replay、cache-control none、clock-control none；21次未采集调用后采第22次，正反顺序各一轮。profile每次调用后同步，不用采集耗时替代上方普通计时。

| 2048指标 | 单缓冲两轮 | 双缓冲两轮 |
|---|---:|---:|
| 全部warp指令 | 357,552,128.00—357,552,128.00 | 347,226,112.00—347,226,543.00 |
| shared有效读指令 | 16,777,216.00—16,777,216.00 | 16,777,216.00—16,777,216.00 |
| 普通shared写指令 | 2,621,440.00—2,621,440.00 | 2,097,152.00—2,097,152.00 |
| 普通global读指令 | 1,048,576.00—1,048,576.00 | 524,288.00—524,288.00 |
| shared读bank冲突 | 166,203.00—167,603.00 | 0.00—0.00 |
| shared写bank冲突 | 8,718,230.00—8,719,611.00 | 6,292,377.00—6,292,444.00 |
| active warp占用 % | 31.91—32.43 | 16.65—16.65 |
| active周期发射 % | 53.07—53.11 | 59.09—59.71 |
| FMA elapsed吞吐 % | 36.90—36.92 | 41.75—43.76 |
| SM时钟Hz | 2,862,994,046.99—2,863,156,137.15 | 2,179,299,520.60—2,863,169,309.16 |

| InstructionStats warp执行次数 | 单缓冲 | 双缓冲 |
|---|---:|---:|
| FFMA | 268,435,456 | 268,435,456 |
| LDS | 16,777,216 | 18,350,080 |
| LDG | 1,048,576 | 524,288 |
| LDGSTS | 0 | 524,288 |
| STS | 2,621,440 | 2,097,152 |
| BAR | 262,144 | 133,120 |
| LDGDEPBAR | 0 | 131,072 |
| DEPBAR | 0 | 131,072 |

两版pred-on线程FFMA均8,589,934,592，主体算术量不变。B路径由普通LDG+STS换为LDGSTS.E.BYPASS.128，实际LDGSTS524,288次；普通global加载减半不代表数据量减半。A仍为LDG.E.128.CONSTANT，下一轮A向量读取及B拷贝出现在当前计算前，DEPBAR等待和A发布在当前乘加后，SASS可核对该安排；没有单独测出重叠时间或将总加速全归因重叠。

InstructionStats的LDS含1,572,864条@!PT LDS RZ,[RZ]，其pred-on线程计数为0；因此其总LDS18,350,080与有效shared加载计数16,777,216不同，不误判为又拆窄。最终主计算真实读取仍4条LDS.128。

总warp指令约357.55m→347.23m（少约2.89%），BAR262,144→133,120；同样的shared读取次数、低冲突和更少有效驻留warp并存。双缓冲首轮profile时钟约2.18GHz、第二轮2.86GHz，单缓冲两轮约2.86GHz；未锁频，不能直接用profile elapsed排名或唯一归因利用率变化。

## 公共入口与范围

gemm_self_bench.cu仅增加GEMM_DYNAMIC_SHARED_BYTES默认0、启动前opt-in、启动第三参数；默认源和原计时参数均不变。用-DGEMM_SELF_SOURCE=\"gemm_self_float4_reg8_bpad_double_buffer.cu\"选择候选即可。已从公开入口重新编译并跑五形状采样检查，全部PASS；该smoke日志只验证入口，不纳入上方性能汇总。gemm_self_check.cu通过run_gemm使用候选自带配置；本轮十六组检查/资源/profile实际源码均另存。

本轮不加Tensor Core、不改精度、不继续调K或输出分工，不切换默认kernel，不提交或push。收益是在当前padding128/8单缓冲基础上约1.08倍；还没有成为超过旧64/4的整体最佳方案。

## 保存与复现

notes/data/2026-11-10-gemm-padded-double-buffer保存三个真实源码、共同bench/check/resource/profile/sustained入口、编译/校验/计时命令、所有原始结果及SASS/opcode数组、racecheck限制、最终公开bench快照和experiment.py.txt；initial_scalar_B保留初版。build/padded_double_buffer_2026-11-10保留忽略的二进制/报告。

脚本prepare使用同目录hybrid_double_kernel.cu.txt，并复用前一轮build内测量入口（实际副本已保存）；最终snippet已含显式B float4。measure/profile/sustained/sanitize/record为后续阶段，prepare/record要求新目录。rebuild_double仅记录本轮从初版修正的历史步骤，不是最终prepare后必须执行的步骤。

参考：[CUDA pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)、[动态shared opt-in](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)。
