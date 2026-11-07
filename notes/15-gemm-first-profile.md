# 15 · 当前 GEMM 的首次计数器采集

2026-11-07。用户在 Windows 主机开启 GPU 性能计数器权限后，Nsight Compute 已能正常采集。此次仅观测当前 kernel，没有实现双缓冲，也没有修改学习源码。

## 采集对象与方法

- RTX 5070，48 SM；M=N=K=2048，FP32；OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4；block=16×16，grid=32×32。
- Nsight Compute 2026.2.1.0；独立临时入口引用当前 gemm_self_re.cu，以 nvcc -O3 -lineinfo -arch=sm_120 编译。
- 输入随机序列与原测量入口中 2048³ 一组一致。A/B 只上传一次，连续启动 7 次，每次同步；跳过前 6 次，采集第 7 次，对应一次验证启动和五次预热之后。未清空缓存。
- application replay；cache-control none；clock-control none；16 项指标需要 11 次重放。两份报告为独立采集。缓存状态、桌面 GPU 工作与动态时钟没有隔离控制。
- 原始报告保存在忽略的 build/gemm_profile_2026-11-07/ 中。可读原始指标、命令与入口快照见 [data/2026-11-07-gemm-profile](data/2026-11-07-gemm-profile/)。入口快照需放回 build/gemm_profile_2026-11-07/profile_current.cu，保持相对 include 路径后再编译。

## 观测结果

| 指标 | 第一次 | 第二次 |
|---|---:|---:|
| 采集中的 kernel 耗时（ms） | 2.259 | 1.442 |
| DRAM 读取（MiB） | 72.34 | 39.26 |
| DRAM 写入（MiB） | 40.80 | 18.42 |
| DRAM 读写吞吐（GB/s，十进制） | 52.51 | 41.95 |
| DRAM 平均活跃周期比例（%） | 11.44 | 6.25 |
| L2 sector 命中率（%） | 90.10 | 95.46 |
| L2 吞吐指标（持续峰值的 %） | 20.36 | 20.39 |
| long scoreboard 停顿（%） | 9.95 | 9.99 |
| short scoreboard 停顿（%） | 8.28 | 8.28 |
| barrier 停顿（%） | 6.64 | 6.59 |
| not selected（%） | 33.60 | 33.58 |

DRAM 平均活跃周期比例由 dram__cycles_active.avg / dram__cycles_elapsed.avg 计算。完整指标名保留于原始 JSON，避免将不同口径的百分比混用。

## 可以支持的判断

对当前 2048³、重复使用同一输入、未清缓存的工作负载，两次采集均支持：**显存 DRAM 带宽没有吃满，不能把源码约 1 GiB 的逻辑读取量都算成 DRAM 读取。** L2 命中率较高，实际 DRAM 流量明显低于源码重复读取的总量。

两次 DRAM 流量、活跃比例与耗时有明显波动，不能将第一次的 53 GB/s 或第二次的 42 GB/s 视为固定特性。相比之下，本次几类 warp 停顿比例接近。多轮重放、缓存与时钟未控制使这些数据适合诊断，不能替代普通 CUDA event 的同期性能对照。DRAM 写入计数也不能直接解释为 C 的唯一写回字节数：观测可能包含缓存写回等影响，未进一步隔离归因。

## “搬运多久、等待多久”的口径

- DRAM 平均活跃约 6.25%—11.44%，是内存单元活跃周期的统计比例；余下部分不能称为“GPU 在等任务”。SM 计算、其他 warp 执行与内存请求可以重叠。
- long scoreboard 约 10% 表示 warp 等待 L1TEX 路径相关数据依赖的统计，可能涉及不同缓存层级；不能直接叫“等显存的 kernel 时间”。short scoreboard 和 barrier 同样是 warp 周期口径，不能直接乘 kernel 耗时算全卡停顿时间。
- not selected 约 33.6% 表示有 warp 已准备好、但当时调度器选择了别的 warp，并不等价于 GPU 空闲或损失了这部分时间。
- 因此可以测到真实流量、活跃情况和等待类型，但不能简单将整段墙钟时间切成互斥的“读、算、写、闲置”四块。

## 对双缓冲假设的影响

当前结果支持继续考虑通过流水重叠加载和计算来隐藏延迟；它没有证明内存延迟是唯一或首要瓶颈，也没有证明双缓冲一定加速。即便 L2 命中，数据到达 shared/寄存器仍有指令与依赖延迟。

还需保留资源代价：当前每线程 62 个寄存器、每块 16 KiB 静态 shared、理论 4 block/SM。仅以额外 16 KiB shared 做占用率容量估算时变成 3 block/SM；真正实现后寄存器、同步和代码生成还可能变化。后续若用户选择实现，需做单项改动，再以普通计时入口对照。此次没有开始该实现。

参考：[NVIDIA 指标定义与重放说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[性能诊断流程](https://docs.nvidia.com/nsight-compute/ComputeTriage/)、[CUDA 异步拷贝](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。
