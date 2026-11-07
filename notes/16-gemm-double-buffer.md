# 16 · 首次异步双缓冲实验

2026-11-07。用户用“更少的人、但每个人更快吃完一个苹果”类比驻留量与单块效率的取舍，并同意尝试双缓冲。

## 假设与改动范围

当前预热 2048³ 的 DRAM 未饱和，尝试在计算当前输入块时提前搬运下一块，以减少数据依赖等待。容量足够不保证加速：新增 shared、寄存器及控制指令都可能抵消收益。

- 基线仍为 gemm/gemm_self_re.cu，未改动；配置 OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4，block=16×16。
- 新增独立 gemm/gemm_self_double_buffer.cu，Codex 编写并标注加载、等待与缓冲切换。
- 两组 As/Bs：先加载第一组并等齐；每轮先向另一组发起下一轮异步加载，再计算当前组；等待加载完成并 block 同步后切换。越界补零，乘加顺序与输出分工保留。
- 使用 cuda_pipeline.h 的 __pipeline_memcpy_async / commit / wait_prior。每次拷贝一个 float（4 字节），未同时尝试 float4、padding、K_TILE 调整或 warp 专门分工。
- 反汇编已见 LDGSTS 异步 global→shared 搬运指令；说明确实发出了异步请求，不等于已量化运行时隐藏了多少延迟。
- 测量/检查入口仅增加 GEMM_SELF_SOURCE 编译选择，默认行为不变。学习 kernel 内不增加版本选择分支。

## 同期性能对照

沿用五组输入、固定随机数据与 64 点 CPU double 抽查。每进程预热 5 次、3 轮各 20 次启动，CUDA events；每版本 3 次进程，顺序 single/double/double/single/single/double。表中为九轮批量计时均值的中位数；相同缓冲、未清 L2，分配/拷贝/校验不计时。全部性能抽查通过。

| M×N×K | 单缓冲 ms | 双缓冲 ms | 加速比：单 / 双 |
|---|---:|---:|---:|
| 128×128×128 | 0.010942 | 0.011859 | 0.923× |
| 512×512×512 | 0.043904 | 0.043048 | 1.020× |
| 2048×2048×2048 | 1.630683 | 1.847767 | 0.883× |
| 128×1024×512 | 0.034635 | 0.027819 | 1.245× |
| 127×513×257 | 0.021488 | 0.017648 | 1.218× |

- 2048³ 的双缓冲耗时约增加 13.3%，三次进程的中位数都比相邻原版更慢。此次实现未达到大方阵加速的预期。
- 512³ 仅约 2% 差别，不把它判断为可靠收益。
- 128×1024×512、127×513×257 分别约 1.245×、1.218× 加速，对应耗时约减少 19.7%、17.9%；三次进程中均保持相同方向。这是当前测量口径下的结果，不外推所有形状。
- 128³ 太短且波动大，暂不做稳定排名。原始每轮数据包含于下面的数据目录。

## 编译资源与结果解释

| 项目 | 单缓冲 | 双缓冲 |
|---|---:|---:|
| 寄存器 / 线程 | 62 | 70 |
| 静态 shared / block | 16 KiB | 32 KiB |
| local memory / 线程 | 0 | 0 |
| 理论驻留 block / SM | 4 | 3 |
| 全卡理论同时驻留 block | 192 | 144 |

双缓冲编译器报告无寄存器溢出，资源 API 确认每 SM 最多 3 block。大方阵有 1024 个 block，驻留容量的减少可能影响隐藏延迟；两组较窄输出只有 32 和 18 个 block，本来就不足以铺满 48 SM，4→3 的上限通常不是它们的约束。这能给出一个有条件的解释，但没有做反事实隔离，不能断言它就是大方阵变慢的唯一原因。

此次同时改变了搬运指令、缓冲数量和循环时序；未增加“相同异步指令但不重叠”的控制组，因此不能将长方形收益全部归因于双缓冲重叠，也不能定量拆分寄存器、驻留与指令开销各占多少。

## 正确性与验证边界

现有七组逐输出 CPU double 检查全部通过，覆盖小矩阵、非方阵、K 尾部、输出块边缘以及缓冲复用；五组性能输入的随机采样也全部通过。

尝试 Compute Sanitizer memcheck 时，Windows 主机 WDDM 调试接口未启用，工具初始化失败并报告设备不支持；未获得有效 memcheck/racecheck/synccheck 结果。保存原始错误，不把工具失败当作 kernel 错误，也不声称已通过竞争检测。未为此更改主机调试设置。

## 复现

默认原版测量命令保持不变。双缓冲版在仓库根目录执行：

```bash
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 '-DGEMM_SELF_SOURCE="gemm_self_double_buffer.cu"' gemm/gemm_self_bench.cu -o build/gemm_self_double_buffer_bench
./build/gemm_self_double_buffer_bench
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 '-DGEMM_SELF_SOURCE="gemm_self_double_buffer.cu"' gemm/gemm_self_check.cu -o build/gemm_self_double_buffer_check
./build/gemm_self_double_buffer_check
```

原始日志、九轮汇总、源码哈希和资源记录：[data/2026-11-07-gemm-double-buffer](data/2026-11-07-gemm-double-buffer/)。构建快照与反汇编位于忽略的 build/double_buffer_2026-11-07/。

当前保留单缓冲为默认版本、双缓冲作为实验版本，不因大方阵回退而丢掉这次结果；未自动提交或 push。进一步优化等待用户选择。

参考：[NVIDIA 异步拷贝与预取模式](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。


后续状态更新：notes/18 已按用户要求将当前独立双缓冲学习文件改为 32/2。本页 64/4 双缓冲是历史实验；对应原始源码快照保存在 notes/18 的数据目录中。复现本页时应使用该 double64 快照，而非直接编译当前 32/2 文件。
