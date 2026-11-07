# 14 · 当前 GEMM 的带宽推算与性能计数器检查

2026-11-07。用户追问：能否测量/推算搬运吞吐及等待，而不止停留在“尚未测量”。以下仅针对当前 OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4 的实现。

## 可以计算的源码逻辑需求

对 M=N=K=2048，grid 有 1024 个 block，每块 K 循环 64 轮，每轮搬入两个 64×32 输入块，共 4096 个 float，即 16 KiB。

| 项目 | 数量 |
|---|---:|
| 源码逻辑输入读取 | 1024×64×16 KiB = 1 GiB |
| 输出写回 | 2048×2048×4 字节 = 16 MiB |
| A、B 独立输入数据合计 | 32 MiB |
| 本机实际查询的 L2 容量 | 48 MiB |
| 前一轮同期对照的新版耗时中位数 | 1.649162 ms |
| 逻辑输入读取量 / 该耗时 | 651.08 GB/s |
| 逻辑输入读取及输出写回 / 该耗时 | 661.26 GB/s |

这里是源码需求与已有计时相除，不是新采集的 DRAM 吞吐。重复读取可能由 L2/L1 等缓存提供，输出写回也涉及缓存；32 MiB 输入可以在容量上放入 48 MiB L2，但输出及其他 GPU 工作也会竞争缓存，不能由容量直接断言全部命中。不能把 651 GB/s 当作实测 DRAM 带宽，也不能据此声称显存接近饱和或很宽裕。

## 需要计数器观测的量

- DRAM 的真实读取/写入字节数和吞吐，L2 流量及命中情况。
- 显存/内存管线的活跃周期与吞吐相对其持续峰值的比例。
- SM 指令发射、等待数据和屏障等停顿情况。

吞吐百分比不是闲忙时间百分比。SM 等待数据、内存正在处理请求、其他 warp 执行计算可以重叠，不能把不同单位或不同统计口径直接相减成“搬运时间 + 空闲时间”。计数器能显著缩小判断范围，但要按定义解释。

参考：[Nsight Compute 指标与采集说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[性能诊断流程](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。

## 首次工具检查与当时的权限限制

已找到 `/usr/local/cuda/bin/ncu`，版本 2026.2.1.0。以 WSL 用户 unn 及 root 查询指标均收到：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

此次权限检查阶段未采集到 DRAM 活跃率、真实流量或停顿计数。没有将该错误解释为工具未安装或硬件不支持，也没有更改驱动权限。

NVIDIA 当前说明提供 Windows NVIDIA App 的 `System > Advanced > Developer > Manage GPU Performance Counters` 入口，修改需要管理员权限。WSL 的计数器访问由 Windows 主机侧设置控制。官方说明见 [权限设置](https://developer.nvidia.com/ERR_NVGPUCTRPERM)。开启后再对当前 kernel 定点采集，保留缓存/时钟口径；profiler 的采集耗时不直接替代普通 CUDA event 对照。


## 同日更新：权限已开启，采集已完成

用户随后开启主机侧权限，已对当前 2048³ 配置完成两次采集。实际 DRAM 吞吐约 42—53 GB/s、平均活跃周期约 6.25%—11.44%，L2 sector 命中约 90.10%—95.46%。两次的耗时与显存流量有波动，不能当作固定性能值。完整口径、停顿指标与限制见 [15 · 首次计数器采集](15-gemm-first-profile.md)。
