# ai-infra-lab

> **2026-11-09 收尾：** 已完成A局部转置、严格FP32参照、编译期线程分工及float4四组实验；最新候选、验证边界和下次亲手编码入口见 [当日收尾](notes/30-gemm-day-end.md)。

> **2026-11-07 收尾：** 双缓冲、驻留、1024线程与padding实验已整理；当前版本、验证边界及续学入口见 [当日收尾](notes/22-gemm-day-end.md)。

> **2026-11-07 里程碑：** 已完成基础 GEMM、寄存器分块实现和多组性能对照。当前成果、验证范围与复现命令见 [里程碑记录](notes/12-gemm-milestone.md)。
>
> 2026-11-06 接手核对：以 WSL Ubuntu 的 `/home/unn/ai-infra-lab` 为当前仓库。目录、Git 基线和验证范围见 [HANDOFF.md](HANDOFF.md)。当前学习记录见 [一个 block 怎样算出一块 C](notes/06-gemm-tiling.md)。
>
> 下方 Softmax 性能数字来自历史实验，本次没有复测性能；缓存、带宽和瓶颈解释尚未全面复核。`ROADMAP.md` 是早期助手建议，并非用户确认的计划。

一个 GPU 算子的学习与优化实验台。目标不是"把算子写出来"，而是**知道自己离硬件上限还有多远、差在哪**。

---

## 目前做了什么

| 工作 | 状态 | 结果 |
|---|---|---|
| 逐行 softmax（朴素版） | ✅ | `8192×8192` 达 **85.5%** 显存带宽 |
| 逐行 softmax（warp 版） | ✅ | `N=128` 从 **19% → 93%**，**4.96×** |
| naive Softmax 正确性测试 | ✅ | 2026-11-06 从源码重编译，5/5 PASS；不覆盖全部独立优化版本 |
| GEMM 基准与参照 | ✅ | 已有固定输入检查/计时及严格FP32 cuBLAS同期参照；整体唯一主瓶颈仍未确定 |
| GEMM 首个实验里程碑 | ✅ | 基础实现 → TILE 对照 → 每线程 2×2 / 4×4；当前七组 CPU 全输出检查通过，逐尺寸性能与退步均保留 |
| GEMM A转置与float4 | ✅ | 四组十用例检查通过；2048的float4增量约1.13倍，小输入与回退路径另行观察 |

### v2 的 4.96× 是怎么来的

拆成两个正交的改动，分别测：

| 改动 | 针对的瓶颈 | 收益 |
|---|---|---|
| 归约：shared 树形 → `__shfl_down_sync` | 屏障开销（9 次 → 0~2 次） | **1.63×** |
| 结构：一行 256 线程 → 一行 32 线程（多行共享 block） | 短行时一半 warp 空转 + 在飞请求不足 | **3.04×** |

`1.63 × 3.04 = 4.96`，和实测对得上。

---

## 历史实验的三个观察

1. **源码重复读取，不等于真实 DRAM 流量按次数增加。** 缓存可能吸收重复访问；历史实验中减少遍数没有带来原先预期的加速。具体 DRAM 流量尚未由 profiler 核实。
2. **短行和长行可能需要不同的线程安排。** 历史对比中，短行受益于 warp 归约和多行共享 block；不将这些观测泛化为所有尺寸的固定瓶颈。
3. **`%spec` 是估算指标。** 它根据假设字节数和标称带宽计算，不能直接视为实测 DRAM 利用率。小工作集、缓存状态与测量噪声都可能影响解释。

原始过程和数据保留在 [`notes/`](notes/)；其中待复核的技术说法列于 [接手记录](HANDOFF.md)。

---

## 目录

```
softmax/
  softmax_v1.cu        朴素版：block 一行、shared 归约、带 L2 冲刷的计时框架
  softmax_v2.cu        优化版：多行/warp 归约，W=1/2/4/8 可切
  softmax_self.cu      自己重写的 v1
  softmax_v2_self.cu   自己重写后完善的 v2（W dispatcher + float4）
  softmax_test.cu      正确性测试（5 个用例）
  hello_cuda.cu        环境自检
gemm/
  gemm_bench.cu        GEMM 基准（朴素 / tiled / cuBLAS 对照）
  gemm_self.cu         原自写 tiled 基线、host 调用与手算小例子
  gemm_self_re.cu      默认对照：64×64 输出块、16×16 线程、每线程 4×4 输出
  gemm_self_double_buffer.cu  独立双缓冲实验：当前32×32输出、每线程2×2，历史64版本有快照
  gemm_self_check.cu   当前 re 版本的七组 CPU 全输出正确性检查
  gemm_self_bench.cu   Codex 编写的独立测量入口，引用 gemm_self_re.cu
  gemm_self_float4.cu  最新有收益候选：A局部转置、float4搬运、编译期线程分工
  gemm_self_float4_plain.cu  未转置的float4对照；其余独立候选见notes/30
  gemm_reference_bench.cu    严格FP32 cuBLAS与学习版的参照测量入口
notes/                 实验记录
```

---

## 运行

需要 CUDA Toolkit ≥ 12.8（RTX 50 系需要 `sm_120`）。

```bash
mkdir -p build

nvcc -O3 -arch=sm_120 softmax/softmax_test.cu        -o build/softmax_test    && ./build/softmax_test
nvcc -O3 -arch=sm_120 softmax/softmax_v1.cu          -o build/softmax_v1      && ./build/softmax_v1
nvcc -O3 -arch=sm_120 softmax/softmax_v2.cu          -o build/softmax_v2      && ./build/softmax_v2 1
nvcc -O3 -arch=sm_120 softmax/softmax_v2_self_bench.cu -o build/bench_v2self && ./build/bench_v2self
nvcc -O3 -arch=sm_120 gemm/gemm_bench.cu -o build/gemm_bench -lcublas && ./build/gemm_bench
nvcc -O3 -arch=sm_120 gemm/gemm_self_check.cu -o build/gemm_self_check && ./build/gemm_self_check
nvcc -O3 -arch=sm_120 gemm/gemm_self_bench.cu -o build/gemm_self_bench && ./build/gemm_self_bench
```

非交互环境中若找不到 `nvcc`，使用 `/usr/local/cuda/bin/nvcc`。

自写 GEMM 的独立检查与测量入口见上方命令。实验按阶段记录：

| 记录 | 内容 |
|---|---|
| [07 · 测量基线](notes/07-gemm-baseline.md) | 五组输入、计时与缓存口径 |
| [08 · 首次 TILE 对照](notes/08-gemm-tile-size.md) | 16/32 与 64 的启动限制 |
| [09 · TILE 扫描](notes/09-gemm-tile-sweep.md) | naive、TILE=32/16/8/4 |
| [10 · 每线程 2×2](notes/10-gemm-register-2x2.md) | 固定 32×32 输出块，调整线程分工 |
| [11 · 每线程 4×4](notes/11-gemm-register-4x4.md) | 扩大输出块的收益与退步 |
| [12 · 里程碑](notes/12-gemm-milestone.md) | 当前状态、验证范围和下一步候选 |
| [13 · 复用与并行度](notes/13-reuse-and-parallelism.md) | 输出分块的结构性取舍与按输入选择配置的思考 |
| [14 · 带宽推算与测量](notes/14-gemm-bandwidth-analysis.md) | 逻辑读取需求、真实 DRAM 流量的区别及计数器权限状态 |
| [15 · 首次计数器采集](notes/15-gemm-first-profile.md) | 当前 2048³ 的真实 DRAM 流量、缓存与等待指标 |
| [16 · 首次双缓冲](notes/16-gemm-double-buffer.md) | 异步预取实验、资源代价及不同形状的性能变化 |
| [17 · 双缓冲带宽对照](notes/17-gemm-double-buffer-profile.md) | 同期 DRAM 字节带宽与 SM 指令发射指标，含异常采集记录 |
| [18 · 双缓冲缩回 32/2](notes/18-gemm-double-buffer32.md) | 四版本同期对照，降低资源占用后的性能回退 |
| [19 · 驻留与执行管线](notes/19-gemm-residency-and-pipelines.md) | 寄存器驻留约束、FMA/发射指标与 shared 冲突观测 |
| [20 · 1024线程实验](notes/20-gemm-block1024.md) | 每线程2×2/4×4、输出64/128的单缓冲同期对照 |
| [21 · As padding](notes/21-gemm-padding-as.md) | +1减少读取冲突，但对齐变化使加载指令增多、性能回退 |
| [22 · 当日收尾](notes/22-gemm-day-end.md) | 当前版本、实验索引、验证边界与续学入口 |
| [23 · As +2 padding](notes/23-gemm-padding-as2.md) | 恢复宽加载并减少bank冲突；2048未测出稳定净加速 |
| [24 · 调度与指令诊断](notes/24-gemm-scheduler-diagnosis.md) | 发射空缺、shared队列/读取等待与搬运辅助指令的源码关联 |
| [25 · A tile局部转置](notes/25-gemm-transpose-a.md) | 计算取数连续且宽加载减少；2048同期约1.085倍加速，写冲突增加 |
| [26 · K_TILE扩大到64](notes/26-gemm-k-tile64.md) | 单block shared翻倍、驻留减半；2048本轮耗时增加约15.7% |
| [27 · 性能参照与理想计算尺度](notes/27-gemm-reference-anchor.md) | 严格FP32 cuBLAS同期约快1.78倍；指令总量对照支持检查辅助指令成本 |
| [28 · 编译期线程分工](notes/28-gemm-const-block.md) | 动态指令减少约7.85%，该轮2048普通计时未加速 |
| [29 · float4四组对照](notes/29-gemm-float4-factorial.md) | 有无A转置分别加入float4；2048增量约1.13倍，转置+float4本轮最快 |
| [30 · 本日收尾](notes/30-gemm-day-end.md) | 最新候选、实验索引、验证范围与用户亲手编码的续学入口 |

**编译产物统一输出到 `build/`**，源码目录保持干净。`build/` 已在 `.gitignore` 里，不用每次想“这个要不要提交”。

硬件：RTX 5070（48 SM / 12 GB GDDR7 / 峰值带宽 ≈ 672 GB/s）

### ⚠️ 测性能前先做这些

**测量噪声可能比你要测的优化还大。** 这台机器同时还在渲染 Windows 桌面，桌面一刷新就抢显存带宽。

1. 关 **Wallpaper Engine** 等动态壁纸
2. 关任何 GPU 加速程序（安卓模拟器、MAA、浏览器视频）
3. **每个配置跑 3 次，取中位数**（单次数字不可用，哪怕环境干净）

历史测量中出现过明显波动和离群值，因此比较时需保留多次运行数据与中位数；不保证执行这些步骤后噪声一定降到某个固定范围。`%spec` 超过 100% 时应检查字节数假设、缓存和计时口径，不能解释为真实 DRAM 带宽突破上限。详见 [`notes/04-measurement-protocol.md`](notes/04-measurement-protocol.md)。

---

## 待办

- [x] 已实现 `float4` 实验，历史数据见 `notes/05-float4.md`；不将它视为普遍加速结论
- [ ] Softmax：拿到真实 DRAM 流量，修正 `%spec` 口径（计数器权限已开启；GEMM 首次采集见 notes/15，Softmax 尚未重测）
- [ ] 把 v2 的两层归约拆成独立可测的中间版本
- [x] GEMM naive / shared tiled / cuBLAS 已有历史运行输出
- [x] 走完基础 tiled GEMM 的 kernel 流程并保存首次自写草稿
- [x] 核对自写版本的索引与编译问题，补 host 入口并完成初步正确性验证
- [ ] 继续给自写 GEMM 加解释注释，理解 Codex 标注的补充与坐标修正；见 [学习记录](notes/06-gemm-tiling.md)

- [x] 完成首轮 GEMM 寄存器分块与性能实验里程碑；保留原始数据和独立正确性入口

- [x] 完成GEMM局部转置、K64、严格FP32参照、编译期线程分工与float4四组对照，见notes/23—30
- [ ] 下次由用户亲手写A的float4读取与转置写入，Codex先提示/review，见notes/30
