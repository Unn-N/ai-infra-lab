# 37 · GEMM实验检查点与续学入口

2026-11-17。按用户要求，先提交并推送已完成成果，再更新文档。

## Git状态与保存范围

实验成果已提交为 [16f3e54](https://github.com/Unn-N/ai-infra-lab/commit/16f3e54f8edc023ac2fbab6497bda7571b059f6c)，推送到Unn-N/ai-infra-lab的main；推送后通过git ls-remote核对远端main与该完整提交号一致。

该提交包含此前未提交的notes/31—36、五个独立kernel候选、公共benchmark动态shared适配，以及六组实验的源码快照/编译/正确性/计时/计数器/SASS证据。二进制与.ncu-rep没有纳入。原有Softmax、默认GEMM学习源码保留。

此文档和测量约定随后单独提交；最终文档提交号以Git历史为准。实验笔记中“未提交/未推送”描述的是各轮结束时状态，统一由本检查点更新。

## 当前代码位置

| 文件 | 用途 |
|---|---|
| [gemm_self.cu](../gemm/gemm_self.cu) | 初次自写基础tiled实现 |
| [gemm_self_re.cu](../gemm/gemm_self_re.cu) | 默认学习入口，64/32/4；没有自动切换到实验候选 |
| [gemm_self_float4.cu](../gemm/gemm_self_float4.cu) | 64/32/4、A局部转置与float4参考；小输入仍有优势 |
| [gemm_self_float4_reg8_bpad_double_buffer.cu](../gemm/gemm_self_float4_reg8_bpad_double_buffer.cu) | 128/32/8、B分组padding、global→shared双缓冲 |
| [gemm_self_float4_reg8_bpad_register_pipeline.cu](../gemm/gemm_self_float4_reg8_bpad_register_pipeline.cu) | 再加shared→寄存器双缓冲，当前大方阵路线优先基线 |
| [gemm_self_float4_reg8_bpad_warp32x64.cu](../gemm/gemm_self_float4_reg8_bpad_warp32x64.cu) | 只改warp输出归属，保留为未证实净收益的对照 |

## 本阶段获得的证据

- 8×8寄存器分块增加复用，也带来寄存器、驻留及访问布局代价，不能仅凭搬运减少预言加速。
- B分组padding显著降低读bank冲突，完整实现没有因此自动提速。见notes/32—33。
- 同布局128/8版本加global→shared双缓冲，在11月10日的2048长批次中约1.08倍加速；仍略慢于当时的64/4参考。见notes/34。
- 11月17日进一步增加shared→寄存器双缓冲，2048同进程配对相对原外层约1.11倍、相对64/4约1.09倍。见notes/35。
- 改为每warp32×64输出，2048配对约1.004倍，仅13/24轮更快。当前测量噪声下尚未分辨出稳定收益。见notes/36。

上述都是各自同期对照，不将不同日期的绝对耗时相除。保持FP32 SIMT，尚未引入Tensor Core，也未确定唯一主瓶颈或达到理论上限。

## 验证边界

最近两轮三版各16组CPU double完整输出检查、全部测量前随机抽查和正式公共benchmark入口smoke通过。发布前复核保存的检查结果、源码SHA和证据清单；这是完整性复核，本次收尾没有重新运行GPU性能实验。

Compute Sanitizer racecheck受本机WDDM初始化失败阻挡，不能记为通过或已定位kernel竞争，详见notes/34。性能使用用户日常桌面显卡，外部负载可能变化；新约定见[测量协议](04-measurement-protocol.md)。

## 从哪里继续

用户当前处于积累优化工具箱阶段，可以继续先理解方案，再授权单项实验，不设强制手写某段代码的前置门槛。当前以notes/35版本为这条路线的基线，下一项优化尚未选定；不自动叠加更多warp/shared布局或参数组合。
