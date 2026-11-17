# 2026-11-17 warp分工对照

reg4=原64/4；base=notes/35双层流水线；warp=只改warp32×64输出分工。
metadata记录源SHA及资源；measure是五形状3×100、三进程；paired是2048同进程每版200次/批、12轮×2进程。
profile两轮顺序反转；首轮base时钟偏低，不使用采集耗时判定性能。各项原始数据全部保留。
warp_layout.py是实际实验脚本；make_warp_experiment.py是其初始生成器，之后单独加入了shared wavefront与global store sector两项指标，以实际脚本为准。
命令JSON含本机build绝对路径；复现到其他目录时应调整。完整check/bench/paired源已保存，可不依赖旧build直接编译。

在此数据目录执行：

```bash
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 warp/check.cu -o /tmp/gemm-warp-check
/tmp/gemm-warp-check
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 paired.cu -o /tmp/gemm-warp-paired
/tmp/gemm-warp-paired
```

正式公共benchmark以编译参数 GEMM_SELF_SOURCE 选择 gemm_self_float4_reg8_bpad_warp32x64.cu；示例见public_bench_compile.command.json。
