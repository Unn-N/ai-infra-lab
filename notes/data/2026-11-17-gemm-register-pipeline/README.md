# 2026-11-17 shared→寄存器双缓冲实验数据

- reg4：64/32/4 float4参考；outer：notes/34最终外层双缓冲；inner：本次候选。
- metadata.json有源码SHA与资源；*_check.txt是16组完整输出检查。
- measure_*：五形状3×100、三进程；sustained_*：2048的3×1000、三进程。
- paired.cu与paired{1,2}.txt：同一进程交错三版，每轮各200次，两个进程共24轮；paired_runs/summary保留原值和配对比。
- profile_summary.json：正反顺序各一轮；*_sass.txt、opcode_arrays.json保留实际指令证据。计数器不是独立的因果证明。
- *.command.json为当时命令，含本机build绝对路径；重放到其他目录时应调整路径。
- *.py记录构建、测量和归档过程；原prepare依赖上一轮build中的辅助入口，但本目录已完整保存这些入口，可直接重编译。

在本数据目录中运行（输出放/tmp，也可自行指定build目录）：

```bash
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 -Xptxas=-v inner/check.cu -o /tmp/gemm-register-pipeline-check
/tmp/gemm-register-pipeline-check
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 paired.cu -o /tmp/gemm-register-pipeline-paired
/tmp/gemm-register-pipeline-paired
```

从仓库根使用正式公共测量入口：

```bash
/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 -DGEMM_SELF_SOURCE='"gemm_self_float4_reg8_bpad_register_pipeline.cu"' gemm/gemm_self_bench.cu -o build/gemm_register_pipeline_bench
./build/gemm_register_pipeline_bench
```

本轮不重试WDDM初始化失败的racecheck；参见notes/34历史日志。无commit/push。
