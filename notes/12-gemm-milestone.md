# 12 · 第一个 GEMM 实现与性能实验里程碑

确认时间：2026-11-07，Asia/Hong_Kong。用户认为到当前位置可以作为一个里程碑，要求提交保存。

## 已留下什么

- `gemm_self.cu`：用户搭建、Codex 标注补齐的基础 shared-tiled 版本，包含 host 调用和手算小例子；每线程一个输出。
- `gemm_self_re.cu`：当前寄存器分块版本，OUTPUT_TILE=64、K_TILE=32、THREAD_TILE=4，block=16×16；线程私有累加器和输入复用，协同加载与逐元素边界写回。实现由 Codex 按用户提出并确认的方案补充，代码保留作者标记。
- `gemm_self_check.cu`：独立 CPU 全输出对照，七组尺寸，覆盖单元素、非方阵、K 尾部、线程输出小块边缘与跨输出 tile。
- `gemm_self_bench.cu`：由 Codex 维护的独立测量入口，用户的学习代码不承担计时、随机输入与参考检查的组织工作。
- `notes/07`—`11` 及 `notes/data/`：固定口径下的逐尺寸结果、原始轮次、改善、退步、失败配置和未解问题。

## 本轮得到的证据

1. 只改基础版本 TILE 时，在本轮五组输入中的四组非最小输入上，16 的汇总表现优于 32、8、4；不能将此分界推广为所有输入的最佳配置。
2. 基础版本直接设 TILE=64 会要求 4096 线程/block，超过设备 1024 上限，启动失败；这是线程配置限制，不是不能设计 64×64 输出块。
3. 输出保持 32×32、每线程改算 2×2 时，同期对照中四组非最小输入约 1.96—2.94× 加速；128³ 汇总退步且存在明显波动。
4. 再扩大为 64×64 输出、每线程 4×4，保持 K_TILE=32 和 256 线程/block，2048³ 同期约 1.57× 加速，512³ 小幅改善，另三组退步。
5. 64/32/4 编译资源为每线程 62 个寄存器、每 block 16 KiB shared、理论最多 4 block/SM；这不是实测利用率。理论驻留能力下降但大方阵仍加速，表明不能只用一个资源指标判断总性能。

不同阶段在不同时间测量，不能把各表的数字直接拼成严格的总加速比。具体对照条件和原始结果分别见 [TILE 扫描](09-gemm-tile-sweep.md)、[每线程 2×2](10-gemm-register-2x2.md)、[每线程 4×4](11-gemm-register-4x4.md)。

## 复现入口

在仓库根目录、WSL Ubuntu 中运行；若交互 PATH 已配置 CUDA，可将完整路径写成 nvcc。

```bash
mkdir -p build
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self.cu -o build/gemm_self
./build/gemm_self
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self_re.cu -o build/gemm_self_re
./build/gemm_self_re
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self_check.cu -o build/gemm_self_check
./build/gemm_self_check
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 gemm/gemm_self_bench.cu -o build/gemm_self_bench
./build/gemm_self_bench
```

同一套当前 re 计算代码也能复现上一档配置：将 OUTPUT_TILE 设为 32、THREAD_TILE 设为 2，保持 K_TILE=32，然后重新编译检查和测量入口。原始二进制不进入 Git；历史原始测量数据保存在 notes/data。

## 当前位置与边界

已经有能运行、验证和反复实验的小项目，但不把助手实现的部分等同于用户已经独立掌握。继续以用户推演、自己读写和按需协助为主。

当前没有自动选择配置、没有改精度、没有 Tensor Core 或双缓冲。数据复用、同步、寄存器需求和任务并行度提供了解释线索，尚未用 profiler 分离它们的实际贡献；不宣称已经锁定唯一瓶颈或达到硬件上限。

下一项候选是保持输出块和每线程输出不变，对照 K_TILE=16/32/64；尚未执行。先理解现有协同加载中的一维编号与步长，后续实验由用户决定。
