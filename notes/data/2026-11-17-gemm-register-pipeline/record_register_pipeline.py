from pathlib import Path
import json,hashlib,shutil,subprocess

repo=Path('/home/unn/ai-infra-lab')
root=repo/'build/register_pipeline_2026-11-17'
data=repo/'notes/data/2026-11-17-gemm-register-pipeline'
data.mkdir(exist_ok=False)
meta=json.loads((root/'metadata.json').read_text())
for name,original in [('reg4','gemm_self_float4.cu'),('outer','gemm_self_float4_reg8_bpad_double_buffer.cu')]:
    assert hashlib.sha256((repo/'gemm'/original).read_bytes()).hexdigest()==meta[name]['source_sha256']
candidate=repo/'gemm/gemm_self_float4_reg8_bpad_register_pipeline.cu'
assert not candidate.exists()
shutil.copyfile(root/'inner/gemm_self_re.cu',candidate)
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==meta['inner']['source_sha256']

# 验证正式入口可以选择独立候选；测试与实验源使用相同的已核对字节。
cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_120',
     '-DGEMM_SELF_SOURCE="gemm_self_float4_reg8_bpad_register_pipeline.cu"',
     str(repo/'gemm/gemm_self_bench.cu'),'-o',str(root/'public_bench')]
(root/'public_bench_compile.command.json').write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,capture_output=True,text=True)
(root/'public_bench_compile.txt').write_text(p.stdout+p.stderr)
assert p.returncode==0,p.stderr
p=subprocess.run([str(root/'public_bench')],capture_output=True,text=True)
(root/'public_bench_smoke.txt').write_text(p.stdout+p.stderr)
assert p.returncode==0 and 'All sampled checks PASS' in p.stdout

for path in root.iterdir():
    if path.is_file() and path.suffix in ['.txt','.json','.cu']:
        shutil.copyfile(path,data/path.name)
for name in ['reg4','outer','inner']:
    (data/name).mkdir()
    for path in (root/name).glob('*.cu'):shutil.copyfile(path,data/name/path.name)
for name in ['register_pipeline.py','paired_measure.py','record_register_pipeline.py']:
    shutil.copyfile(Path(__file__).with_name(name),data/name)

measure=json.loads((root/'measure_summary.json').read_text())
sustained=json.loads((root/'sustained_summary.json').read_text())[0]
paired=json.loads((root/'paired_summary.json').read_text())
profile=json.loads((root/'profile_summary.json').read_text())
arrays=json.loads((root/'opcode_arrays.json').read_text())
def profile_range(name,key):
    a=[r[key] for r in profile if r['variant']==name]
    return f'{min(a):,.2f}—{max(a):,.2f}'
rows=[]
for r in measure:
    rows.append('| '+', '.join(map(str,r['shape']))+' | '+' | '.join(f"{r[n]['median_ms']:.6f}" for n in ['reg4','outer','inner'])+f" | {r['inner_speedup_vs_outer']:.3f}× |")
stats=[]
for label,key in [
    ('全部warp指令','smsp__inst_executed.sum'),
    ('shared有效读指令','smsp__sass_inst_executed_op_shared_ld.sum'),
    ('普通shared写指令','smsp__sass_inst_executed_op_shared_st.sum'),
    ('普通global读指令','smsp__sass_inst_executed_op_global_ld.sum'),
    ('shared读bank冲突','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
    ('shared写bank冲突','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),
    ('active warp占用 %','sm__warps_active.avg.pct_of_peak_sustained_active'),
    ('active周期发射 %','smsp__issue_active.avg.pct_of_peak_sustained_active'),
    ('FMA elapsed吞吐 %','sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed'),
    ('SM时钟 Hz','sm__cycles_elapsed.avg.per_second')]:
    stats.append(f'| {label} | {profile_range("outer",key)} | {profile_range("inner",key)} |')
ops=[]
for key in ['FFMA','LDS','LDG','LDGSTS','BAR','IMAD','BRA']:
    vals=[arrays[n]['sass__inst_executed_per_opcode'].get(key,0) for n in ['outer','inner']]
    ops.append(f'| {key} | {vals[0]:,.0f} | {vals[1]:,.0f} |')

note='''# 35 · shared→寄存器双缓冲：在现有外层流水线上再推进一层

2026-11-17。用户认为这是核心改动，授权独立尝试。基于 notes/34 最终版，保留严格 FP32 SIMT 路径、128×128 输出、每线程8×8、16×16线程、K_TILE=32、A转置与B分组padding，以及原 global→shared 双缓冲。未更换默认学习文件，未叠加其他参数实验。

## 改了什么

候选 `gemm/gemm_self_float4_reg8_bpad_register_pipeline.cu`。新增 `load_register_inputs` 与 `accumulate_register_inputs`，前者仍用A/B各两个对齐float4读取一组输入，后者把这一组展开为原来的64个FP32累加。

内层K循环使用两组固定命名的寄存器输入：预读k=0；每次成对处理k/k+1，先发出另一组读取，再消费当前组；在最后一对跳过k+2预读，避免读到k=32。累计顺序仍是0到31，K尾部由原shared加载的零填充覆盖。没有运行时寄存器数组下标，也没有新增同步；缓冲是线程私有，外层shared的等待/发布/屏障原样保留。没有添加pragma unroll或寄存器数量限制。

两组输入在源码上共32个float，累加器仍只有一份64个float。最终寄存器数由编译器的活跃区间和调度决定，不等于在原126个上机械加16。

## 正确性与资源

RTX5070，48SM；CUDA13.3，sm_120，-O3 -lineinfo。三版各16组CPU double逐输出检查通过，覆盖小尺寸、非方阵、输出边界、K尾部、单/多stage和向量/标量加载。所有普通、长批次及配对计时前随机数据64点抽查通过；正式公共benchmark选择新文件后的五形状抽查也通过。

| 资源 | 原64/4参考 | 原外层双缓冲128/8 | 加内层双缓冲128/8 |
|---|---:|---:|---:|
| 寄存器/线程 | 39 | 126 | 128 |
| shared/block B | 16896 | 82944 | 82944 |
| local/线程 B | 0 | 0 | 0 |
| spill load/store B | 0/0 | 0/0 | 0/0 |
| 理论block/SM | 5 | 1 | 1 |

两种128/8实现都需要81KiB动态shared opt-in，计时中不包含属性设置。寄存器增加没有进一步降低驻留block数；shared布局和容量完全相同。原有Compute Sanitizer在本机被WDDM初始化失败阻挡，见notes/34；本轮没有重新更改系统配置或声称racecheck通过。

## 同期五形状计时

相同随机数、20预热、3轮×100次启动，每版3进程九轮，交错正反顺序；CUDA events，重复同一缓冲，不清L2。下面取九轮批量平均时间的中位数，不包含分配/拷贝/检查。机器同时承担Windows桌面工作，保留所有波动数据。

| M,N,K | 原64/4 ms | 原外层双缓冲 ms | 加内层双缓冲 ms | 新版对原外层加速比 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

512、长方形与带边缘输入在三次进程中新版均快于原外层版；128小任务波动明显，不能据此排名。64/4在小输入仍明显占优。不要把这一批数据同11月10日的历史绝对耗时拼接比较。

## 2048长批次与短间隔交错复核

原测量入口3轮×1000次、三次进程的九轮中位数：

| 版本 | ms | 三次进程中位数 ms |
|---|---:|---|
'''
for name in ['reg4','outer','inner']:
    note+=f"| {name} | {sustained[name]['median_ms']:.6f} | "+' / '.join(f'{x:.6f}' for x in sustained[name]['process_medians_ms'])+' |\n'
note+='''
这一批存在明显跨进程漂移，不能只报汇总中位数作为稳定提升。为缩短比较间隔，额外将三个原封不动的kernel放进同一程序的不同namespace（编译资源仍为39/126/128寄存器），共用同一2048输入缓冲，每轮以200次为一批，六种先后排列各两遍，启动两个进程，共24轮。每版独立检查并预热20次；保持默认stream及CUDA event测量，不计传输。数据序列跳过前两组形状，和普通入口的2048输入一致。

| 版本 | 24批耗时中位数 ms |
|---|---:|
'''
for name in ['reg4','outer','inner']:note+=f"| {name} | {paired[name]['median_ms']:.6f} |\n"
for base,label in [('outer','原外层双缓冲'),('reg4','原64/4')]:
    p=paired['inner_vs_'+base]
    note+=f"\n新版相对{label}：逐轮配对加速比中位数 **{p['paired_speedup_median']:.4f}×**，{p['faster_rounds']}/{p['rounds']}轮更快。\n"
note+='''
两次进程分别也保持新版优于两种参考的中位数方向。该结果支持当前2048重复缓冲负载的净收益；不是所有输入或所有硬件上的保证。配对比的中位数与两个耗时中位数直接相除不是同一种统计量，表中有意分别保留。

## 实际指令和调度

Nsight Compute application replay、cache-control none、clock-control none，21次未采集调用后采第22次；正反顺序各一次。不使用profile耗时替代普通计时。

| 2048指标 | 原外层双缓冲两轮 | 加内层双缓冲两轮 |
|---|---:|---:|
'''+ '\n'.join(stats)+'''

| InstructionStats warp执行次数（首轮） | 原外层 | 加内层 |
|---|---:|---:|
'''+ '\n'.join(ops)+'''

两版pred-on线程FFMA均8,589,934,592，主体算术量不变。有效shared读取量没有减少，寄存器双缓冲也没有额外复制整块shared。SASS中原版每轮集中4条LDS.128后做64条FFMA；候选在当前组FFMA之间出现下一组的LDS.128，后面的FFMA才使用新寄存器，确实改变了依赖安排。候选仍保留LDGSTS和原外层等待，未丢失global→shared异步路径。

新版总warp指令减少约1.77%，其中成对K循环及编译器地址计算/分支安排也变了。因此可观察到“预读取提前、发射效率上升、净耗时下降”，但没有单独隔离延迟隐藏和循环控制减少各贡献多少。InstructionStats的LDS总数包含不同谓词计数口径，不能把它和pred-on有效shared读取数混为一谈。

候选读取bank冲突从0变为约4万，写冲突仍约629万；没有单凭这些计数去判定收益或主瓶颈。当前结果也不证明已经达到算力上限或贴近cuBLAS。

## 入口与证据

完整源码快照、16用例check、benchmark、长批次入口、同进程交错入口、编译资源、命令、原始JSON/文本、SASS和两轮计数器均保存在 `notes/data/2026-11-17-gemm-register-pipeline/`。二进制和ncu-rep仍只在忽略的build目录。数据目录README给出从快照重编译的入口。

候选独立保存，默认re与原64/4、原外层双缓冲文件未改动。未提交、未推送；没有继续下一项优化。
'''
(repo/'notes/35-gemm-register-pipeline.md').write_text(note)

readme=repo/'README.md';s=readme.read_text();anchor='| [34 · padding版双缓冲]'
lines=s.splitlines();idx=next(i for i,x in enumerate(lines) if x.startswith(anchor))
lines.insert(idx+1,'| [35 · shared到寄存器双缓冲](notes/35-gemm-register-pipeline.md) | 保留外层流水线；2048配对加速约1.11倍，128寄存器且无溢出 |')
readme.write_text('\n'.join(lines)+'\n')
handoff=repo/'HANDOFF.md';s=handoff.read_text();headline='> 最新实验（2026-11-17）：[notes/35 · shared→寄存器双缓冲](notes/35-gemm-register-pipeline.md)。独立候选通过16组完整检查，2048配对比原外层约1.11×、比64/4约1.09×；小输入64/4仍占优。默认文件保留，未提交。\n\n'
s=s.replace('# 学习项目接手记录\n\n','# 学习项目接手记录\n\n'+headline,1)
s+='\n\n2026-11-17：经用户授权，仅在notes/34最终128/8双缓冲上增加shared→寄存器输入双缓冲，固定命名两组输入并成对处理K。寄存器126→128、shared仍82,944B、无溢出，理论仍1block/SM。三版各16组完整检查及所有计时抽查通过；2048同进程24轮配对中新版23轮快于原外层、22轮快于原64/4，配对加速比中位数分别1.1108×/1.0906×。SASS确认下一组LDS穿插到当前FFMA中；乘加/有效shared读取量不变，发射约59%→65%，总指令少约1.77%。不能把循环控制减少和延迟隐藏的贡献完全拆开。独立源码gemm_self_float4_reg8_bpad_register_pipeline.cu，notes/35及对应data留档；未继续其他优化，未提交/推送。\n'
handoff.write_text(s)
(data/'README.md').write_text('''# 2026-11-17 shared→寄存器双缓冲实验数据

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
''')
manifest={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.rglob('*')) if p.is_file()}
(data/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Saved',candidate,repo/'notes/35-gemm-register-pipeline.md',len(manifest),'evidence files',flush=True)
