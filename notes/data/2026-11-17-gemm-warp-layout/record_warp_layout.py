from pathlib import Path
import json,hashlib,shutil,subprocess

repo=Path('/home/unn/ai-infra-lab')
root=repo/'build/warp_layout_2026-11-17'
data=repo/'notes/data/2026-11-17-gemm-warp-layout'
meta=json.loads((root/'metadata.json').read_text())
for name,filename in [('reg4','gemm_self_float4.cu'),('base','gemm_self_float4_reg8_bpad_register_pipeline.cu')]:
    assert hashlib.sha256((repo/'gemm'/filename).read_bytes()).hexdigest()==meta[name]['source_sha256']
candidate=repo/'gemm/gemm_self_float4_reg8_bpad_warp32x64.cu'
assert not candidate.exists()
shutil.copyfile(root/'warp/gemm_self_re.cu',candidate)
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==meta['warp']['source_sha256']

cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_120',
     '-DGEMM_SELF_SOURCE="gemm_self_float4_reg8_bpad_warp32x64.cu"',
     str(repo/'gemm/gemm_self_bench.cu'),'-o',str(root/'public_bench')]
(root/'public_bench_compile.command.json').write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,capture_output=True,text=True)
(root/'public_bench_compile.txt').write_text(p.stdout+p.stderr)
assert p.returncode==0,p.stderr
p=subprocess.run([str(root/'public_bench')],capture_output=True,text=True)
(root/'public_bench_smoke.txt').write_text(p.stdout+p.stderr)
assert p.returncode==0 and 'All sampled checks PASS' in p.stdout

measure=json.loads((root/'measure_summary.json').read_text())
paired=json.loads((root/'paired_summary.json').read_text())
profile=json.loads((root/'profile_summary.json').read_text())
arrays=json.loads((root/'opcode_arrays.json').read_text())
for n in ['base','warp']:
    assert arrays[n]['sass__thread_inst_executed_true_per_opcode']['FFMA']==8589934592
    assert 'LDS.128' in (root/(n+'_sass.txt')).read_text()
    assert 'LDGSTS.E.BYPASS.128' in (root/(n+'_sass.txt')).read_text()
rows=[]
for r in measure:
    rows.append('| '+', '.join(map(str,r['shape']))+' | '+' | '.join(f"{r[n]['median_ms']:.6f}" for n in ['reg4','base','warp'])+f" | {r['warp_speedup_vs_base']:.3f}× |")
def span(n,key):
    v=[x[key] for x in profile if x['variant']==n]
    return f'{min(v):,.2f}—{max(v):,.2f}'
stats=[]
for label,key in [
    ('全部warp指令','smsp__inst_executed.sum'),
    ('有效shared读取指令','smsp__sass_inst_executed_op_shared_ld.sum'),
    ('shared读取服务wavefront','l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'),
    ('shared读取bank冲突','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
    ('shared写入bank冲突','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),
    ('global输出写入sector','l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'),
    ('active周期发射 %','smsp__issue_active.avg.pct_of_peak_sustained_active'),
    ('FMA elapsed吞吐 %','sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed'),
    ('SM时钟 Hz','sm__cycles_elapsed.avg.per_second')]:
    stats.append(f'| {label} | {span("base",key)} | {span("warp",key)} |')
ops=[]
for key in ['FFMA','LDS','LDG','LDGSTS','BAR','IMAD','BRA']:
    v=[arrays[n]['sass__inst_executed_per_opcode'].get(key,0) for n in ['base','warp']]
    ops.append(f'| {key} | {v[0]:,.0f} | {v[1]:,.0f} |')

note='''# 36 · warp输出分工：16×128改为32×64，尚未证实净加速

2026-11-17。用户同意尝试，并明确对效果保持质疑，把本次作为扩充工具箱。实验基于notes/35最终shared→寄存器双缓冲版，只修改warp/线程到输出位置的映射；不改变两层双缓冲、输出128×128、K32、每线程8×8、线程总数256、shared布局及精度。

## 假设与唯一源码改动

当前启动形状仍是16×16线程，tid=ty*16+tx。原映射local_row=ty*8、local_col=tx*8，导致每warp两行×16列线程，负责16×128输出。候选继续使用相同tid执行global→shared加载，只改变计算/写回的输出归属：

```cpp
const int warp_id = tid / 32;
const int lane_id = tid % 32;
const int local_row = (warp_id / 2) * 32 + (lane_id / 8) * 8;
const int local_col = (warp_id % 2) * 64 + (lane_id % 8) * 8;
```

8个warp排成4行×2列，各负责32×64；每warp内4行×8列线程。每线程的8×8仍连续且16B对齐，B不跨padding分组；8个warp恰好无重叠覆盖128×128输出。global加载、A转置发布、B异步、内层寄存器轮换、同步和累计顺序完全保留。代码新增固定前提static_assert，独立文件gemm/gemm_self_float4_reg8_bpad_warp32x64.cu。

对某一个k，一个warp所需的不同输入值由16A+128B=144变为32A+64B=96，复用更均衡。但每线程仍读取8A+8B，shared指令数并不会因不同数值数量减少而自动下降。广播、bank、宽访问的服务方式才决定实际代价；输出写回也会因归属变化而改变地址组合。不能把144→96直接解读成带宽流量或耗时减少1/3。

## 正确性与资源

RTX5070/48SM，CUDA13.3，sm_120，-O3 -lineinfo。reg4参考、notes/35基线、新warp候选各16组CPU double完整输出检查通过；所有五形状/配对计时前64点随机输入抽查通过。正式公共benchmark选择候选的五形状smoke也通过。

| 资源 | notes/35基线 | warp32×64 |
|---|---:|---:|
| 线程/block | 256 | 256 |
| 寄存器/线程 | 128 | 128 |
| 动态shared/block B | 82944 | 82944 |
| local/线程 B | 0 | 0 |
| spill load/store B | 0/0 | 0/0 |
| 理论block/SM | 1 | 1 |

无新增同步。racecheck历史受WDDM初始化失败阻挡，见notes/34，本轮未重试或声称已通过。功能检查不是形式化证明。

## 同期五形状计时

相同随机数、20预热、3轮×100次启动，每版3进程九轮，正反顺序交错；CUDA events、重复缓冲、不清L2。下表取九轮批量平均时间中位数，排除分配、传输、检查和动态shared设置。

| M,N,K | 原64/4 ms | notes/35基线 ms | warp32×64 ms | 候选相对基线加速比 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

各形状都没有三次进程一致的候选胜负方向。128小任务尤其波动大，其汇总改善不作稳定收益。带边缘形状汇总约1.09倍，但第二次进程反向，也未单独确认为稳定改善。

## 2048同进程交错复核

沿用notes/35的同进程三版namespace入口，只替换源文件/标签；共同输入与默认stream，每批200次，六种排列各两遍，两个进程共24轮。每版先检查和20次预热。编译仍为39/128/128寄存器。未再叠加长批次或调参。

| 版本 | 24批耗时中位数 ms |
|---|---:|
'''
for n in ['reg4','base','warp']:note+=f"| {n} | {paired[n]['median_ms']:.6f} |\n"
p=paired['warp_vs_base']
note+=f"\n候选对notes/35基线的逐轮配对加速比中位数 **{p['paired_speedup_median']:.4f}×**，仅{p['faster_rounds']}/24轮更快。第一进程两版中位数1.405366/1.397422ms；第二进程1.363387/1.363588ms，方向也不一致。\n"
note+='''
结论：此次32×64 warp映射在当前2048负载上基本持平，尚未证实稳定净加速。它相对原64/4的优势包含notes/35已经实现的流水线收益，不能当成本次warp重排的收益。保留notes/35为这条实验路线的优先基线。

## 实际访问与指令证据

Nsight Compute application replay、cache-control none、clock-control none；21次调用后采第22次，正反顺序各一轮。首轮base有明显时钟偏低，保留原值，不用profile时间判断加速；普通配对计时是性能依据。

| 2048指标 | notes/35基线两轮 | warp32×64两轮 |
|---|---:|---:|
'''+ '\n'.join(stats)+'''

| InstructionStats warp执行次数（首轮） | 基线 | 候选 |
|---|---:|---:|
'''+ '\n'.join(ops)+'''

两版pred-on线程FFMA均8,589,934,592，SASS都保留LDS.128与LDGSTS.E.BYPASS.128。有效shared读取指令完全相同，shared服务wavefront约5037万基本相同，global输出sector完全相同；总warp指令反而增加约0.40%。读冲突有小幅下降，但绝对量远小于服务wavefront，写冲突也基本不变。没有观察到“不同输入值减少1/3”转化为相应的shared服务量下降。

这些证据解释了为什么不能仅按warp形状更方正预测收益；它们并未唯一证明没有收益的全部原因，也不能推论warp级分工普遍无效。更深入的warp级访问组织可能还要联合设计线程内布局、shared布局或读取/广播方式，那属于后续独立实验，不在本次授权改动中叠加。

## 文件与复现

对应原始数据目录notes/data/2026-11-17-gemm-warp-layout，包含三版完整快照与辅助入口、命令、编译/校验日志、全部计时、配对数据、两轮计数器、SASS和opcode数组。二进制与ncu-rep仅保留在忽略的build目录。

只新增独立候选、学习记录和入口链接；notes/35基线、默认re及旧参考源均未改动。未提交、未推送。
'''
(repo/'notes/36-gemm-warp-layout.md').write_text(note)
readme=repo/'README.md';lines=readme.read_text().splitlines()
i=next(i for i,s in enumerate(lines) if s.startswith('| [35 ·'))
lines.insert(i+1,'| [36 · warp输出分工](notes/36-gemm-warp-layout.md) | 每warp 16×128→32×64；2048配对约1.004倍，未证实稳定收益 |')
readme.write_text('\n'.join(lines)+'\n')
handoff=repo/'HANDOFF.md';s=handoff.read_text()
prefix='> 最新实验（2026-11-17）：[notes/36 · warp输出分工](notes/36-gemm-warp-layout.md)。只把每warp输出16×128改为32×64；2048配对基本持平，未证实净收益。当前优先保留notes/35双层流水线基线；独立候选留档，未提交。\n\n'
s=s.replace('# 学习项目接手记录\n\n','# 学习项目接手记录\n\n'+prefix,1)
s+='\n\n2026-11-17：用户对warp重排收益保持质疑，授权单项尝试。基于notes/35只改输出归属为warp32×64，16×16线程启动及global加载不变。三版16用例完整检查通过，无溢出；候选仍128regs、81KiB shared、1block/SM。2048同进程24轮配对加速中位数1.0043×，仅13轮更快，尚未证实净收益。有效shared读取及global输出sector不变，shared wavefront基本不变，总指令+约0.40%。记录notes/36与对应data；独立候选gemm_self_float4_reg8_bpad_warp32x64.cu，notes/35继续作优先基线。未继续组合布局优化，未提交/推送。\n'
handoff.write_text(s)

data.mkdir(exist_ok=False)
for path in root.iterdir():
    if path.is_file() and path.suffix in ['.txt','.json','.cu']:shutil.copyfile(path,data/path.name)
for n in ['reg4','base','warp']:
    (data/n).mkdir()
    for path in (root/n).glob('*.cu'):shutil.copyfile(path,data/n/path.name)
for n in ['warp_layout.py','paired_warp.py','record_warp_layout.py','make_warp_experiment.py']:
    shutil.copyfile(Path(__file__).with_name(n),data/n)
(data/'README.md').write_text('''# 2026-11-17 warp分工对照

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
''')
manifest={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.rglob('*')) if p.is_file()}
(data/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Saved candidate, note36 and',len(manifest),'evidence files.')
print('Five-shape summary:',[(r['shape'],r['base']['median_ms'],r['warp']['median_ms']) for r in measure])
print('Profile clock ranges:',span('base','sm__cycles_elapsed.avg.per_second'),span('warp','sm__cycles_elapsed.avg.per_second'))
