from pathlib import Path
import subprocess,json,csv,io,shutil,statistics,sys,hashlib

REPO=Path('/home/unn/ai-infra-lab')
ROOT=REPO/'build/warp_layout_2026-11-17'
PREV=REPO/'build/register_pipeline_2026-11-17/inner'
NVCC='/usr/local/cuda/bin/nvcc'
NCU='/usr/local/cuda/bin/ncu'
NAMES=['reg4','base','warp']
METRICS=['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
         'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
         'gpu__time_duration.sum','smsp__inst_executed.sum',
         'smsp__sass_inst_executed_op_shared_ld.sum','smsp__sass_inst_executed_op_shared_st.sum',
         'smsp__sass_inst_executed_op_global_ld.sum',
         'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
         'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
         'sm__warps_active.avg.pct_of_peak_sustained_active',
         'smsp__issue_active.avg.pct_of_peak_sustained_active',
         'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed',
         'sm__cycles_elapsed.avg.per_second']

def run(cmd,tag):
    (ROOT/(tag+'.command.json')).write_text(json.dumps(cmd,indent=2))
    p=subprocess.run(cmd,capture_output=True,text=True)
    (ROOT/(tag+'.txt')).write_text(p.stdout+p.stderr)
    if p.returncode:
        print((p.stdout+p.stderr)[-8000:],flush=True)
        raise RuntimeError((tag,p.returncode))
    return p.stdout

def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2))

def timing_rows(output):
    rows=[]
    for line in output.splitlines():
        p=line.split()
        if len(p)==9 and all(x.isdigit() for x in p[:3]):
            rows.append({'shape':list(map(int,p[:3])),'rounds_ms':list(map(float,p[3:6])),
                         'median_ms':float(p[6]),'sampled_maxerr':float(p[8])})
    assert 'All sampled checks PASS' in output
    return rows

def summarize(runs,count):
    result=[]
    for i in range(count):
        row={'shape':runs[0]['rows'][i]['shape']}
        for name in NAMES:
            selected=[r['rows'][i] for r in runs if r['variant']==name]
            row[name]={'median_ms':statistics.median([v for r in selected for v in r['rounds_ms']]),
                       'all_rounds_ms':[v for r in selected for v in r['rounds_ms']],
                       'process_medians_ms':[r['median_ms'] for r in selected]}
        row['warp_speedup_vs_base']=row['base']['median_ms']/row['warp']['median_ms']
        row['warp_speedup_vs_reg4']=row['reg4']['median_ms']/row['warp']['median_ms']
        result.append(row)
    return result

phase=sys.argv[1]
if phase=='prepare':
    ROOT.mkdir(exist_ok=False)
    base=(REPO/'gemm/gemm_self_float4_reg8_bpad_register_pipeline.cu').read_text()
    assert hashlib.sha256(base.encode()).hexdigest()=='570d2bb49cd98d7ac61217dcf44792da29b85acf182d654b0f3cc02fbffe4898'
    old='    const int local_row = ty * THREAD_TILE, local_col = tx * THREAD_TILE;'
    new="""    // [Codex warp分工实验] 启动仍为16×16线程，global→shared搬运继续使用原tid。
    // 只改输出归属：8个warp排成4行×2列，每warp负责32×64输出。
    // warp内32线程排成4行×8列，每线程仍计算连续8×8。
    static_assert(OUTPUT_TILE == 128 && THREAD_TILE == 8 && BLOCK_THREADS == 256,
                  "This warp mapping requires a 128 tile and 256 eight-by-eight threads.");
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int local_row = (warp_id / 2) * 32 + (lane_id / 8) * THREAD_TILE;
    const int local_col = (warp_id % 2) * 64 + (lane_id % 8) * THREAD_TILE;"""
    assert base.count(old)==1
    candidate='// [Codex warp分工候选] 只把每warp输出从16×128改为32×64，两层双缓冲保留。\n'+base.replace(old,new)
    sources={'reg4':(REPO/'gemm/gemm_self_float4.cu').read_text(),'base':base,'warp':candidate}
    metadata={}
    run(['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw','--format=csv'],'gpu_before')
    run([NVCC,'--version'],'nvcc_version')
    for name in NAMES:
        folder=ROOT/name;folder.mkdir()
        (folder/'gemm_self_re.cu').write_text(sources[name])
        for target in ['bench','check','resource','profile','sustained']:
            shutil.copyfile(PREV/(target+'.cu'),folder/(target+'.cu'))
            run([NVCC,'-O3','-lineinfo','-arch=sm_120','-Xptxas=-v',str(folder/(target+'.cu')),
                 '-o',str(folder/target)],name+'_compile_'+target)
        checks=run([str(folder/'check')],name+'_check')
        assert checks.count('PASS M=')==16
        resource=json.loads(run([str(folder/'resource')],name+'_resource'))
        metadata[name]={'source_sha256':hashlib.sha256(sources[name].encode()).hexdigest(),'resource':resource}
        print(name,'16 full-output checks PASS',resource,flush=True)
    save('metadata.json',metadata)
elif phase in ['measure','sustained']:
    long=phase=='sustained';binary='sustained' if long else 'bench'
    runs=[];counts=dict.fromkeys(NAMES,0)
    orders=[NAMES,list(reversed(NAMES)),['base','reg4','warp']]
    for order in orders:
        for name in order:
            counts[name]+=1
            rows=timing_rows(run([str(ROOT/name/binary)],name+'_'+phase+str(counts[name])))
            assert len(rows)==(1 if long else 5)
            runs.append({'variant':name,'repeat':counts[name],'rows':rows})
            print(phase,name,counts[name],[r['median_ms'] for r in rows],flush=True)
    save(phase+'_runs.json',runs)
    summary=summarize(runs,1 if long else 5);save(phase+'_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
elif phase=='profile':
    results=[];names=['base','warp']
    for repeat,order in enumerate([names,list(reversed(names))],1):
        for name in order:
            tag=name+'_profile'+str(repeat)
            run([NCU,'--profile-from-start','off','--replay-mode','application','--cache-control','none',
                 '--clock-control','none','--section','LaunchStats','--section','InstructionStats',
                 '--metrics',','.join(METRICS),'--export',str(ROOT/tag),'--force-overwrite',
                 str(ROOT/name/'profile'),'2048','2048','2048'],tag+'_capture')
            output=run([NCU,'--import',str(ROOT/(tag+'.ncu-rep')),'--page','raw','--csv',
                        '--print-units','base'],tag+'_raw')
            rows=[r for r in csv.DictReader(io.StringIO(output)) if r.get('ID','').isdigit()]
            assert len(rows)==1
            entry={'variant':name,'repeat':repeat}
            for key,value in rows[0].items():
                if key in METRICS or key.startswith('launch__') or 'spilling' in key:
                    try:entry[key]=float(value.replace(',',''))
                    except (ValueError,AttributeError):entry[key]=value
            for key in METRICS:assert isinstance(entry.get(key),(float,int)),(key,entry.get(key))
            results.append(entry);save('profile_summary.json',results)
            print(name,repeat,{k:entry[k] for k in METRICS},flush=True)
    sys.path.insert(0,'/opt/nvidia/nsight-compute/2026.2.1/extras/python')
    import ncu_report
    arrays={}
    for name in names:
        context=ncu_report.load_report(str(ROOT/(name+'_profile1.ncu-rep')))
        action=context.range_by_idx(0).action_by_idx(0);arrays[name]={}
        for key in ['sass__inst_executed_per_opcode','sass__inst_executed_per_opcode_category',
                    'sass__thread_inst_executed_true_per_opcode']:
            metric=action.metric_by_name(key);ids=metric.correlation_ids()
            arrays[name][key]={str(ids.value(i) if ids else i):metric.value(i) for i in range(metric.num_instances())}
        run([NCU,'--import',str(ROOT/(name+'_profile1.ncu-rep')),'--page','source','--print-source','sass'],name+'_sass')
    save('opcode_arrays.json',arrays)
    print('Saved opcode arrays and SASS',flush=True)
elif phase=='inspect':
    arrays=json.loads((ROOT/'opcode_arrays.json').read_text())
    for name in ['base','warp']:
        a=arrays[name]['sass__inst_executed_per_opcode'];t=arrays[name]['sass__thread_inst_executed_true_per_opcode']
        print(name,json.dumps({'total':sum(a.values()),**{k:a.get(k,0) for k in ['FFMA','LDS','BAR','LDG','LDGSTS']},
                              'thread_FFMA':t.get('FFMA',0)}),flush=True)
        lines=(ROOT/(name+'_sass.txt')).read_text().splitlines()
        selected=[(i,line) for i,line in enumerate(lines) if 'LDS.128' in line]
        if selected:
            first=selected[0][0]
            print('\n'.join(lines[max(0,first-5):first+85]),flush=True)
elif phase=='record':
    print('Record after reviewing final measurements.',flush=True)
else:
    raise ValueError(phase)
