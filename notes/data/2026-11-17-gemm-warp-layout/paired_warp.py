from pathlib import Path
import subprocess,json,statistics,sys

ROOT=Path('/home/unn/ai-infra-lab/build/warp_layout_2026-11-17')
sys.path.insert(0,str(Path(__file__).parent))
# 独立测量辅助：三份kernel放在namespace里，同一进程/同一输入/同一stream交错。
# 不更改任何kernel源文件；所有include先在全局作用域加载。
headers='''#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <array>
#include <random>
#include <vector>
#define GEMM_SELF_BENCH
'''
for name in ['reg4','base','warp']:
    headers+=f'namespace {name} {{\n#include "{name}/gemm_self_re.cu"\n}}\n'
    headers+='#undef GEMM_DYNAMIC_SHARED_BYTES\n#undef CUDA_CHECK\n'
source=headers+r'''
#define CHECK(expr) do { auto s=(expr); if(s!=cudaSuccess){std::fprintf(stderr,"%s line %d\n",cudaGetErrorString(s),__LINE__);std::exit(1);} }while(0)
int main(){
    constexpr int M=2048,N=2048,K=2048,repeats=200;
    CHECK(cudaFuncSetAttribute(base::gemm_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,82944));
    CHECK(cudaFuncSetAttribute(warp::gemm_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,82944));
    std::mt19937 rng(20261107);std::uniform_real_distribution<float> value(-1,1);
    for(size_t i=0;i<size_t(128)*128*2+size_t(512)*512*2;++i)(void)value(rng);
    std::vector<float>a(size_t(M)*K),b(size_t(K)*N),c(size_t(M)*N);
    for(float&x:a)x=value(rng);for(float&x:b)x=value(rng);
    float *da,*db,*dc;CHECK(cudaMalloc(&da,a.size()*4));CHECK(cudaMalloc(&db,b.size()*4));CHECK(cudaMalloc(&dc,c.size()*4));
    CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    auto launch=[&](int v){
        if(v==0)reg4::gemm_tiled<<<dim3(32,32),dim3(16,16)>>>(da,db,dc,M,N,K);
        if(v==1)base::gemm_tiled<<<dim3(16,16),dim3(16,16),82944>>>(da,db,dc,M,N,K);
        if(v==2)warp::gemm_tiled<<<dim3(16,16),dim3(16,16),82944>>>(da,db,dc,M,N,K);
    };
    const char*names[]={"reg4","base","warp"};
    for(int v=0;v<3;++v){
        CHECK(cudaMemset(dc,0xff,c.size()*4));launch(v);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(c.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));
        double worst=0;
        for(int s=0;s<64;++s){int ends[]={0,M/2,M-1};int r=s<9?ends[s/3]:(s*37+11)%M;int q=s<9?ends[s%3]:(s*71+13)%N;
            double ref=0;for(int k=0;k<K;++k)ref+=double(a[size_t(r)*K+k])*b[size_t(k)*N+q];
            double e=std::fabs(double(c[size_t(r)*N+q])-ref);worst=std::max(worst,e);
            if(!std::isfinite(c[size_t(r)*N+q])||e>1e-4+1e-4*std::fabs(ref))return 2;
        }
        std::printf("CHECK %s PASS maxerr=%g\n",names[v],worst);
        for(int i=0;i<20;++i)launch(v);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
    }
    cudaEvent_t start,stop;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&stop));
    // 六种排列各用两次，避免固定先后顺序；一轮中每版200次，比较间隔约0.3秒。
    const int orders[6][3]={{0,1,2},{2,1,0},{1,0,2},{2,0,1},{0,2,1},{1,2,0}};
    for(int round=0;round<12;++round)for(int pos=0;pos<3;++pos){
        int v=orders[round%6][pos];float ms;
        CHECK(cudaEventRecord(start));for(int i=0;i<repeats;++i)launch(v);CHECK(cudaEventRecord(stop));
        CHECK(cudaGetLastError());CHECK(cudaEventSynchronize(stop));CHECK(cudaEventElapsedTime(&ms,start,stop));
        std::printf("TIME %d %s %.6f\n",round,names[v],ms/repeats);std::fflush(stdout);
    }
    CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(stop));CHECK(cudaFree(da));CHECK(cudaFree(db));CHECK(cudaFree(dc));
}
'''
(ROOT/'paired.cu').write_text(source)
cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_120','-Xptxas=-v',str(ROOT/'paired.cu'),'-o',str(ROOT/'paired')]
(ROOT/'paired_compile.command.json').write_text(json.dumps(cmd,indent=2))
p=subprocess.run(cmd,capture_output=True,text=True);(ROOT/'paired_compile.txt').write_text(p.stdout+p.stderr)
if p.returncode:print(p.stdout+p.stderr);sys.exit(p.returncode)
runs=[]
for repeat in range(1,3):
    p=subprocess.run([str(ROOT/'paired')],capture_output=True,text=True)
    (ROOT/f'paired{repeat}.txt').write_text(p.stdout+p.stderr)
    if p.returncode:print(p.stdout+p.stderr);sys.exit(p.returncode)
    assert p.stdout.count(' PASS ')==3
    rounds={}
    for line in p.stdout.splitlines():
        if line.startswith('TIME '):
            _,r,name,ms=line.split();rounds.setdefault(int(r),{})[name]=float(ms)
    assert len(rounds)==12 and all(len(r)==3 for r in rounds.values())
    runs.append({'repeat':repeat,'rounds':rounds})
    print('paired',repeat,{name:statistics.median(r[name] for r in rounds.values()) for name in ['reg4','base','warp']},flush=True)
summary={name:{'median_ms':statistics.median(r[name] for run in runs for r in run['rounds'].values())} for name in ['reg4','base','warp']}
for base in ['base','reg4']:
    ratios=[r[base]/r['warp'] for run in runs for r in run['rounds'].values()]
    summary['warp_vs_'+base]={'paired_speedup_median':statistics.median(ratios),'faster_rounds':sum(x>1 for x in ratios),'rounds':len(ratios),'speedups':ratios}
(ROOT/'paired_runs.json').write_text(json.dumps(runs,indent=2));(ROOT/'paired_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
