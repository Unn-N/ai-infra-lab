from pathlib import Path

folder=Path(__file__).parent
original=(folder/'register_pipeline.py').read_text()
script=original.replace('register_pipeline_2026-11-17','warp_layout_2026-11-17')
script=script.replace("PREV=REPO/'build/padded_double_buffer_2026-11-10/double'", "PREV=REPO/'build/register_pipeline_2026-11-17/inner'")
script=script.replace('outer','BASE_TOKEN').replace('inner','warp').replace('BASE_TOKEN','base')
# The previous replacement also touched PREV's folder name; restore the actual saved harness path.
script=script.replace("PREV=REPO/'build/register_pipeline_2026-11-17/warp'", "PREV=REPO/'build/register_pipeline_2026-11-17/inner'")
start=script.index("    base=(REPO/'gemm/")
end=script.index('    metadata={}',start)
prepare='''    base=(REPO/'gemm/gemm_self_float4_reg8_bpad_register_pipeline.cu').read_text()
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
    candidate='// [Codex warp分工候选] 只把每warp输出从16×128改为32×64，两层双缓冲保留。\\n'+base.replace(old,new)
    sources={'reg4':(REPO/'gemm/gemm_self_float4.cu').read_text(),'base':base,'warp':candidate}
'''
script=script[:start]+prepare+script[end:]
(folder/'warp_layout.py').write_text(script)

# Same-process timing harness retains the same data/checks/order and only changes variant labels/root.
paired=(folder/'paired_measure.py').read_text()
paired=paired.replace('register_pipeline_2026-11-17','warp_layout_2026-11-17')
paired=paired.replace('outer','BASE_TOKEN').replace('inner','warp').replace('BASE_TOKEN','base')
(folder/'paired_warp.py').write_text(paired)
print('Generated independent warp-layout experiment and paired measurement scripts.')
