# SPDX-License-Identifier: Apache-2.0
"""Map one pinned native launch schedule to inferred pretrained model blocks.

The correspondence is inferred from order and kernel families. These are
instrumented costs, not removable latency: chained kernels synchronize through
shared counters and cannot safely be deleted from the native submission stream.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REMOVABLE = (set(range(1,4)) | set(range(5,8)) | set(range(9,14)) |
             set(range(15,22)) | set(range(23,30)) | set(range(31,39)) |
             set(range(40,48)) | set(range(49,56)) | set(range(57,62)) |
             set(range(63,66)) | set(range(67,70)))


def expected_schedule():
    rows=[]
    def add(block,name):rows.append({'chain':len(rows)+1,'block':block,'name':name})
    add(None,'cg2r_copy_kernel');add(None,'cc_cb_clear')
    add(0,'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8')
    for start,end,heads in [(1,4,1),(5,8,2),(9,14,4),(15,22,8)]:
        for block in range(start,end+1):
            suffix='inpview_tilesync' if block==start else 'ds_wait' if block==end else 'chained'
            add(block,f'cc_tinlayout_fused_swin_{heads}h_{heads*32}_{heads}_{suffix}_fp8')
    for block in range(23,31):
        add(block,'cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8' if block==23 else 'cc_split_swin_16h_ffwd_512_chained_fp8')
        add(block,'cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8' if block==23 else 'cc_split_swin_16h_ffwd_proj_512_chained_fp8')
        add(block,'cc_split_swin_16h_qkv_512_chained_fp8')
        add(block,'cc_split_swin_16h_proj_pool_512_chained_fp8' if block==30 else 'cc_split_swin_16h_proj_512_chained_fp8')
    add(30,'cc_split_swin_16h_final_head_512_wait_fp8')
    add(None,'cc_vit_1d_repack_2d_to_1d_fp8')
    for block in range(31,39):
        add(block,'cc_vit_1d_ffn_expand_publish_fp8' if block==31 else 'cc_vit_1d_ffn_expand_chained_fp8')
        for operation in ('ffn_contract','qkv','attention'):
            add(block,f'cc_vit_1d_{operation}_chained_fp8')
        add(block,'cc_vit_1d_projection_wait_fp8' if block==38 else 'cc_vit_1d_projection_chained_fp8')
    add(None,'cc_vit_1d_repack_1d_to_2d_fp8')
    add(39,'cc_dec_input_upsample_1024_512_tilesync_fp8')
    for block in range(40,48):
        for operation in ('ffwd','ffwd_proj','qkv'):
            add(block,f'cc_split_swin_16h_{operation}_512_chained_fp8')
        add(block,'cc_split_swin_16h_proj_512_outview_wait_fp8' if block==47 else 'cc_split_swin_16h_proj_512_chained_fp8')
    for start,end,heads in [(48,55,8),(56,61,4),(62,65,2),(66,69,1)]:
        for block in range(start,end+1):
            suffix='upsample_tilesync' if block==start else 'outview_wait' if block==end else 'chained'
            add(block,f'cc_tinlayout_fused_swin_{heads}h_{heads*32}_{heads}_{suffix}_fp8')
    add(70,'cc_tinlayout_fused_post_block_swin_1h_32_fp8')
    add(None,'cg2r_copy_kernel');add(None,'cg2r_post_process_kernel')
    if len(rows)!=158 or {x['block'] for x in rows if x['block'] is not None}!=set(range(71)):
        raise ValueError('Internal schedule is incomplete.')
    return rows


def summarize(trace,after_frame=600):
    schedule=expected_schedule();frames=defaultdict(list)
    with trace.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if int(row['frame'])>after_frame:frames[int(row['frame'])].append(row)
    if not frames:raise ValueError('No post-warmup frames.')
    samples=defaultdict(list);totals=[];kept=[];discarded=[]
    for frame,rows in sorted(frames.items()):
        if len(rows)!=158 or any(int(row['status'])!=0 for row in rows):
            discarded.append(frame);continue
        costs=defaultdict(float)
        for expected,row in zip(schedule,rows):
            if (int(row['chain'])!=expected['chain'] or int(row['kernels'])!=1
                    or row['names']!=expected['name']):
                raise ValueError('Trace does not match the pinned 158-chain schedule.')
            ms=float(row['gpu_ms'])
            if not math.isfinite(ms) or ms<0:raise ValueError('Invalid GPU interval.')
            costs[expected['block']]+=ms
        kept.append(frame);totals.append(sum(costs.values()))
        for block,value in costs.items():samples[block].append(value)
    if not kept:raise ValueError('No complete successful matching frames.')
    def describe(values):
        return {'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'samples_ms':values}
    return {'schema':1,'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'frames':kept,'discarded_frames':discarded,'chains_per_evaluation':158,
        'block_mapping_is_inferred':True,'instrumentation_overhead_not_subtracted':True,
        'native_graph_modified':False,'target_achieved':False,'quality_gate_passed':False,
        'total_intervals':describe(totals),'auxiliary_intervals':describe(samples[None]),
        'blocks':[{'block':b,'same_shape_ablation_supported_in_reconstruction':b in REMOVABLE,
                   'chains':[r['chain'] for r in schedule if r['block']==b],
                   'kernel_names':[r['name'] for r in schedule if r['block']==b],
                   **describe(samples[b])} for b in range(71)],
        'limitations':['Block correspondence is inferred from the recovered model schedule and exact native kernel sequence.',
          'Kernel intervals include instrumentation and synchronization; block medians are not additive speedup guarantees.',
          'Native chained launches cannot be removed directly: counter publication, layout and downstream waits must remain valid.',
          'Same-shape ablation is allowed only in the isolated reconstruction and does not establish image quality.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after-frame',type=int,default=600)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=summarize(args.trace,args.after_frame)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'frames':len(report['frames']),'blocks':len(report['blocks']),
       'top_ten':sorted([{'block':r['block'],'median_ms':r['median_ms']} for r in report['blocks']],key=lambda x:x['median_ms'],reverse=True)[:10]},indent=2))


if __name__=='__main__':main()
