# SPDX-License-Identifier: Apache-2.0
"""Measure complete pretrained reconstruction output when same-shape blocks are skipped.

Uses private, matched first-reset captures; does not load native activations,
modify weights, launch applications or patch a native kernel stream. The
baseline must reproduce an earlier saved reconstruction exactly for each view.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from compare_output_grade import read_capture,read_contract,observed_grade
from summarize_block_costs import REMOVABLE

SOURCE_COMMIT='0ca2deab092fe6f3e331bf4f616271dbc64521d0'
WEIGHTS_SHA='fa6ebc71bc6f91347d51b3368b0ef6e952b6157b6d9a164e7db82cffb88d3143'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def image_metrics(image,target):
    error=image-target;absolute=np.abs(error)
    hp=lambda x:x[1:-1,1:-1]-(x[:-2,1:-1]+x[2:,1:-1]+x[1:-1,:-2]+x[1:-1,2:])*.25
    return {'rgb_mae':float(absolute.mean(dtype=np.float64)),
        'rgb_rmse':float(np.sqrt(np.mean(error.astype(np.float64)**2))),
        'rgb_max_abs':float(absolute.max()),'rgb_p99_abs':float(np.percentile(absolute,99)),
        'highpass_mae':float(np.abs(hp(image)-hp(target)).mean(dtype=np.float64))}


def setup_model(source,weights):
    import torch
    sys.path.insert(0,str(source/'python'))
    from mlxdlss import model as reference
    from mlxdlss.pipeline import NeuralRenderingPipeline
    from fused_norm import FusedNorm
    import batched_ffn
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_accumulation=False
    pipeline=NeuralRenderingPipeline.from_safetensors(weights,device='cuda',precision='fast')
    fused=FusedNorm()
    old_norm=reference.vendor_cosine_normalize;old_softmax=reference.vendor_approximate_softmax
    old_round=reference.e4m3_round_trip;old_publish=reference.vendor_cosine_publish
    reference.vendor_cosine_normalize=lambda x:fused(x) if x.device.type=='cuda' and x.shape[-1]==32 else old_norm(x)
    reference.vendor_approximate_softmax=lambda x:fused.softmax(x) if x.device.type=='cuda' and 2<=x.shape[-1]<=2048 and x.shape[-1]%2==0 else old_softmax(x)
    reference.e4m3_round_trip=lambda x:fused.roundtrip(x) if x.device.type=='cuda' and x.ndim<=8 else old_round(x)
    reference.vendor_cosine_publish=lambda x,scale=None:fused.publish(x,scale) if x.ndim==4 and x.device.type=='cuda' and x.shape[-1]==32 else old_publish(x,scale)
    reference.quadratic_gate_activation=fused.gate
    old_branched=reference.branched_feed_forward;old_split=reference.split_group_feed_forward
    reference.branched_feed_forward=lambda x,**kw:batched_ffn.branched_feed_forward(x,**kw) if x.dtype==torch.float16 else old_branched(x,**kw)
    reference.split_group_feed_forward=lambda x,**kw:batched_ffn.split_group_feed_forward(x,**kw) if x.dtype==torch.float16 else old_split(x,**kw)
    old_bias=reference.recover_attention_bias_layout;cache={}
    def cached_bias(bias):
        key=(bias.data_ptr(),tuple(bias.shape),bias.dtype,bias.device)
        if key not in cache:cache[key]=old_bias(bias)
        return cache[key]
    reference.recover_attention_bias_layout=cached_bias
    return pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--contract-trial',type=Path,required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,metavar=('LABEL','CAPTURE','SAVED_BASELINE'))
    parser.add_argument('--single-blocks',default='all',help='all, none, or comma-separated supported block indices')
    parser.add_argument('--group',nargs=2,action='append',default=[],metavar=('LABEL','BLOCKS'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=900)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    repository=Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repository):raise ValueError('Use a private result path outside the repository.')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Wrong reference source commit.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Reference tracked source has local modifications.')
    if sha(args.weights)!=WEIGHTS_SHA:raise ValueError('Wrong private weights.')
    singles=sorted(REMOVABLE) if args.single_blocks=='all' else [] if args.single_blocks=='none' else sorted({int(x) for x in args.single_blocks.split(',')})
    experiments=[(f'skip-{b}',{b}) for b in singles]
    experiments.extend((label,{int(x) for x in blocks.split(',')}) for label,blocks in args.group)
    if not experiments or len({label for label,_ in experiments})!=len(experiments):raise ValueError('Need distinct experiments.')
    for _,blocks in experiments:
        if not blocks or not blocks<=REMOVABLE:raise ValueError('Cannot skip a structural transition or boundary block.')
    if len({case[0] for case in args.case})!=len(args.case):raise ValueError('Duplicate case labels.')
    fields=read_contract(args.contract_trial);contract_controls,_,contract_hashes=read_capture(args.contract_trial/'capture')
    import torch
    pipeline=setup_model(args.source,args.weights)
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    skipped=set();seen=[]
    for method in ('_window','_split_window','_global'):
        original=getattr(pipeline.model,method)
        def wrapper(value,index,*pos,_original=original,**kw):
            if index in skipped:
                seen.append(index);return value
            return _original(value,index,*pos,**kw)
        setattr(pipeline.model,method,wrapper)
    report={'schema':1,'target_achieved':False,'quality_gate_passed':False,
      'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source_commit':SOURCE_COMMIT,
      'weights_sha256':WEIGHTS_SHA,'native_runtime_modified':False,'uses_captured_native_activations':False,
      'output_extent':[1920,1080],'network_extent':[1920,1152],'noise_frame':0,
      'output_grade_fields':fields,'contract_capture_hashes':contract_hashes,
      'requested_experiments':[{'label':label,'skipped_blocks':sorted(blocks)} for label,blocks in experiments],
      'cases':[],'complete':False,'limitations':[
        'The pretrained reconstruction differs from the native renderer before ablation.',
        'Metrics on a few first-reset views rank diagnostics; they do not establish perceptual, temporal or cross-scene quality.',
        'No native launch is removed: chained kernels require valid synchronization and layouts.',
        'No latency claim; eager diagnostic wall time includes CPU composition, metrics and transfers.',
        'Output grading is an algebraic approximation reused under identical recorded controls.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    started=time.perf_counter()
    with torch.inference_mode():
        for label,capture,baseline_directory in args.case:
            controls,images,hashes=read_capture(Path(capture))
            previous_dir=Path(baseline_directory);previous=json.loads((previous_dir/'comparison.json').read_text())
            if controls!=contract_controls or previous['controls']!=controls or previous['capture_hashes']!=hashes:
                raise ValueError('Mismatched capture, baseline or controls.')
            options={'precision':'fast','noise_frame':0,'batched_ffn':True,'fused_gate':True,
                     'fused_publish':True,'fused_roundtrip':True,'first_block_rounding':False,'mma_first_block':False,
                     'weights_sha256':WEIGHTS_SHA,'network_extent':[1152,1920]}
            if any(previous.get(k)!=v for k,v in options.items()) or previous.get('native_pre_pool_diagnostic') or previous.get('native_pre_stem_diagnostic') or previous.get('cublas_fp16_accumulation',False):
                raise ValueError('Saved baseline uses a different model configuration.')
            source,target=images['color'],images['output']
            automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
            prepared=pipeline.prepare(source,frame_index=0,normalized_style=controls['DLSSNR.Style']/128,
              local_tone_strength=controls['DLSSNR.LocalToneStrength'],local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            prepared.geometry=NetworkGeometry(1920,1080,1920,1152)
            prepared.features=make_features(source,geometry=prepared.geometry,frame_index=0,
              normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
              local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            tensor=torch.from_numpy(prepared.features[None]).cuda().half()
            def run():
                head=pipeline.model(tensor).float().cpu().numpy()[0]
                return pipeline.finish(prepared,head,intensity=1).image
            skipped.clear();seen.clear();ungraded=run()
            saved=np.load(previous_dir/'reconstruction.npy',allow_pickle=False)
            if not np.array_equal(ungraded,saved):
                raise ValueError(f'Baseline does not reproduce saved output for {label}: max_abs={np.max(np.abs(ungraded-saved))}')
            baseline=observed_grade(ungraded,fields)
            case={'label':label,'capture_hashes':hashes,'controls':controls,'saved_baseline_reproduced_exactly':True,
                  'saved_baseline_sha256':sha(previous_dir/'reconstruction.npy'),
                  'baseline_vs_native':image_metrics(baseline,target),'ablations':[]}
            report['cases'].append(case);save()
            print(label,'baseline',case['baseline_vs_native']['rgb_mae'],flush=True)
            for experiment,blocks in experiments:
                if time.perf_counter()-started>args.max_seconds:
                    report['stopped_by_time_limit']=True;save();raise SystemExit('Bounded experiment ended before all cases completed.')
                skipped.clear();skipped.update(blocks);seen.clear()
                candidate=observed_grade(run(),fields)
                if sorted(seen)!=sorted(blocks):raise ValueError('Requested blocks did not run exactly once.')
                record={'label':experiment,'skipped_blocks':sorted(blocks),'vs_native':image_metrics(candidate,target),
                        'vs_unpruned':image_metrics(candidate,baseline)}
                case['ablations'].append(record);save()
                print(label,experiment,'native_mae',record['vs_native']['rgb_mae'],'change_mae',record['vs_unpruned']['rgb_mae'],flush=True)
            skipped.clear();del tensor,prepared,baseline,saved,ungraded
    report['complete']=True;report['diagnostic_wall_seconds']=time.perf_counter()-started
    report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;save()
    print('Complete; no candidate accepted or installed.',flush=True)


if __name__=='__main__':main()
