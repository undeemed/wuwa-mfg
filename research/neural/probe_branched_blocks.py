# SPDX-License-Identifier: Apache-2.0
"""Compare branched-MMA hypotheses against full native images and decoder targets.

All candidates compute their own activations. Native intermediate captures are
comparison targets only. No application is launched or runtime changed.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from compare_output_grade import observed_grade
from decode_post_inputs import load_prefixes,decode_decoder
from decode_pre_tensor import e4m3_lut
from probe_block_sensitivity import setup_model,sha,image_metrics,SOURCE_COMMIT,WEIGHTS_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--case',nargs=2,action='append',required=True,metavar=('LABEL','TRIAL'))
    parser.add_argument('--baseline-results',type=Path,required=True)
    parser.add_argument('--compact-bias',action='store_true')
    parser.add_argument('--compare-results',type=Path,help='Require every candidate image to match a previous run exactly.')
    parser.add_argument('--time-block',action='store_true',help='Time diagnostic block 5 with expanded and compact bias.')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.exists() or args.output_directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    labels=[x[0] for x in args.case]
    if len(set(labels))!=len(labels) or any(not re.fullmatch('[a-z0-9-]+',x) for x in labels):raise ValueError('Invalid labels.')
    if sha(args.weights)!=WEIGHTS_SHA or subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Requires pinned source and weights.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():raise ValueError('Modified reference.')
    previous=json.loads((args.baseline_results/'result.json').read_text())
    if not previous['complete']:raise ValueError('Incomplete previous comparison.')
    import torch
    from fused_norm import FusedNorm
    from mma_first_block import MmaFirstBlock
    from mma_branched_block import MmaBranchedBlock,BRANCHED
    pipeline=setup_model(args.source,args.weights);kernel=FusedNorm()
    from mlxdlss import model as ref
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    single={index:MmaFirstBlock(ref,pipeline.model,kernel,block_index=index,fused_packing=True,fused_input_packing=True)
            for index in (*range(5),*range(66,71))}
    branched={index:MmaBranchedBlock(ref,pipeline.model,kernel,block_index=index,compact_bias=args.compact_bias) for index in sorted(BRANCHED)}
    original_window=pipeline.model._window;selected=set();seen=[];observed={};mode='all'
    def window(value,index,**kw):
        if index==5 and args.time_block:observed['timing-input']=value
        if index in selected:
            if kw['head_count']!=branched[index].heads:raise ValueError('Mismatched head count.')
            output=branched[index](value,ffn_mma=mode in ('ffn','all'),attention_mma=mode in ('attention','all'))
            seen.append(index)
        elif index in single:
            output=single[index](value,attention_mma=True,seed_residual=True,seed_logits=True)
        else:return original_window(value,index,**kw)
        if index not in (0,70) and kw.get('publish',True):output=ref.e4m3_round_trip(output)
        if index==69:observed['decoder']=output.clone()
        if index==70:observed['post']=output
        return output
    pipeline.model._window=window
    head_weight=torch.cat((pipeline.model.weight('block70.layer0.out_gain'),pipeline.model.weight('block70.layer0.out_conv_weight')),dim=0).contiguous()
    experiments=[('single-head-baseline',set(),'all'),('branched-ffn',BRANCHED,'ffn'),
        ('branched-attention',BRANCHED,'attention'),('branched-all',BRANCHED,'all'),
        ('encoder-branched',set(range(5,23)),'all'),('decoder-branched',set(range(48,66)),'all')]
    experiments.extend((f'heads-{heads}',{index for index,block in branched.items() if block.heads==heads},'all') for heads in (2,4,8))
    report={'schema':1,'complete':False,'target_achieved':False,'quality_gate_passed':False,
        'source_commit':SOURCE_COMMIT,'weights_sha256':WEIGHTS_SHA,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'output_extent':[1920,1080],'network_extent':[1920,1152],'native_activations_substituted':False,
        'native_runtime_modified':False,'compact_bias':args.compact_bias,'cases':[],
        'limitations':['Native arithmetic equivalence is a hypothesis; this is not a production kernel.',
            'First-reset same-scene comparisons do not establish temporal or perceptual quality.',
            'No latency improvement is claimed by these image comparisons.']}
    args.output_directory.mkdir(parents=True)
    def save():(args.output_directory/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lut=e4m3_lut().astype(np.float16)
    with torch.inference_mode():
        for label,trial in args.case:
            decoder_raw,_,data,images=load_prefixes(Path(trial))
            old=next(x for x in previous['cases'] if x['label']==label)
            if old['capture_hashes']!=data['capture_hashes'] or old['controls']!=data['controls']:raise ValueError('Mismatched previous frame.')
            native_decoder=torch.from_numpy(lut[decode_decoder(decoder_raw)][None]).cuda()
            controls=data['controls'];source=images['color']
            automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
            prepared=pipeline.prepare(source,frame_index=0,normalized_style=controls['DLSSNR.Style']/128,
                local_tone_strength=controls['DLSSNR.LocalToneStrength'],local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            prepared.geometry=NetworkGeometry(1920,1080,1920,1152)
            prepared.features=make_features(source,geometry=prepared.geometry,frame_index=0,
                normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
                local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            features=torch.from_numpy(prepared.features[None]).cuda().half()
            record={'label':label,'capture_hashes':data['capture_hashes'],'controls':controls,
                'native_input_sha256':data['input_sha256'],'variants':{}}
            for name,blocks,mode in experiments:
                selected.clear();selected.update(blocks);seen.clear();observed.clear()
                pipeline.model(features)
                if sorted(seen)!=sorted(blocks):raise ValueError('Selected blocks were not each applied once.')
                post=observed['post']
                head=kernel.mma(post.reshape(-1,32),head_weight).reshape(*post.shape[:-1],4)
                image=observed_grade(pipeline.finish(prepared,head.float().cpu().numpy()[0],intensity=1).image,data['output_fields'])
                delta=observed['decoder'].float()-native_decoder.float()
                result={'blocks':sorted(blocks),'mode':mode,'vs_native':image_metrics(image,images['output']),
                    'decoder_vs_native':{'mae':float(delta.abs().mean()),'rmse':float(delta.square().mean().sqrt()),'max_abs':float(delta.abs().max())}}
                if name=='single-head-baseline':
                    if not np.array_equal(image,np.load(args.baseline_results/(label+'-all-single-head.npy'),allow_pickle=False)):
                        raise ValueError('Saved single-head baseline did not reproduce exactly.')
                    if result['vs_native']!=old['variants']['all-single-head']['vs_native']:raise ValueError('Baseline metric mismatch.')
                    result['previous_output_reproduced_exactly']=True;baseline=image.copy()
                result['vs_single_head_baseline']=image_metrics(image,baseline)
                if args.compare_results:
                    earlier=json.loads((args.compare_results/'result.json').read_text())
                    old_case=next(item for item in earlier['cases'] if item['label']==label)
                    if not earlier['complete'] or old_case['capture_hashes']!=data['capture_hashes'] or old_case['controls']!=controls:
                        raise ValueError('Mismatched prior candidate inputs.')
                    if not np.array_equal(image,np.load(args.compare_results/(label+'-'+name+'.npy'),allow_pickle=False)):
                        raise ValueError('Candidate image differs from prior output.')
                    result['previous_candidate_reproduced_exactly']=True
                record['variants'][name]=result
                np.save(args.output_directory/(label+'-'+name+'.npy'),image)
                print(label,name,'RGB',result['vs_native']['rgb_mae'],'decoder',result['decoder_vs_native']['mae'],flush=True)
                del head,image,delta,post
            if args.time_block:
                from test_output_grade import graph_measure
                times={};value=observed['timing-input']
                for compact in (False,True):
                    adapter=MmaBranchedBlock(ref,pipeline.model,kernel,block_index=5,compact_bias=compact)
                    expected=adapter(value)
                    timing,actual=graph_measure(adapter,value)
                    if not torch.equal(expected,actual):raise ValueError('Graph differs from eager block output.')
                    if not compact:expanded_output=expected
                    elif not torch.equal(expected,expanded_output):raise ValueError('Compact bias changes block output.')
                    times['compact' if compact else 'expanded']={'graph':timing,'graph_matches_eager':True}
                record['block5_timing']={'input_shape':list(value.shape),'variants':times,'expanded_compact_equal':True,
                    'scope':'Diagnostic two-head block 5 only; not a native or full-renderer timing.'}
                del value,expanded_output,expected,actual
            report['cases'].append(record);save()
            del native_decoder,features,prepared,baseline
            observed.clear()
    report['complete']=True;report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;save()
    print('Complete; no native activations substituted.',flush=True)


if __name__=='__main__':main()
