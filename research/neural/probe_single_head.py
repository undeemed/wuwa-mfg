# SPDX-License-Identifier: Apache-2.0
"""Compare direct-MMA single-head stages against native decoder and RGB output.

Every candidate computes its own activations from the full-resolution image.
Captured native intermediate features are comparison targets, never inputs.
No application is launched and no native runtime or weights are changed.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from compare_output_grade import observed_grade
from decode_post_inputs import load_prefixes,decode_decoder,decode_skip
from decode_pre_tensor import e4m3_lut
from probe_block_sensitivity import setup_model,sha,image_metrics,SOURCE_COMMIT,WEIGHTS_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--case',nargs=2,action='append',required=True,metavar=('LABEL','TRIAL'))
    parser.add_argument('--baseline-results',type=Path,required=True)
    parser.add_argument('--fused-packing',choices=('none','gate','all'),default='none')
    parser.add_argument('--final-only',action='store_true',help='Compare only the reference and all single-head changes.')
    parser.add_argument('--compare-results',type=Path,help='Require every candidate image to match an earlier run exactly.')
    parser.add_argument('--time-post',action='store_true',help='Time the diagnostic final block and head, not the complete native model.')
    parser.add_argument('--time-model',action='store_true',help='Time all reconstructed blocks and the direct head; excludes application integration.')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.exists() or args.output_directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    labels=[item[0] for item in args.case]
    if len(set(labels))!=len(labels) or any(not re.fullmatch('[a-z0-9-]+',label) for label in labels):
        raise ValueError('Require distinct simple lowercase labels.')
    if sha(args.weights)!=WEIGHTS_SHA or subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Requires the pinned reference and weights.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Modified reference source.')
    old=json.loads((args.baseline_results/'result.json').read_text())
    if not old['complete'] or old['source_commit']!=SOURCE_COMMIT or old['weights_sha256']!=WEIGHTS_SHA:
        raise ValueError('Incomplete or mismatched previous comparison.')
    import torch
    from fused_norm import FusedNorm
    from mma_first_block import MmaFirstBlock
    pipeline=setup_model(args.source,args.weights);kernel=FusedNorm()
    from mlxdlss import model as reference
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    indices=(*range(5),*range(66,71))
    adapters={index:MmaFirstBlock(reference,pipeline.model,kernel,block_index=index,
        fused_packing=args.fused_packing!='none',fused_input_packing=args.fused_packing=='all') for index in indices}
    selected=set();seen=[];observed={};original_window=pipeline.model._window;record_features=True
    def window(value,index,**kw):
        if index==70 and args.time_post and record_features:observed['post-input']=value
        if index in selected:
            if kw['head_count']!=1:raise ValueError('Selected block is not single-head.')
            output=adapters[index](value,attention_mma=True,seed_residual=True,seed_logits=True)
            if index not in (0,70) and kw.get('publish',True):output=reference.e4m3_round_trip(output)
            seen.append(index)
        else:output=original_window(value,index,**kw)
        if index==0 and record_features:observed['skip']=reference.e4m3_round_trip(output).clone()
        if index==69 and record_features:observed['decoder']=output.clone()
        if index==70:observed['post']=output
        return output
    pipeline.model._window=window
    gain=pipeline.model.weight('block70.layer0.out_gain');conv=pipeline.model.weight('block70.layer0.out_conv_weight')
    head_weight=torch.cat((gain,conv),dim=0).contiguous()
    experiments=[('reference',set()),('post-only',{70}),('decoder-last-one',{69,70}),
        ('decoder-last-two',{68,69,70}),('decoder-last-three',{67,68,69,70}),
        ('decoder-last-four',set(range(66,71))),('encoder-single-head',set(range(5))|{70}),
        ('all-single-head',set(indices))]
    if args.final_only:experiments=[item for item in experiments if item[0] in ('reference','all-single-head')]
    def feature_metrics(value,target):
        delta=value.float()-target.float()
        return {'mae':float(delta.abs().mean()),'rmse':float(delta.square().mean().sqrt()),
                'max_abs':float(delta.abs().max()),'exact_value_fraction':float((value==target).float().mean())}
    report={'schema':1,'complete':False,'target_achieved':False,'quality_gate_passed':False,
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source_commit':SOURCE_COMMIT,'weights_sha256':WEIGHTS_SHA,
        'output_extent':[1920,1080],'network_extent':[1920,1152],
        'native_activations_substituted':False,'native_runtime_modified':False,'cases':[],
        'fused_packing':args.fused_packing,
        'limitations':['First-reset same-scene comparisons only; no temporal, perceptual or latency acceptance.',
                       'The native decoder layout is supported by previous evidence, not exhaustively proven.',
                       'Direct MMA is a numerical diagnostic, not an optimized native replacement.']}
    args.output_directory.mkdir(parents=True)
    def save():(args.output_directory/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lut=e4m3_lut().astype(np.float16)
    with torch.inference_mode():
        for label,trial in args.case:
            decoder_raw,skip_raw,data,images=load_prefixes(Path(trial))
            previous=next(item for item in old['cases'] if item['label']==label)
            if previous['capture_hashes']!=data['capture_hashes'] or previous['controls']!=data['controls']:
                raise ValueError('Previous reference belongs to a different frame or controls.')
            native_decoder=torch.from_numpy(lut[decode_decoder(decoder_raw)][None]).cuda()
            native_skip=torch.from_numpy(lut[decode_skip(skip_raw)][None]).cuda()
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
            for name,blocks in experiments:
                selected.clear();selected.update(blocks);seen.clear();observed.clear()
                head=pipeline.model(features)
                if sorted(seen)!=sorted(blocks):raise ValueError('Selected blocks were not each applied once.')
                if 70 in blocks:
                    post=observed['post']
                    head=kernel.mma(post.reshape(-1,32),head_weight).reshape(*post.shape[:-1],4)
                image=observed_grade(pipeline.finish(prepared,head.float().cpu().numpy()[0],intensity=1).image,data['output_fields'])
                result={'blocks':sorted(blocks),'vs_native':image_metrics(image,images['output']),
                    'decoder_vs_native':feature_metrics(observed['decoder'],native_decoder),
                    'skip_vs_native':feature_metrics(observed['skip'],native_skip)}
                if name in ('reference','post-only'):
                    old_name='reference-inputs' if name=='reference' else 'reference-inputs-mma-all'
                    saved=np.load(args.baseline_results/(label+'-'+old_name+'.npy'),allow_pickle=False)
                    if not np.array_equal(image,saved):raise ValueError('Saved complete-image baseline did not reproduce.')
                    if result['vs_native']!=previous['variants'][old_name]['vs_native']:
                        raise ValueError('Saved native image metrics differ.')
                    result['previous_output_reproduced_exactly']=True
                if name=='reference':baseline=image.copy()
                if args.compare_results:
                    earlier=json.loads((args.compare_results/'result.json').read_text())
                    before=next(item for item in earlier['cases'] if item['label']==label)
                    if not earlier['complete'] or before['capture_hashes']!=data['capture_hashes'] or before['controls']!=controls:
                        raise ValueError('Comparison run has different input or controls.')
                    if not np.array_equal(image,np.load(args.compare_results/(label+'-'+name+'.npy'),allow_pickle=False)):
                        raise ValueError('Candidate differs from the earlier complete image.')
                    result['previous_candidate_reproduced_exactly']=True
                result['vs_reference']=image_metrics(image,baseline)
                np.save(args.output_directory/(label+'-'+name+'.npy'),image)
                record['variants'][name]=result
                print(label,name,'RGB',result['vs_native']['rgb_mae'],'decoder',result['decoder_vs_native']['mae'],flush=True)
                del head,image
            if args.time_post:
                from test_output_grade import graph_measure
                timings={};post_input=observed['post-input']
                for packing in ('none','gate','all'):
                    adapter=MmaFirstBlock(reference,pipeline.model,kernel,block_index=70,
                        fused_packing=packing!='none',fused_input_packing=packing=='all')
                    def operation(value):
                        post=adapter(value,attention_mma=True,seed_residual=True,seed_logits=True)
                        return kernel.mma(post.reshape(-1,32),head_weight).reshape(*post.shape[:-1],4)
                    expected=operation(post_input)
                    times,output=graph_measure(operation,post_input)
                    if not torch.equal(expected,output):raise ValueError('Diagnostic graph differs from eager output.')
                    if packing=='none':prior_output=expected
                    elif not torch.equal(prior_output,expected):raise ValueError('Fused packing changes diagnostic final-block output.')
                    timings[packing]={'graph':times,'graph_matches_eager':True}
                record['diagnostic_post_timing']={'shape':list(post_input.shape),'variants':timings,
                    'prior_fused_outputs_equal':True,'scope':'Diagnostic block 70 and FP16 head only, not full model or native GPU time.'}
                del post_input,prior_output,expected,output
            if args.time_model:
                from test_output_grade import graph_measure
                if selected!=set(indices):raise ValueError('Whole-network timing requires the all-single-head candidate.')
                timing={};saved_adapters=adapters.copy();record_features=False
                try:
                    for packing in ('none','all'):
                        adapters.clear()
                        adapters.update({index:MmaFirstBlock(reference,pipeline.model,kernel,block_index=index,
                            fused_packing=packing=='all',fused_input_packing=packing=='all') for index in indices})
                        def operation(value):
                            pipeline.model(value)
                            post=observed['post']
                            return kernel.mma(post.reshape(-1,32),head_weight).reshape(*post.shape[:-1],4)
                        expected=operation(features)
                        times,output=graph_measure(operation,features)
                        if not torch.equal(expected,output):raise ValueError('Whole-network graph differs from eager output.')
                        if packing=='none':prior_output=expected
                        elif not torch.equal(prior_output,expected):raise ValueError('Whole-network packing fusion changes output.')
                        timing[packing]={'graph':times,'graph_matches_eager':True}
                    record['reconstructed_model_timing']={'variants':timing,'prior_fused_outputs_equal':True,
                        'scope':'All 71 reconstructed blocks from prepared input features plus the direct head. Includes the unused original head; excludes feature preparation, output grading and application integration. Native GPU time is not measured.'}
                finally:
                    record_features=True;adapters.clear();adapters.update(saved_adapters)
                del saved_adapters,prior_output,expected,output
            report['cases'].append(record);save()
            del native_decoder,native_skip,features,prepared,baseline
            observed.clear()
    report['complete']=True;report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;save()
    print('Complete; no native activations substituted or native speedup claimed.',flush=True)


if __name__=='__main__':main()
