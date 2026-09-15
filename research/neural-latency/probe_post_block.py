# SPDX-License-Identifier: Apache-2.0
"""Localize reconstruction error using captured inputs to the final native block.

Diagnostic substitutions only, not independent models or latency benchmarks.
Requires the exact pinned reference and private weights; never launches an app.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from compare_output_grade import observed_grade
from decode_post_inputs import load_prefixes,decode_decoder,decode_skip,DECODER_BYTES
from decode_pre_tensor import e4m3_lut
from probe_block_sensitivity import setup_model,sha,image_metrics,SOURCE_COMMIT,WEIGHTS_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,metavar=('LABEL','TRIAL','BASELINE_OR_DASH'))
    parser.add_argument('--prefix-check-trial',type=Path)
    parser.add_argument('--mma-post',action='store_true',help='Also test direct FP8 MMA and seeded residual/bias in block 70.')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    if args.output_directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Raw features and outputs must stay private.')
    if sha(args.weights)!=WEIGHTS_SHA or subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Requires the pinned reference and weights.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Modified reference source.')
    import torch
    from fused_norm import FusedNorm
    pipeline=setup_model(args.source,args.weights);kernel=FusedNorm()
    from mlxdlss import model as reference
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    if args.mma_post:
        from mma_first_block import MmaFirstBlock
        mma_post=MmaFirstBlock(reference,pipeline.model,kernel,block_index=70)
    observed={};original_window=pipeline.model._window
    def window(value,index,**kw):
        output=original_window(value,index,**kw)
        if index==0:observed['skip']=reference.e4m3_round_trip(output).clone()
        if index==69:observed['decoder']=output.clone()
        return output
    pipeline.model._window=window
    lut=e4m3_lut().astype(np.float16)
    def tensor(codes):return torch.from_numpy(lut[codes][None]).cuda()
    def feature_metrics(a,b):
        af,bf=a.float(),b.float();difference=af-bf
        ac=af-af.mean();bc=bf-bf.mean()
        return {'mae':float(difference.abs().mean()),'rmse':float(difference.square().mean().sqrt()),
                'max_abs':float(difference.abs().max()),'exact_value_fraction':float((a==b).float().mean()),
                'correlation':float((ac*bc).sum()/torch.sqrt(ac.square().sum()*bc.square().sum()))}
    def merge_inputs(decoder,skip):
        up=reference.nearest_upsample2_crop(decoder,height=1152,width=1920)
        sin=pipeline.model.weight('block70.layer0.inp_merge_sin')
        cos=pipeline.model.weight('block70.layer0.inp_merge_cos')
        return reference._rows(lambda a,b:a*sin+b*cos,up,skip)
    def post_features(decoder,skip):
        return original_window(merge_inputs(decoder,skip),70,head_count=1)
    gain=pipeline.model.weight('block70.layer0.out_gain');conv=pipeline.model.weight('block70.layer0.out_conv_weight')
    combined=torch.cat((gain,conv),dim=0).contiguous()
    def head_from_features(value,direct=False):
        if direct:return kernel.mma(value.reshape(-1,32),combined).reshape(*value.shape[:-1],4)
        return reference._per_token(lambda t:t[...,:16]@gain+t[...,16:]@conv,value)
    report={'schema':1,'target_achieved':False,'quality_gate_passed':False,'complete':False,
       'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source_commit':SOURCE_COMMIT,'weights_sha256':WEIGHTS_SHA,
       'output_extent':[1920,1080],'network_extent':[1920,1152],'native_runtime_modified':False,
       'uses_captured_native_activations':True,'mma_post_enabled':args.mma_post,'cases':[],
       'limitations':['Captured input substitutions are diagnostic and cannot run independently.',
          'Decoder layouts are explicit hypotheses; feature correlation alone does not prove their semantics.',
          'The output grading approximation and reconstruction remain unproven against all native arithmetic.',
          'First-reset same-scene views only; no temporal or perceptual acceptance and no latency claim.']}
    args.output_directory.mkdir(parents=True)
    def save():
        (args.output_directory/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with torch.inference_mode():
        for label,trial_name,previous_name in args.case:
            decoder_raw,skip_raw,data,images=load_prefixes(Path(trial_name))
            controls=data['controls'];source=images['color'];observed.clear()
            automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
            prepared=pipeline.prepare(source,frame_index=0,normalized_style=controls['DLSSNR.Style']/128,
                local_tone_strength=controls['DLSSNR.LocalToneStrength'],local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            prepared.geometry=NetworkGeometry(1920,1080,1920,1152)
            prepared.features=make_features(source,geometry=prepared.geometry,frame_index=0,
                normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
                local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
            features=torch.from_numpy(prepared.features[None]).cuda().half()
            full_head=pipeline.model(features)
            ungraded=pipeline.finish(prepared,full_head.float().cpu().numpy()[0],intensity=1).image
            record={'label':label,**data,'decoder_layout_metrics':{},'variants':{},'previous_baseline_checked':False}
            if previous_name!='-':
                previous=Path(previous_name);old=json.loads((previous/'comparison.json').read_text())
                if old['capture_hashes']!=data['capture_hashes'] or old['controls']!=controls:
                    raise ValueError('Previous baseline belongs to a different frame.')
                if not np.array_equal(ungraded,np.load(previous/'reconstruction.npy',allow_pickle=False)):
                    raise ValueError('Full baseline does not reproduce its saved output.')
                record['previous_baseline_checked']=True;record['previous_baseline_reproduced_exactly']=True
            native_skip=tensor(decode_skip(skip_raw))
            record['skip_metrics']=feature_metrics(native_skip,observed['skip'])
            native_decoder=None
            for layout in ('two-plane-permuted','two-plane-identity','interleaved','tiled'):
                candidate=tensor(decode_decoder(decoder_raw,layout))
                record['decoder_layout_metrics'][layout]=feature_metrics(candidate,observed['decoder'])
                if layout=='two-plane-permuted':native_decoder=candidate
            del candidate
            if args.prefix_check_trial:
                check=args.prefix_check_trial;frame=json.loads((check/'capture/frame-0.json').read_text())
                color=check/'capture'/frame['resources']['color']['file']
                if sha(color)==data['capture_hashes']['color'] and frame['controls']==controls:
                    meta=json.loads((check/'pre-tensor/metadata.json').read_text())
                    if not (meta['complete'] and meta['gpu_completed'] and meta['frame']==1 and meta['noise_counter']==0):
                        raise ValueError('Old first-block prefix is incomplete.')
                    raw=(check/'pre-tensor'/meta['file']).read_bytes()
                    if bytes(skip_raw[:len(raw)])!=raw:raise ValueError('Skip buffer differs from matching first-block prefix.')
                    record['first_block_prefix_reproduced_exactly']={'bytes':len(raw),'sha256':sha(check/'pre-tensor'/meta['file'])}
            def compose(head):return observed_grade(pipeline.finish(prepared,head.float().cpu().numpy()[0],intensity=1).image,data['output_fields'])
            baseline=compose(full_head)
            record['baseline_vs_native']=image_metrics(baseline,images['output'])
            for variant,decoder,skip in [('reference-inputs',observed['decoder'],observed['skip']),
                    ('native-skip',observed['decoder'],native_skip),('native-decoder',native_decoder,observed['skip']),
                    ('both-native-inputs',native_decoder,native_skip)]:
                post=post_features(decoder,skip);head=head_from_features(post)
                if variant=='reference-inputs':
                    if not torch.equal(head,full_head):raise ValueError('Isolated post-block replay differs from complete model head.')
                    record['isolated_reference_head_matches_full_model']=True
                image=compose(head)
                np.save(args.output_directory/(label+'-'+variant+'.npy'),image)
                record['variants'][variant]={'vs_native':image_metrics(image,images['output']),
                    'vs_full_reconstruction':image_metrics(image,baseline)}
                if variant=='both-native-inputs':
                    direct=head_from_features(post,direct=True);direct_image=compose(direct)
                    np.save(args.output_directory/(label+'-both-native-direct-head.npy'),direct_image)
                    record['variants']['both-native-direct-head']={'vs_native':image_metrics(direct_image,images['output']),
                        'vs_full_reconstruction':image_metrics(direct_image,baseline),'head_vs_separate_matmuls':feature_metrics(direct,head)}
                print(label,variant,record['variants'][variant]['vs_native']['rgb_mae'],flush=True)
                del post,head,image
            if args.mma_post:
                post=mma_post(merge_inputs(observed['decoder'],observed['skip']),
                    attention_mma=True,seed_residual=True,seed_logits=True)
                image=compose(head_from_features(post,direct=True))
                variant='reference-inputs-mma-all'
                np.save(args.output_directory/(label+'-'+variant+'.npy'),image)
                record['variants'][variant]={'vs_native':image_metrics(image,images['output']),
                    'vs_full_reconstruction':image_metrics(image,baseline),
                    'uses_captured_native_activations':False,
                    'mma_options':{'attention_mma':True,'seed_residual':True,'seed_logits':True}}
                print(label,variant,record['variants'][variant]['vs_native']['rgb_mae'],flush=True)
                del post,image
                merged=merge_inputs(native_decoder,native_skip)
                for variant,options in [
                    ('native-inputs-mma-ffn',{'attention_mma':False,'seed_residual':False,'seed_logits':False}),
                    ('native-inputs-mma-seeded-ffn',{'attention_mma':False,'seed_residual':True,'seed_logits':False}),
                    ('native-inputs-mma-all',{'attention_mma':True,'seed_residual':True,'seed_logits':True})]:
                    post=mma_post(merged,**options)
                    image=compose(head_from_features(post,direct=True))
                    np.save(args.output_directory/(label+'-'+variant+'.npy'),image)
                    record['variants'][variant]={'vs_native':image_metrics(image,images['output']),
                        'vs_full_reconstruction':image_metrics(image,baseline),'mma_options':options}
                    print(label,variant,record['variants'][variant]['vs_native']['rgb_mae'],flush=True)
                    del post,image
                del merged
            report['cases'].append(record);save()
            print(label,'layout correlations',{k:v['correlation'] for k,v in record['decoder_layout_metrics'].items()},flush=True)
            del native_skip,native_decoder,full_head,features,prepared,baseline,ungraded
            observed.clear()
    report['complete']=True;report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;save()
    print('Complete; no native replacement or speed claim.',flush=True)


if __name__=='__main__':main()
