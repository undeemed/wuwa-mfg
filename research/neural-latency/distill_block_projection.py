# SPDX-License-Identifier: Apache-2.0
"""Fit an affine feature projection to one pretrained split-window block.

Local feature distillation only; the teacher is the imperfect reconstruction.
Private weights remain outside the repository. Final-image comparisons against
native captures and unchanged reconstruction are both required and reported.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from compare_output_grade import read_capture,read_contract,observed_grade
from probe_block_sensitivity import setup_model,sha,image_metrics,SOURCE_COMMIT,WEIGHTS_SHA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--contract-trial',type=Path,required=True)
    parser.add_argument('--train-capture',type=Path,action='append',required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,metavar=('LABEL','CAPTURE','SAVED_BASELINE'))
    parser.add_argument('--block',type=int,choices=range(40,48),default=43)
    parser.add_argument('--ridge',type=float,default=.001)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2]
    if args.output_directory.exists():raise FileExistsError(args.output_directory)
    if args.output_directory.resolve().is_relative_to(repository):raise ValueError('Private weights must stay outside the repository.')
    if not 0<args.ridge<=1:raise ValueError('Ridge fraction must lie in (0,1].')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip()!=SOURCE_COMMIT:
        raise ValueError('Wrong reference source commit.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=args.source,text=True).strip():
        raise ValueError('Reference tracked source has local modifications.')
    if sha(args.weights)!=WEIGHTS_SHA:raise ValueError('Wrong private weights.')
    contract=read_contract(args.contract_trial)
    controls,_,contract_hashes=read_capture(args.contract_trial/'capture')
    training=[];train_hashes=set()
    for path in args.train_capture:
        c,images,hashes=read_capture(path)
        if c!=controls or hashes['color'] in train_hashes:raise ValueError('Training controls or uniqueness check failed.')
        train_hashes.add(hashes['color']);training.append((images,hashes))
    validation=[]
    for label,path,previous_path in args.case:
        c,images,hashes=read_capture(Path(path));previous_dir=Path(previous_path)
        previous=json.loads((previous_dir/'comparison.json').read_text())
        if c!=controls or previous['controls']!=c or hashes!=previous['capture_hashes'] or hashes['color'] in train_hashes:
            raise ValueError('Validation must match its baseline and remain outside projection training.')
        if previous.get('native_pre_pool_diagnostic') or previous.get('native_pre_stem_diagnostic') or previous.get('cublas_fp16_accumulation',False):
            raise ValueError('Unsupported baseline configuration.')
        for key,value in {'precision':'fast','noise_frame':0,'network_extent':[1152,1920],
             'batched_ffn':True,'fused_gate':True,'fused_publish':True,'fused_roundtrip':True,
             'first_block_rounding':False,'mma_first_block':False,'weights_sha256':WEIGHTS_SHA}.items():
            if previous.get(key)!=value:raise ValueError('Baseline configuration differs.')
        validation.append((label,images,hashes,previous_dir))
    if len({row[0] for row in validation})!=len(validation):raise ValueError('Duplicate case labels.')
    import torch
    from fused_norm import FusedNorm
    from test_output_grade import graph_measure
    pipeline=setup_model(args.source,args.weights)
    from mlxdlss.features import AutomaticMask,NetworkGeometry,make_features
    backend=FusedNorm();mode='teacher';observed=[];weight=None;bias=None
    original=pipeline.model._split_window
    def projection(value):return backend.roundtrip(value@weight+bias)
    def hook(value,index):
        if index!=args.block:return original(value,index)
        if mode=='identity':return value
        if mode=='projection':return projection(value)
        result=original(value,index)
        if mode=='capture':observed.append((value.clone(),result.clone()))
        return result
    pipeline.model._split_window=hook
    def prepare(source):
        automatic=AutomaticMask(controls['DLSSNR.SkinStructureStrength'],controls['DLSSNR.LocalStructureStrength']) if controls['DLSSNR.UseAutoMask'] else None
        prepared=pipeline.prepare(source,frame_index=0,normalized_style=controls['DLSSNR.Style']/128,
          local_tone_strength=controls['DLSSNR.LocalToneStrength'],local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
        prepared.geometry=NetworkGeometry(1920,1080,1920,1152)
        prepared.features=make_features(source,geometry=prepared.geometry,frame_index=0,
          normalized_style=controls['DLSSNR.Style']/128,local_tone_strength=controls['DLSSNR.LocalToneStrength'],
          local_structure_strength=controls['DLSSNR.LocalStructureStrength'],automatic_mask=automatic)
        return prepared,torch.from_numpy(prepared.features[None]).cuda().half()
    def finish(prepared,tensor):
        head=pipeline.model(tensor).float().cpu().numpy()[0]
        return pipeline.finish(prepared,head,intensity=1).image
    report={'schema':1,'target_achieved':False,'quality_gate_passed':False,'complete':False,
      'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source_commit':SOURCE_COMMIT,
      'weights_sha256':WEIGHTS_SHA,'block':args.block,'ridge_fraction':args.ridge,
      'native_runtime_modified':False,'uses_captured_native_activations':False,
      'output_extent':[1920,1080],'network_extent':[1920,1152],'noise_frame':0,
      'contract_capture_hashes':contract_hashes,'controls':controls,'training':[],'cases':[],
      'teacher':'Pinned pretrained reconstruction; not exact native intermediate features.',
      'fit':'Float64 centered ridge regression with an unpenalized intercept; ridge is fraction of mean covariance diagonal.',
      'inference':'FP16 matmul, FP16 bias add, E4M3 publication back to FP16.',
      'limitations':['Training and validation are first-reset camera views from one scene, with overlapping scene content.',
        'Validation cameras guided the prior block-selection diagnostic and are not an independent final quality test.',
        'The unmodified reconstruction already differs from native output.',
        'Isolated block CUDA Graph timings exclude the surrounding model and application; not native speedup.',
        'No weights, raw activations or captured images are published.']}
    args.output_directory.mkdir(parents=True)
    output=args.output_directory/'result.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    start=time.perf_counter()
    with torch.inference_mode():
        pairs=[]
        for images,hashes in training:
            prepared,tensor=prepare(images['color']);mode='capture';observed.clear()
            finish(prepared,tensor)
            if len(observed)!=1:raise ValueError('Teacher block was not captured exactly once.')
            x,y=observed.pop()
            if x.shape!=y.shape or x.shape[-1]!=512 or not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise ValueError('Unexpected split-window feature contract.')
            pairs.append((x,y));report['training'].append({'capture_hashes':hashes,'feature_shape':list(x.shape)})
            print('Captured projection training view',len(pairs),list(x.shape),flush=True)
            del prepared,tensor
        design=torch.cat([x.reshape(-1,512) for x,y in pairs]).double()
        target=torch.cat([y.reshape(-1,512) for x,y in pairs]).double()
        mx=design.mean(0);my=target.mean(0);design=design-mx;target=target-my
        covariance=design.T@design
        regularizer=args.ridge*covariance.diagonal().mean()
        covariance.diagonal().add_(regularizer)
        fitted=torch.linalg.solve(covariance,design.T@target)
        fitted_bias=my-mx@fitted
        weight=fitted.half().contiguous();bias=fitted_bias.half().contiguous()
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):raise ValueError('Nonfinite fitted projection.')
        report['sample_tokens']=design.shape[0];report['parameters']=weight.numel()+bias.numel()
        report['ridge_absolute']=float(regularizer)
        def feature_metrics(actual,expected):
            error=(actual.float()-expected.float()).abs()
            return {'mae':float(error.mean()),'rmse':float(error.square().mean().sqrt()),'max_abs':float(error.max())}
        for entry,(x,y) in zip(report['training'],pairs):
            entry['projection_vs_teacher']=feature_metrics(projection(x),y)
            entry['identity_vs_teacher']=feature_metrics(x,y)
        checkpoint=args.output_directory/'projection-private.pt'
        torch.save({'block':args.block,'weight':weight.cpu(),'bias':bias.cpu(),
                    'training_capture_hashes':[x['capture_hashes'] for x in report['training']]},checkpoint)
        report['private_projection_sha256']=sha(checkpoint)
        del design,target,covariance,fitted,fitted_bias,pairs
        for label,images,hashes,previous_dir in validation:
            prepared,tensor=prepare(images['color']);mode='capture';observed.clear()
            raw=finish(prepared,tensor);saved=np.load(previous_dir/'reconstruction.npy',allow_pickle=False)
            if not np.array_equal(raw,saved) or len(observed)!=1:raise ValueError('Baseline does not reproduce saved output exactly.')
            x,y=observed.pop();baseline=observed_grade(raw,contract)
            case={'label':label,'capture_hashes':hashes,'saved_baseline_reproduced_exactly':True,
              'saved_baseline_sha256':sha(previous_dir/'reconstruction.npy'),
              'baseline_vs_native':image_metrics(baseline,images['output']),
              'projection_feature_error':feature_metrics(projection(x),y),
              'identity_feature_error':feature_metrics(x,y),'variants':{}}
            for mode in ('identity','projection'):
                candidate=observed_grade(finish(prepared,tensor),contract)
                case['variants'][mode]={'vs_native':image_metrics(candidate,images['output']),
                                       'vs_unpruned':image_metrics(candidate,baseline)}
            if not report['cases']:
                teacher_time,teacher_output=graph_measure(lambda value:original(value,args.block),x)
                projection_time,projection_output=graph_measure(projection,x)
                if not torch.equal(teacher_output,y) or not torch.equal(projection_output,projection(x)):
                    raise ValueError('Captured block graph differs from eager output.')
                report['isolated_block_graphs']={'teacher':teacher_time,'projection':projection_time,
                  'each_graph_matches_its_eager_output':True,'feature_shape':list(x.shape)}
            report['cases'].append(case);save()
            print(label,{k:v['vs_native']['rgb_mae'] for k,v in case['variants'].items()},flush=True)
            del prepared,tensor,raw,saved,baseline,x,y
        mode='teacher'
    report['complete']=True;report['wall_seconds']=time.perf_counter()-start
    report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;save()
    print('Complete; no candidate accepted or installed.',flush=True)


if __name__=='__main__':main()
