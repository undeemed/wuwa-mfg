# SPDX-License-Identifier: Apache-2.0
"""Fit a correction decoder on frozen student features from the audited TRAIN set."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from collect_training_brightness import audited_brightness
from paired_gradient import combine_pair
from progressive_student import load_first
from shared_feature_student import SharedFeatureRefinement
from student_training_pairs import read,sha,training_pairs,load_pair,rgb_loss
from cluster_sampling import draw_pair,validate_coverage


def fit_pair(decoder,views,rng,probabilities=None,sampled_counts=None):
    parameters=list(decoder.parameters());vectors=[];losses=[];pixels=[]
    for index in draw_pair(rng,probabilities):
        if sampled_counts is not None:sampled_counts[index]+=1
        source,base,features,target=views[index]
        loss,pixel=rgb_loss(decoder(source,base,features),target)
        gradients=torch.autograd.grad(loss,parameters)
        vectors.append(torch.cat([g.flatten() for g in gradients]).detach())
        losses.append(loss.detach());pixels.append(pixel.detach())
    combined,_=combine_pair(*vectors,project=False);offset=0
    for parameter in parameters:
        parameter.grad=combined[offset:offset+parameter.numel()].view_as(parameter);offset+=parameter.numel()
    assert offset==combined.numel()
    return .5*(losses[0]+losses[1]),.5*(pixels[0]+pixels[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','photos','images','brightness','baseline','first','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--coverage-sampling',type=Path,help='Optional preselected TRAIN-only cluster mixture report.')
    parser.add_argument('--region-mode',choices=('dense','routed'),help='Optional matched region-context experiment.')
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(args.first)
    pairs=training_pairs(args.base,args.photos,args.images,read(args.baseline/'result.json'))
    for row in audited_brightness(args.brightness,args.base,record['controls'],args.photos,args.images):
        pairs.append({'label':row['label'],'group':'bright-photos','path':args.base/(row['label']+'-state')/'capture','capture_hashes':row['capture_hashes']})
    assert len(pairs)==62 and [row['capture_hashes'] for row in pairs]==record['training_capture_hashes']
    validation={record['validation_capture_hashes']['color']}|{row['capture_hashes']['color'] for row in record['extra_validation']}
    assert len(validation)==16 and not validation&{row['capture_hashes']['color'] for row in pairs}
    coverage=read(args.coverage_sampling) if args.coverage_sampling else None
    probabilities=validate_coverage(coverage,record['training_capture_hashes'],sha(args.first/'student-private.pt')) if coverage else None
    sampled_counts=np.zeros(62,dtype=np.int64)
    torch.manual_seed(28411);rng=np.random.default_rng(28411)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if args.region_mode:
        from region_context_student import RegionFeatureRefinement
        model=RegionFeatureRefinement(first,args.region_mode)
    else:model=SharedFeatureRefinement(first)
    model=model.cuda().float().to(memory_format=torch.channels_last)
    frozen={key:value.detach().cpu().clone() for key,value in model.first.state_dict().items()}
    views=[];initial=[]
    with torch.no_grad():
        for pair in pairs:
            source,target=load_pair(pair,record['controls']);base,features=model.extract(source)
            prediction=model.refinement(source,base,features)
            assert torch.equal(prediction,model.first(source))
            assert all(not tensor.requires_grad for tensor in (base,*features))
            views.append((source,base,features,target));initial.append(float((prediction-target).abs().mean()))
    args.output.mkdir(parents=True)
    optimizer=torch.optim.AdamW(model.refinement.parameters(),lr=.002,weight_decay=.0001)
    history=[];torch.cuda.synchronize();started=time.perf_counter();model.train()
    for step in range(4500):
        optimizer.zero_grad(set_to_none=True)
        lr=.00002+.5*(.002-.00002)*(1+math.cos(math.pi*step/4499))
        for group in optimizer.param_groups:group['lr']=lr
        loss,pixel=fit_pair(model.refinement,views,rng,probabilities,sampled_counts);optimizer.step()
        if step%100==0 or step==4499:
            row={'step':step+1,'loss':float(loss),'pixel_mae':float(pixel),'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
        if time.perf_counter()-started>600:break
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    assert step==4499,'Bounded training did not complete; not a finished candidate.'
    assert all(torch.equal(value.cpu(),frozen[key]) for key,value in model.first.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.first.parameters())
    assert sampled_counts[:30].sum()==sampled_counts[30:].sum()==4500
    model.eval();metrics=[]
    with torch.no_grad():
        for pair,(source,base,features,target),before in zip(pairs,views,initial):
            output=model.refinement(source,base,features)
            metrics.append({key:value for key,value in pair.items() if key!='path'}|{'initial_base_mae':before,'refined_mae':float((output-target).abs().mean())})
    torch.save({'state_dict':model.cpu().state_dict(),'grade_parameters':model.first.grade_parameters},args.output/'student-private.pt')
    report={'complete':True,'variant':'shared-features','target_achieved':False,'quality_gate_passed':False,'completed_steps':4500,
            'training_seconds':elapsed,'native_shape':[1080,1920],'controls':record['controls'],
            'first_checkpoint_sha256':sha(args.first/'student-private.pt'),'first_result_sha256':sha(args.first/'result.json'),
            'first_stage_unchanged':True,'first_stage_has_gradients':False,
            'parameters':{'frozen':sum(p.numel() for p in model.first.parameters()),'trainable':sum(p.numel() for p in model.refinement.parameters())},
            'training_capture_hashes':record['training_capture_hashes'],'validation_capture_hashes':record['validation_capture_hashes'],
            'extra_validation_hashes':[row['capture_hashes'] for row in record['extra_validation']],
            'training':history,'training_image_metrics':metrics,
            'optimization':{'updates':4500,'examples':9000,'seed':28411,'initial_lr':.002,'final_lr':.00002,'optimizer':'AdamW','weight_decay':.0001,
                            'domain_weights':[.5,.5],'loss':'L1 + 0.25 * horizontal and vertical gradient L1'},
            'scope':'Frozen base plus correction decoder with seven current-image student features and original input. FP32 features cached only for training; complete first stage runs for every inference.'}
    if coverage:
        report['optimization']['sampling']={'coverage_sha256':sha(args.coverage_sampling),'rule':coverage['sampling_rule'],
            'per_image_counts':sampled_counts.tolist(),'scene_probabilities':probabilities[0].tolist(),'photo_probabilities':probabilities[1].tolist(),
            'validation_or_target_used_to_select_probabilities':False}
    if args.region_mode:
        from region_context_student import architecture
        report['variant']='region-context'
        report['region_context']=architecture(args.region_mode)
        report['sampled_counts']=sampled_counts.tolist()
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed_steps':4500,'training_seconds':elapsed,'mean_training_mae':sum(r['refined_mae'] for r in metrics)/62}),flush=True)


if __name__=='__main__':main()
