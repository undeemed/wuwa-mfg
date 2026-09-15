# SPDX-License-Identifier: Apache-2.0
"""Measure training-gradient compatibility without updating a checkpoint."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,training_pairs,load_pair,rgb_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['base','photos','images','model','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private report file.')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    record=read(args.model/'result.json');architecture=record['architecture']
    assert architecture['variant']=='hierarchical' and architecture['width']==16 and architecture['blocks']==2
    assert architecture['noise_channels']==0 and record['completed_steps']==4500
    pairs=training_pairs(args.base,args.photos,args.images,record)
    checkpoint=torch.load(args.model/'student-private.pt',map_location='cpu',weights_only=True)
    assert checkpoint['architecture']==architecture
    model=GradedStudent(HierarchicalStudent(16,2),architecture['explicit_output_grading'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model=model.cuda().float().eval().to(memory_format=torch.channels_last)
    parameters=list(model.parameters())
    gradients,rows=[],[]
    for pair in pairs:
        source,target=load_pair(pair,record['controls'])
        loss,pixel=rgb_loss(model(source),target)
        grad=torch.autograd.grad(loss,parameters)
        vector=torch.cat([g.flatten() for g in grad]).detach()
        assert torch.isfinite(vector).all()
        gradients.append(vector)
        rows.append({k:v for k,v in pair.items() if k!='path'}|{'loss':float(loss),'pixel_mae':float(pixel),'gradient_l2':float(vector.double().norm())})
        del source,target,loss,pixel,grad
    matrix=torch.stack(gradients).double()
    lengths=matrix.norm(dim=1)
    assert (lengths>0).all()
    cosines=(matrix@matrix.T/(lengths[:,None]*lengths[None,:])).clamp(-1,1).cpu().numpy()
    categories={'cross_domain':[],'within_scene':[],'within_photos':[]}
    for i,a in enumerate(rows):
        for j,b in enumerate(rows[:i]):
            name='cross_domain' if a['group']!=b['group'] else 'within_'+a['group']
            categories[name].append(float(cosines[i,j]))
    summaries={name:{'pairs':len(values),'negative_pairs':sum(v<0 for v in values),
        'mean_cosine':float(np.mean(values)),'median_cosine':float(np.median(values)),
        'min_cosine':min(values),'max_cosine':max(values)} for name,values in categories.items()}
    means={g:matrix[[i for i,r in enumerate(rows) if r['group']==g]].mean(dim=0) for g in ['scene','photos']}
    mean_cosine=float(torch.dot(means['scene'],means['photos'])/(means['scene'].norm()*means['photos'].norm()))
    assert all(p.grad is None for p in parameters)
    assert all(torch.equal(v.cpu(),checkpoint['state_dict'][k]) for k,v in model.state_dict().items())
    report={'schema':1,'complete':True,'target_achieved':False,'quality_gate_passed':False,
        'model_result_sha256':sha(args.model/'result.json'),'checkpoint_sha256':sha(args.model/'student-private.pt'),
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'training_only':True,
        'weights_unchanged':True,'parameter_grad_fields_unchanged':True,'optimizer_steps':0,
        'rows':rows,'pair_summaries':summaries,'mean_domain_gradient_cosine':mean_cosine,
        'mean_domain_gradient_l2':{g:float(v.norm()) for g,v in means.items()},
        'limitations':['Diagnostic FP32 gradients at one existing checkpoint, not a causal proof of harmful interference.',
            'Scene/photo groups are content domains for the same renderer loss, not the separate tasks in the PCGrad paper.',
            'No validation images enter this diagnostic. Negative gradient alignment alone does not prove projection will improve quality.']}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['pair_summaries','mean_domain_gradient_cosine','mean_domain_gradient_l2']},indent=2))


if __name__=='__main__':
    main()
