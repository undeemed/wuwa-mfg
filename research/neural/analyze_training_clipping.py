# SPDX-License-Identifier: Apache-2.0
"""Measure pre-grading clipping on audited training frames, without fitting."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from collect_demo_photo_training import read,sha,dump
from collect_training_brightness import audited_brightness
from student_training_pairs import training_pairs,load_pair,rgb_loss
from output_grade import GradedStudent,grade_torch
from student_probe import HierarchicalStudent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','photos','images','brightness','baseline','model','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    baseline=read(a.baseline/'result.json');record=read(a.model/'result.json');arch=record['architecture']
    pairs=training_pairs(a.base,a.photos,a.images,baseline)
    for r in audited_brightness(a.brightness,a.base,record['controls'],a.photos,a.images):
        pairs.append({'label':r['label'],'group':'bright-photos','path':a.base/(r['label']+'-state')/'capture','capture_hashes':r['capture_hashes']})
    assert len(pairs)==62 and [r['capture_hashes'] for r in pairs]==record['training_capture_hashes']
    val={record['validation_capture_hashes']['color']}|{r['capture_hashes']['color'] for r in record['extra_validation']}
    assert not val & {r['capture_hashes']['color'] for r in pairs}
    assert arch['variant']=='hierarchical-film' and arch['width']==16 and arch['blocks']==2
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True),arch['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().eval().to(memory_format=torch.channels_last)
    cached={};handle=model.network.head.register_forward_hook(lambda _,args,out:cached.update(head=out))
    rows=[]
    try:
        for pair in pairs:
            source,target=load_pair(pair,record['controls'])
            with torch.no_grad():
                predicted=model(source)
                raw=source+.25*F.pixel_shuffle(cached.pop('head'),4)[:,:,:1080,:1920]
                assert torch.equal(predicted,grade_torch(raw.clamp(0,1),model.grade_parameters))
                clipped=(raw<0)|(raw>1);error=(predicted-target).abs()
                bound=raw.clamp(0,1).detach().requires_grad_(True)
            loss,pixel=rgb_loss(grade_torch(bound,model.grade_parameters),target)
            desired=torch.autograd.grad(loss,bound)[0]
            # A gradient step points back into range if its sign pushes raw
            # values above one down, or values below zero up.
            inward=((raw>1)&(desired>0))|((raw<0)&(desired<0))
            n=int(clipped.sum());inward_n=int(inward.sum())
            rows.append({'label':pair['label'],'group':pair['group'],'capture_hashes':pair['capture_hashes'],
                'mae':float(error.mean()),'clipped_channel_fraction':float(clipped.float().mean()),
                'error_fraction_at_clipped_channels':float(error[clipped].sum()/error.sum().clamp(min=1e-30)),
                'clipped_channel_mae':float(error[clipped].mean()) if n else 0.,
                'inward_gradient_clipped_fraction':inward_n/n if n else 0.,
                'blocked_inward_gradient_l1':float(desired[inward].abs().sum()),
                'preclip_min':float(raw.min()),'preclip_max':float(raw.max())})
    finally:handle.remove()
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    import statistics
    groups={g:{k:statistics.mean(r[k] for r in rows if r['group']==g) for k in ['mae','clipped_channel_fraction','error_fraction_at_clipped_channels','inward_gradient_clipped_fraction','blocked_inward_gradient_l1']} for g in sorted({r['group'] for r in rows})}
    report={'complete':True,'training_only':True,'optimizer_steps':0,'weights_unchanged':True,'parameter_grad_fields_unchanged':True,
        'target_achieved':False,'quality_gate_passed':False,'checkpoint_sha256':sha(a.model/'student-private.pt'),'rows':rows,'groups':groups,
        'limitations':['Gradient is measured at the bounded pre-grade image; it is not proof that changing training gradients will improve validation.',
            'Only 62 training first-reset frames, FP32 diagnostic arithmetic; no fitting or held-out evaluation.']}
    dump(a.output,report);print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
