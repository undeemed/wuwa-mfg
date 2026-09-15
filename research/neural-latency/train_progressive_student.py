# SPDX-License-Identifier: Apache-2.0
"""Train only a second residual stage on the audited 62-frame collection."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from collect_training_brightness import audited_brightness
from progressive_student import FrozenStudentRefinement,load_first
from student_training_pairs import read,sha,training_pairs,load_pair,rgb_loss
from paired_gradient import paired_gradients


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','photos','images','brightness','baseline','first','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(a.first)
    pairs=training_pairs(a.base,a.photos,a.images,read(a.baseline/'result.json'))
    for r in audited_brightness(a.brightness,a.base,record['controls'],a.photos,a.images):
        pairs.append({'label':r['label'],'group':'bright-photos','path':a.base/(r['label']+'-state')/'capture','capture_hashes':r['capture_hashes']})
    assert len(pairs)==62 and [r['capture_hashes'] for r in pairs]==record['training_capture_hashes']
    validation={record['validation_capture_hashes']['color']}|{r['capture_hashes']['color'] for r in record['extra_validation']}
    assert len(validation)==16 and not validation & {r['capture_hashes']['color'] for r in pairs}
    torch.manual_seed(28411);rng=np.random.default_rng(28411)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=FrozenStudentRefinement(first).cuda().float().to(memory_format=torch.channels_last)
    first_state={k:v.detach().cpu().clone() for k,v in model.first.state_dict().items()}
    views=[];initial=[]
    with torch.no_grad():
        for pair in pairs:
            source,target=load_pair(pair,record['controls']);base=model.first_ungraded(source)
            x=torch.cat((base,source),1).contiguous(memory_format=torch.channels_last)
            prediction=model.refinement(x)
            assert torch.equal(prediction,model.first(source))
            views.append((x,target,1920));initial.append(float((prediction-target).abs().mean()))
    a.output.mkdir(parents=True)
    optimizer=torch.optim.AdamW(model.refinement.parameters(),lr=.002,weight_decay=.0001)
    history=[];torch.cuda.synchronize();started=time.perf_counter()
    model.train()
    for step in range(4500):
        optimizer.zero_grad(set_to_none=True)
        lr=.00002+.5*(.002-.00002)*(1+math.cos(math.pi*step/4499))
        for group in optimizer.param_groups:group['lr']=lr
        loss,pixel,_=paired_gradients(model.refinement,views,rng,'mean')
        optimizer.step()
        if step%100==0 or step==4499:
            row={'step':step+1,'loss':float(loss),'pixel_mae':float(pixel),'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True)
        if time.perf_counter()-started>600:break
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    assert step==4499,'Bounded run did not finish; do not treat an incomplete fit as a matched comparison.'
    assert all(torch.equal(v.cpu(),first_state[k]) for k,v in model.first.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.first.parameters())
    model.eval();trained=[]
    with torch.no_grad():
        for pair,(source,target,_),before in zip(pairs,views,initial):
            prediction=model.refinement(source)
            trained.append({k:v for k,v in pair.items() if k!='path'}|{'initial_base_mae':before,'refined_mae':float((prediction-target).abs().mean())})
    torch.save({'state_dict':model.cpu().state_dict(),'grade_parameters':model.first.grade_parameters},a.output/'student-private.pt')
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'completed_steps':4500,'training_seconds':elapsed,
        'native_shape':[1080,1920],'controls':record['controls'],'first_checkpoint_sha256':sha(a.first/'student-private.pt'),
        'first_result_sha256':sha(a.first/'result.json'),'first_stage_unchanged':True,'first_stage_has_gradients':False,
        'parameters':{'frozen':sum(p.numel() for p in model.first.parameters()),'trainable':sum(p.numel() for p in model.refinement.parameters())},
        'training_capture_hashes':record['training_capture_hashes'],'validation_capture_hashes':record['validation_capture_hashes'],
        'extra_validation_hashes':[r['capture_hashes'] for r in record['extra_validation']],
        'training':history,'training_image_metrics':trained,'optimization':{'updates':4500,'examples':9000,'seed':28411,'initial_lr':.002,'final_lr':.00002,
            'optimizer':'AdamW','weight_decay':.0001,'domain_weights':[.5,.5],'loss':'L1 + 0.25 * horizontal and vertical gradient L1'},
        'scope':'Frozen first student plus a newly trained full-resolution conditioned residual stage. FP32 first-stage outputs are cached only for fitting. Both stages run during inference; no teacher pixels are inputs.'}
    (a.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed_steps':4500,'training_seconds':elapsed,'first_unchanged':True,'mean_training_mae':sum(r['refined_mae'] for r in trained)/62}),flush=True)


if __name__=='__main__':main()
