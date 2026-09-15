# SPDX-License-Identifier: Apache-2.0
"""Verify gradient routing through all conditioned parameters on native pairs."""
import argparse,json,os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from output_grade import GradedStudent
from paired_gradient import combine_pair,paired_gradients
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,training_pairs,load_pair,rgb_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','photos','images','model','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):raise ValueError('Use a fresh private report.')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    record=read(a.model/'result.json');arch=record['architecture'];pairs=training_pairs(a.base,a.photos,a.images,record)
    assert arch['variant']=='hierarchical-film' and arch['width']==16 and arch['blocks']==2
    state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True);assert state['architecture']==arch
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True),arch['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().to(memory_format=torch.channels_last)
    parameters=list(model.parameters());views=[None]*46
    for i,group in [(0,'scene'),(30,'photos')]:
        assert pairs[i]['group']==group
        source,target=load_pair(pairs[i],record['controls']);views[i]=(source,target,1920)
    losses,pixels=[],[]
    for index in (0,30):
        source,target,_=views[index];loss,pixel=rgb_loss(model(source),target);losses.append(loss);pixels.append(pixel)
    gradients=[torch.cat([g.flatten() for g in torch.autograd.grad(loss,parameters,retain_graph=True)]).detach() for loss in losses]
    combined_loss=torch.cat([g.flatten() for g in torch.autograd.grad(.5*(losses[0]+losses[1]),parameters)]).detach()
    averaged,_=combine_pair(*gradients,project=False)
    mean_max_error=float((averaged-combined_loss).abs().max());assert torch.allclose(averaged,combined_loss,atol=1e-7,rtol=1e-5)
    class FirstInEachDomain:
        def __init__(self):self.bounds=[]
        def integers(self,bound):self.bounds.append(bound);return 0
    rows=[]
    for mode in ('mean','pcgrad'):
        model.zero_grad(set_to_none=True);rng=FirstInEachDomain()
        loss,pixel,conflict=paired_gradients(model,views,rng,mode)
        actual=torch.cat([p.grad.flatten() for p in parameters]);expected,expected_conflict=combine_pair(*gradients,project=mode=='pcgrad')
        assert rng.bounds==[30,16] and torch.equal(conflict,expected_conflict)
        assert torch.equal(actual,expected),(mode,float((actual-expected).abs().max()),float(expected.abs().max()),int((actual!=expected).sum()))
        assert torch.equal(loss,.5*(losses[0].detach()+losses[1].detach()))
        assert torch.equal(pixel,.5*(pixels[0].detach()+pixels[1].detach()))
        condition={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if n.startswith('network.decoder_conditioning.')}
        assert len(condition)==6 and all(v>0 for v in condition.values())
        rows.append({'mode':mode,'gradient_exact':True,'domain_sampling_bounds':rng.bounds,'conflict':bool(conflict),
                     'conditioning_gradient_max':condition,'all_parameters_have_gradients':True})
    model.zero_grad(set_to_none=True)
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    report={'complete':True,'optimizer_steps':0,'weights_unchanged':True,'target_achieved':False,'quality_gate_passed':False,
        'deterministic_algorithms_for_mechanical_test':True,
        'source_checkpoint_sha256':sha(a.model/'student-private.pt'),'capture_hashes':[pairs[i]['capture_hashes'] for i in (0,30)],
        'parameter_count':sum(p.numel() for p in parameters),'mean_vs_joint_loss_gradient_max_abs':mean_max_error,'modes':rows,
        'scope':'One checked scene/photo training pair. Gradient routing and averaging only; no fitting or validation evaluation.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
