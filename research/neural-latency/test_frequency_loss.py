# SPDX-License-Identifier: Apache-2.0
"""Check spectral weighting, its detached-weight gradient and paired routing."""
import argparse,json,os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from frequency_loss import frequency_loss
from paired_gradient import paired_gradients
from student_training_pairs import read,sha,load_pair,rgb_loss
from student_probe import HierarchicalStudent
from output_grade import GradedStudent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','scene','photo','calibration','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    torch.manual_seed(19341);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    tests=[]
    for dtype,tolerance in ((torch.float32,1e-6),(torch.float64,1e-12)):
        for shape in ((1,3,8,12),(2,2,7,11)):
            x=torch.randn(shape,device='cuda',dtype=dtype,requires_grad=True);y=torch.randn_like(x)
            loss=frequency_loss(x,y);gradient=torch.autograd.grad(loss,x)[0]
            difference=torch.fft.fft2(x.detach()-y,norm='ortho');weight=difference.abs()
            weight=weight/weight.amax(dim=(-2,-1),keepdim=True).clamp(min=1e-12)
            expected=2*torch.fft.ifft2(weight*difference,norm='ortho').real/x.numel()
            assert torch.allclose(gradient,expected,rtol=tolerance,atol=tolerance)
            assert 0<=float(loss)<=float((x-y).square().mean())+tolerance
            zero=frequency_loss(x,x.detach());zg=torch.autograd.grad(zero,x)[0]
            assert float(zero)==0 and torch.count_nonzero(zg)==0
            tests.append({'dtype':str(dtype),'shape':list(shape),'gradient_max_abs':float((gradient-expected).abs().max()),'zero_error_exact':True})
    impulse=torch.zeros(1,1,8,8,device='cuda',dtype=torch.float64);impulse[0,0,0,0]=1
    assert torch.allclose(frequency_loss(impulse,torch.zeros_like(impulse)),impulse.square().mean(),atol=1e-14)
    shifted=impulse.roll(1,-1)
    assert torch.allclose(torch.fft.fft2(impulse).abs(),torch.fft.fft2(shifted).abs(),rtol=1e-12,atol=1e-12)
    assert float(frequency_loss(impulse,shifted))>0
    record=read(a.model/'result.json');state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    coefficient=read(a.calibration)['frequency_coefficient'];assert 0<coefficient<=100
    selected=[]
    for path in (a.scene,a.photo):
        frame=read(path/'frame-0.json');hashes={n:sha(path/frame['resources'][n]['file']) for n in ('color','output')}
        assert hashes in record['training_capture_hashes'];source,target=load_pair({'path':path,'capture_hashes':hashes},record['controls'])
        selected.append((source,target,1920))
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True),record['architecture']['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().train().to(memory_format=torch.channels_last)
    views=[None]*62;views[0]=selected[0];views[61]=selected[1]
    class FixedSelection:
        def __init__(self):self.calls=0
        def integers(self,n):
            self.calls+=1;assert n==(30 if self.calls==1 else 32)
            return 0 if self.calls==1 else 31
    routing=[]
    for weight in (0.,coefficient):
        grads=[];losses=[]
        for source,target,_ in selected:
            prediction=model(source);loss,_=rgb_loss(prediction,target)
            if weight:loss=loss+weight*frequency_loss(prediction,target)
            grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,list(model.parameters()))]));losses.append(loss.detach())
        expected=.5*(grads[0]+grads[1])
        loss,_,_=paired_gradients(model,views,FixedSelection(),'mean',weight)
        actual=torch.cat([p.grad.flatten() for p in model.parameters()])
        assert torch.equal(expected,actual) and torch.equal(loss,.5*(losses[0]+losses[1]))
        routing.append({'weight':weight,'gradient_exact':True,'loss_exact':True,'last_photo_slot_checked':True})
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    report={'complete':True,'optimizer_steps':0,'weights_unchanged':True,'target_achieved':False,'quality_gate_passed':False,
        'checkpoint_sha256':sha(a.model/'student-private.pt'),'calibration_sha256':sha(a.calibration),'operator_tests':tests,
        'impulse_parseval_passed':True,'phase_sensitivity_passed':True,'paired_routing':routing,'deterministic_mechanical_test_only':True,
        'scope':'Gradient is checked with the adaptive spectrum weights held fixed, as defined by the loss; no finite-difference claim for differentiating those weights.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
