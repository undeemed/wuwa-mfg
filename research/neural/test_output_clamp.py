# SPDX-License-Identifier: Apache-2.0
"""Check the exact forward and explicit surrogate-gradient contract."""
import argparse,json,os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from output_clamp import StraightThroughClamp
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha,load_pair,rgb_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','capture','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    tests=[]
    for dtype in (torch.float32,torch.float64,torch.float16):
        x=torch.tensor([-3.,-.1,0.,.2,1.,1.1,7.],device='cuda',dtype=dtype,requires_grad=True)
        weights=torch.tensor([1.,2.,3.,4.,5.,6.,7.],device='cuda',dtype=dtype)
        y=StraightThroughClamp.apply(x);assert torch.equal(y,x.clamp(0,1))
        g=torch.autograd.grad((y*weights).sum(),x)[0];assert torch.equal(g,weights)
        exact=torch.autograd.grad((x.clamp(0,1)*weights).sum(),x)[0]
        assert torch.equal(exact,weights*((x>=0)&(x<=1)))
        tests.append({'dtype':str(dtype),'forward_exact':True,'straight_through_gradient_exact':True,'ordinary_gradient_exact':True})
    bits=np.arange(65536,dtype=np.uint16).view(np.float16);v=torch.from_numpy(bits[np.isfinite(bits)].copy()).cuda()
    assert torch.equal(StraightThroughClamp.apply(v).view(torch.int16),v.clamp(0,1).view(torch.int16))
    record=read(a.model/'result.json');state=torch.load(a.model/'student-private.pt',map_location='cpu',weights_only=True)
    frame=read(a.capture/'frame-0.json');hashes={n:sha(a.capture/frame['resources'][n]['file']) for n in ('color','output')}
    assert hashes in record['training_capture_hashes']
    source,target=load_pair({'path':a.capture,'capture_hashes':hashes},record['controls'])
    model=GradedStudent(HierarchicalStudent(16,2,conditioned=True),record['architecture']['explicit_output_grading'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.cuda().float().to(memory_format=torch.channels_last)
    rows=[]
    for training in (True,False):
        model.train(training);outputs=[];gradients=[]
        for enabled in (False,True):
            model.network.straight_through_output_clamp=enabled
            output=model(source);loss,_=rgb_loss(output,target)
            gradient=torch.autograd.grad(loss,list(model.parameters()))
            outputs.append(output.detach());gradients.append(torch.cat([g.flatten() for g in gradient]).detach())
        assert torch.equal(outputs[0],outputs[1])
        changed=not torch.equal(gradients[0],gradients[1]);assert changed==training
        assert all(torch.isfinite(g).all() for g in gradients)
        rows.append({'training_mode':training,'output_exact':True,'gradient_changed':changed,'gradient_difference_l2':float(torch.linalg.vector_norm(gradients[1]-gradients[0]))})
    assert all(torch.equal(v.cpu(),state['state_dict'][k]) for k,v in model.state_dict().items())
    dump={'complete':True,'optimizer_steps':0,'weights_unchanged':True,'target_achieved':False,'quality_gate_passed':False,
        'checkpoint_sha256':sha(a.model/'student-private.pt'),'training_capture_hashes':hashes,'operator_tests':tests,
        'all_finite_half_forward_bit_exact':True,'model_cases':rows,'deterministic_mechanical_test_only':True,
        'scope':'The backward identity is intentionally a surrogate, not the derivative of a clamp. No finite-difference gradient correctness claim.'}
    a.output.write_text(json.dumps(dump,indent=2,allow_nan=False)+'\n');print(json.dumps(dump,indent=2))


if __name__=='__main__':main()
