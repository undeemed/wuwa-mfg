# SPDX-License-Identifier: Apache-2.0
"""Check two-task projection algebra and the ordinary mean-gradient control."""
import argparse
import json
from pathlib import Path

import torch

from paired_gradient import combine_pair


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private report.')
    rows=[]
    for dtype in [torch.float32,torch.float64]:
        for label,aa,bb in [('aligned',[1.,2.],[2.,1.]),('conflicting',[1.,0.],[-1.,1.]),
                            ('opposed',[2.,0.],[-3.,0.]),('zero',[0.,0.],[2.,3.])]:
            a=torch.tensor(aa,dtype=dtype);b=torch.tensor(bb,dtype=dtype)
            saved=(a.clone(),b.clone())
            mean,conflict=combine_pair(a,b,project=False)
            projected,other=combine_pair(a,b,project=True)
            expected=.5*(a+b)
            dot=torch.dot(a,b)
            if dot<0:
                expected=.5*((a-dot/torch.dot(b,b)*b)+(b-dot/torch.dot(a,a)*a))
            assert torch.equal(mean,.5*(a+b)) and torch.allclose(projected,expected,atol=1e-7,rtol=1e-7)
            assert torch.dot(projected,a)>=-1e-6 and torch.dot(projected,b)>=-1e-6
            assert torch.equal(a,saved[0]) and torch.equal(b,saved[1]) and conflict==other
            rows.append({'case':label,'dtype':str(dtype),'conflict':bool(conflict),'projected':projected.tolist(),'mean':mean.tolist()})
    weight=torch.tensor([.4,-.2],dtype=torch.float64,requires_grad=True)
    losses=[((weight*torch.tensor(x)-y)**2).sum() for x,y in [([1.,2.],.3),([-2.,1.],.5)]]
    a,b=[torch.autograd.grad(loss,weight,retain_graph=True)[0] for loss in losses]
    expected=torch.autograd.grad(.5*(losses[0]+losses[1]),weight)[0]
    mean,_=combine_pair(a,b,project=False)
    assert torch.equal(mean,expected) and weight.grad is None
    bad=[('empty',a[:0],b[:0]),('shape',a,b[:1]),('rank',a[None],b[None]),
         ('dtype',a.half(),b.half()),('autograd',a.requires_grad_(),b)]
    rejected=[]
    for label,x,y in bad:
        try:combine_pair(x,y,project=True)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    report={'complete':True,'projection_cases':rows,'mean_matches_combined_loss_gradient':True,
            'inputs_unchanged':True,'rejected_inputs':rejected,'optimizer_steps':0}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'projection_cases':len(rows),'guards':rejected,'mean_gradient_equivalent':True}))


if __name__=='__main__':
    main()
