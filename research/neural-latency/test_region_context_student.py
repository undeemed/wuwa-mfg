# SPDX-License-Identifier: Apache-2.0
"""CPU checks for routed attention, masking, gradients and frozen detail path."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch

from progressive_student import load_first
from region_context_student import RegionContext, RegionFeatureRefinement
from student_training_pairs import sha


def reference_attention(module, q, k, v, valid, routes):
    batches=[]
    for batch in range(q.shape[0]):
        regions=[]
        for region in range(q.shape[1]):
            selected=routes[batch,region].tolist() if routes is not None else list(range(q.shape[1]))
            keys=torch.cat([k[batch,index][valid[batch,index]] for index in selected])
            values=torch.cat([v[batch,index][valid[batch,index]] for index in selected])
            heads=[]
            width=module.inner//module.heads
            for head in range(module.heads):
                sl=slice(head*width,(head+1)*width)
                probabilities=(q[batch,region,:,sl]@keys[:,sl].T/width**.5).softmax(dim=-1)
                heads.append(probabilities@values[:,sl])
            regions.append(torch.cat(heads,dim=-1))
        batches.append(torch.stack(regions))
    return torch.stack(batches)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    torch.set_num_threads(2)
    torch.manual_seed(28411)
    rows=[]
    for height,width in ((4,6),(5,7),(1,1),(3,2)):
        module=RegionContext(6,6,2,2,2).double()
        source=torch.randn(2,6,height,width,dtype=torch.float64,requires_grad=True)
        q,k,v,valid,_=module.prepare(source)
        routes=module.route(q,k,valid)
        count=min(module.selected_regions,q.shape[1])
        assert routes.shape==(2,q.shape[1],count)
        assert torch.equal(routes[:,:,0],torch.arange(q.shape[1]).expand(2,-1))
        assert all(len(set(row))==count for batch in routes.tolist() for row in batch)
        for mode in ('dense','routed'):
            indices=routes if mode=='routed' else None
            actual=module.attend(q,k,v,valid,indices)
            expected=reference_attention(module,q,k,v,valid,indices)
            torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
            grad_actual=torch.autograd.grad(actual.square().mean(),source,retain_graph=True)[0]
            grad_expected=torch.autograd.grad(expected.square().mean(),source,retain_graph=True)[0]
            torch.testing.assert_close(grad_actual,grad_expected,atol=1e-11,rtol=1e-10)
        altered=[tensor.detach().clone() for tensor in (q,k,v)]
        for tensor in altered:tensor[~valid]=10000
        assert torch.equal(routes,module.route(altered[0],altered[1],valid))
        clean=module.attend(q,k,v,valid,routes)
        poisoned=module.attend(*altered,valid,routes)
        torch.testing.assert_close(clean[valid],poisoned[valid],atol=1e-11,rtol=1e-10)
        # Selecting every region recovers dense attention, including odd-size padding.
        module.selected_regions=q.shape[1]
        all_routes=module.route(q,k,valid)
        dense=module.attend(q,k,v,valid)
        routed=module.attend(q,k,v,valid,all_routes)
        torch.testing.assert_close(dense,routed,atol=1e-11,rtol=1e-10)
        rows.append({'shape':[2,6,height,width],'regions':q.shape[1],'selected_regions':count,
                     'independent_attention_and_input_gradient_match':True,'all_regions_match_dense':True,
                     'padding_poison_does_not_change_valid_outputs_or_routes':True,'self_included_without_duplicates':True})
    # Actual deepest-feature geometry for padded 1920x1080 input.
    module=RegionContext()
    with torch.no_grad():
        source=torch.randn(1,96,34,60)
        q,k,v,valid,_=module.prepare(source);routes=module.route(q,k,valid)
        assert q.shape==(1,135,16,48) and routes.shape==(1,135,4)
        assert int(valid.sum())==34*60
        assert module(source).shape==source.shape
    first,_=load_first(args.first)
    torch.manual_seed(28411);dense=RegionFeatureRefinement(copy.deepcopy(first),'dense')
    torch.manual_seed(28411);routed=RegionFeatureRefinement(copy.deepcopy(first),'routed')
    assert all(torch.equal(value,routed.state_dict()[key]) for key,value in dense.state_dict().items())
    source=torch.rand(1,3,160,288)
    target=torch.rand_like(source)
    frozen={key:value.clone() for key,value in first.state_dict().items()}
    neutral=[]
    for model in (dense,routed):
        with torch.no_grad():
            base,features=model.extract(source)
            assert torch.equal(model(source),model.first(source))
            assert torch.equal(model(source),model.refinement(source,base,features))
        optimizer=torch.optim.SGD(model.refinement.parameters(),lr=.1)
        model.train();assert not model.first.training
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(source)-target).square().mean();loss.backward();optimizer.step()
        branch=model.refinement.deep[1]
        gradients={name:float(parameter.grad.norm()) for name,parameter in branch.named_parameters()}
        assert all(value>0 for value in gradients.values())
        assert all(p.grad is None and not p.requires_grad for p in model.first.parameters())
        assert all(torch.equal(value,frozen[key]) for key,value in model.first.state_dict().items())
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=copy.deepcopy(model);restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        with torch.no_grad():assert torch.equal(model(source),restored(source))
        neutral.append({'mode':branch.mode,'neutral_output_exact':True,'live_cached_features_equal':True,
                        'frozen_first_unchanged':True,'strict_reload_exact':True,'branch_gradient_norms':gradients})
    rejected=[]
    for label,kwargs in [('mode',{'mode':'invalid'}),('heads',{'inner':7,'heads':3}),
                         ('region-size',{'region_size':0}),('selection',{'selected_regions':0})]:
        try:RegionContext(**kwargs)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'device':'cpu','threads':2,
            'first_checkpoint_sha256':sha(args.first/'student-private.pt'),'attention_cases':rows,
            'full_extent_geometry':{'input':[1080,1920],'deepest_features':[34,60],'regions':135,
                                    'tokens_per_region':16,'keys_per_query_region':64,'valid_tokens':2040},
            'matched_initial_weights':True,'student_cases':neutral,'rejected_inputs':rejected,
            'scope':'Synthetic correctness and gradient checks only. No GPU benchmark, training candidate or quality acceptance.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
