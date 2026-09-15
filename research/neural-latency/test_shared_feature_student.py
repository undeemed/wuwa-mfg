# SPDX-License-Identifier: Apache-2.0
"""Check neutral output, live/cached features, frozen gradients and inference."""
import argparse
import copy
import io
import json
from pathlib import Path

import torch

from compare_output_grade import read_capture
from fused_norm import FusedNorm
from progressive_student import load_first
from shared_feature_student import SharedFeatureRefinement, configure_shared
from student_training_pairs import sha, load_pair, rgb_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('first','capture','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--region-mode',choices=('dense','routed'))
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(args.first)
    controls,_,hashes=read_capture(args.capture)
    assert controls==record['controls'] and hashes in record['training_capture_hashes']
    torch.manual_seed(28411)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    source,target=load_pair({'path':args.capture,'capture_hashes':hashes},controls)
    if args.region_mode:
        from region_context_student import RegionFeatureRefinement
        model=RegionFeatureRefinement(first,args.region_mode)
    else:model=SharedFeatureRefinement(first)
    model=model.cuda().to(memory_format=torch.channels_last)
    saved={key:value.clone() for key,value in model.first.state_dict().items()}
    rows=[]
    for dtype in (torch.float32,torch.float16):
        m=copy.deepcopy(model).to(dtype=dtype).eval();x=source.to(dtype=dtype)
        with torch.no_grad():
            base,features=m.extract(x)
            assert torch.equal(base,m.first.network(x))
            assert all(not tensor.requires_grad for tensor in features)
            expected=m.first(x);actual=m(x);cached=m.refinement(x,base,features)
            assert torch.equal(actual,expected) and torch.equal(actual,cached)
        rows.append({'dtype':str(dtype),'neutral_output_exact':True,'feature_capture_preserves_base':True,
                     'cached_output_exact':True,'feature_shapes':[list(t.shape) for t in features]})
    model.train();assert not model.first.training and model.refinement.training
    loss,_=rgb_loss(model(source),target);loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.first.parameters())
    norm=float(model.refinement.head.weight.grad.norm());assert norm>0
    # Disposable step checks the complete correction path after neutral init.
    optimizer=torch.optim.SGD(model.refinement.parameters(),lr=.01)
    optimizer.step()
    assert all(torch.equal(v,saved[k]) for k,v in model.first.state_dict().items())
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=copy.deepcopy(model);restored.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    with torch.no_grad():assert torch.equal(model(source),restored(source))
    m=copy.deepcopy(model).half().eval();x=source.half();backend=FusedNorm()
    with torch.inference_mode():
        expected=m(x);configure_shared(m,'all-conditioning',backend);actual=m(x)
        assert torch.equal(expected.view(torch.int16),actual.view(torch.int16))
    rejected=[]
    with torch.no_grad():base,features=model.extract(source)
    cases=[('features-count',source,base,features[:-1]),('feature-shape',source,base,(features[0][:,:-1],*features[1:])),
           ('dtype',source,base.half(),features),('rgb',source[:,:2],base,features)]
    for label,x,b,f in cases:
        try:model.refinement(x,b,f)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    for label,sink in [('occupied-sink',[None]),('tuple-sink',())]:
        try:model.first.network(source,feature_sink=sink)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'capture_hashes':hashes,
            'first_checkpoint_sha256':sha(args.first/'student-private.pt'),'cases':rows,
            'first_stage_unchanged':True,'first_stage_has_gradients':False,'head_gradient_norm':norm,
            'strict_roundtrip_exact':True,'existing_fusions_match_bitwise':True,'rejected_inputs':rejected,
            'parameters':{'frozen':sum(p.numel() for p in model.first.parameters()),'trainable':sum(p.numel() for p in model.refinement.parameters())},
            'scope':'One training image, two precisions and one disposable SGD step; no saved candidate or validation fitting.'}
    if args.region_mode:
        from region_context_student import architecture
        report['region_context']=architecture(args.region_mode)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
