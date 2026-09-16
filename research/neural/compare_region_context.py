# SPDX-License-Identifier: Apache-2.0
"""Audit the matched region-attention trials and directly time all controls."""
import argparse
import copy
import json
import statistics
from pathlib import Path

import torch

from compare_output_grade import read_capture
from evaluate_progressive_student import paired_timing
from fused_norm import FusedNorm
from progressive_student import load_first
from region_context_student import RegionFeatureRefinement, architecture
from shared_feature_student import SharedFeatureRefinement, configure_shared
from student_training_pairs import read, sha
from test_student_output import configure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('first','shared','shared-evaluation','dense','dense-evaluation','routed','routed-evaluation','capture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(args.first)
    training={name:read(getattr(args,name)/'result.json') for name in ('shared','dense','routed')}
    evaluations={name:read(getattr(args,name+'_evaluation')) for name in ('shared','dense','routed')}
    for name in training:
        row=training[name];evaluation=evaluations[name]
        assert row['complete'] and row['completed_steps']==4500 and evaluation['complete']
        assert row['first_stage_unchanged'] and not row['first_stage_has_gradients']
        assert row['first_checkpoint_sha256']==sha(args.first/'student-private.pt')
        assert row['training_capture_hashes']==record['training_capture_hashes']
        assert row['optimization']==training['shared']['optimization']
        assert evaluation['candidate_checkpoint_sha256']==sha(getattr(args,name)/'student-private.pt')
        assert evaluation['candidate_result_sha256']==sha(getattr(args,name)/'result.json')
        assert evaluation['validation_count']==16 and not evaluation['validation_training_overlap']
        if name!='shared':assert row['region_context']==evaluation['region_context']==architecture(name)
    assert training['dense']['sampled_counts']==training['routed']['sampled_counts']
    assert training['dense']['parameters']==training['routed']['parameters']
    assert sum(training['dense']['sampled_counts'])==9000
    comparisons={}
    for group in ('all','scene','previous-photos','diverse-photos'):
        values={'first':evaluations['shared']['summary'][group]['means']['first']['mae']}
        values.update({name:evaluations[name]['summary'][group]['means']['refined']['mae'] for name in evaluations})
        comparisons[group]={'mean_rgb_error':values,
            'dense_change_vs_first_percent':100*(values['dense']/values['first']-1),
            'routed_change_vs_first_percent':100*(values['routed']/values['first']-1),
            'routed_change_vs_dense_percent':100*(values['routed']/values['dense']-1)}
    for old,dense,routed in zip(*(evaluations[name]['validation'] for name in ('shared','dense','routed'))):
        assert old['capture_hashes']==dense['capture_hashes']==routed['capture_hashes']
        assert old['models']['first']==dense['models']['first']==routed['models']['first']
        assert all(row['first_saved_output_reproduced_exactly'] and row['existing_fusions_match_unfused_bitwise'] for row in (dense,routed))
    controls,arrays,hashes=read_capture(args.capture)
    assert controls==record['controls'] and hashes in record['training_capture_hashes']
    models={'first':first}
    for name in ('shared','dense','routed'):
        model=(SharedFeatureRefinement(copy.deepcopy(first)) if name=='shared'
               else RegionFeatureRefinement(copy.deepcopy(first),name))
        state=torch.load(getattr(args,name)/'student-private.pt',map_location='cpu',weights_only=True)
        assert state['grade_parameters']==first.grade_parameters
        model.load_state_dict(state['state_dict'],strict=True)
        assert all(torch.equal(value,first.state_dict()[key]) for key,value in model.first.state_dict().items())
        models[name]=model
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    backend=FusedNorm()
    for name,model in models.items():
        model.cuda().half().eval().to(memory_format=torch.channels_last)
        if name=='first':model.fused_backend=backend;configure(model,'all-conditioning',backend)
        else:configure_shared(model,'all-conditioning',backend)
    source=torch.from_numpy(arrays['color']).permute(2,0,1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():timing=paired_timing(models,source)
    result={'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_shape':[1080,1920],
            'matched_training_configuration':True,'matched_sample_counts':True,'frozen_first_exact':True,
            'candidate_checkpoint_sha256':{name:sha(getattr(args,name)/'student-private.pt') for name in training},
            'training_record_sha256':{name:sha(getattr(args,name)/'result.json') for name in training},
            'evaluation_sha256':{name:sha(getattr(args,name+'_evaluation')) for name in training},
            'training_mean_mae':{name:statistics.mean(row['refined_mae'] for row in training[name]['training_image_metrics']) for name in training},
            'validation_comparison':comparisons,'timing_capture_hashes':hashes,'timing':timing,
            'scope':'Four complete FP16 model graphs on the same full-1080p input, alternating forward/reverse order. Every graph recomputes the first model and all features; routed graph includes scoring, selection, gather, attention and composition. Existing fused operators are unchanged. D3D12 and game overhead excluded; interval p95 is not a frame-tail measurement.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'validation':comparisons,'timing':timing['summary']},indent=2))


if __name__=='__main__':main()
