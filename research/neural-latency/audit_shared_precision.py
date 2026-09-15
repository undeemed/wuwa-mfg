# SPDX-License-Identifier: Apache-2.0
"""Separate a shared-feature student's validation error from FP16 rounding."""
import argparse,copy,json,statistics
from pathlib import Path
import torch
from compare_output_grade import read_capture
from evaluate_progressive_student import validation_cases
from progressive_student import load_first
from shared_feature_student import SharedFeatureRefinement
from student_training_pairs import read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','lab','photos','images','first','candidate','evaluation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(args.first);evaluation=read(args.evaluation)
    assert evaluation['variant']=='shared-features' and evaluation['candidate_checkpoint_sha256']==sha(args.candidate/'student-private.pt')
    model=SharedFeatureRefinement(first)
    state=torch.load(args.candidate/'student-private.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict'],strict=True)
    model=model.cuda().float().eval().to(memory_format=torch.channels_last)
    half=copy.deepcopy(model).half()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cases=validation_cases(args.base,args.lab,args.photos,args.images,record['controls'])
    training={row['color'] for row in record['training_capture_hashes']}
    rows=[]
    with torch.inference_mode():
        for (label,path,hashes,group),previous in zip(cases,evaluation['validation']):
            controls,arrays,actual=read_capture(path)
            assert hashes==actual==previous['capture_hashes'] and hashes['color'] not in training
            assert controls==record['controls']
            source=torch.from_numpy(arrays['color']).permute(2,0,1).unsqueeze(0).cuda().float().contiguous(memory_format=torch.channels_last)
            target=torch.from_numpy(arrays['output']).permute(2,0,1).unsqueeze(0).cuda().float()
            full=model(source);reduced=half(source.half()).float()
            fp16_mae=float((reduced-target).abs().mean())
            assert fp16_mae==previous['models']['refined']['mae']
            difference=(reduced-full).abs()
            rows.append({'label':label,'group':group,'capture_hashes':hashes,
                         'first_fp32_mae':float((model.first(source)-target).abs().mean()),
                         'refined_fp32_mae':float((full-target).abs().mean()),'refined_fp16_mae':fp16_mae,
                         'fp16_output_mae_to_fp32':float(difference.mean()),'fp16_output_max_to_fp32':float(difference.max())})
    summary={}
    for group in ['all','scene','previous-photos','diverse-photos']:
        selected=[row for row in rows if group=='all' or row['group']==group]
        summary[group]={key:statistics.mean(row[key] for row in selected) for key in
                        ['first_fp32_mae','refined_fp32_mae','refined_fp16_mae','fp16_output_mae_to_fp32']}
    result={'complete':True,'target_achieved':False,'quality_gate_passed':False,
            'candidate_checkpoint_sha256':evaluation['candidate_checkpoint_sha256'],'validation_count':len(rows),
            'all_previous_fp16_mae_reproduced_exactly':True,'summary':summary,'validation':rows,
            'scope':'Read-only FP32/FP16 inference on existing validation identities. No fitting, output arrays, kernel changes or application launch.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
