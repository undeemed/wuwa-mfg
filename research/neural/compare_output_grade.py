# SPDX-License-Identifier: Apache-2.0
"""Test observed output grading on saved, exactly paired reconstructions.

CPU image comparison only: no application launch, inference or latency result.
This is an algebraic approximation of one observed first-reset configuration,
not a bit-exact implementation of NVIDIA's texture, HSL or special-function math.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_pre_kernel import DLL_SHA


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_capture(directory):
    frame = json.loads((directory/'frame-0.json').read_text())
    controls = frame['controls']
    if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1
            and controls['DLSSNR.Reset']==1 and controls['DLSSNR.Intensity']==1
            and controls['DLSSNR.Width']==1920 and controls['DLSSNR.Height']==1080):
        raise ValueError('Requires a successful, unit-intensity 1080p first-reset frame.')
    images, hashes = {}, {}
    for role in ('color','output'):
        resource = frame['resources'][role]
        filename = resource['file']
        if Path(filename).name!=filename or not (resource['format']==10
                and resource['width']==1920 and resource['height']==1080
                and resource['row_bytes']==15360 and resource['rows']==1080):
            raise ValueError('Unexpected texture filename, format or extent.')
        file = directory/filename
        if file.stat().st_size!=1080*15360:
            raise ValueError('Unexpected texture payload size.')
        image = np.fromfile(file,dtype='<f2').reshape(1080,1920,4)[...,:3].astype(np.float32)
        if not np.isfinite(image).all():raise ValueError('Nonfinite texture.')
        images[role],hashes[role] = image,sha256(file)
    return controls,images,hashes


def read_contract(trial):
    run = json.loads((trial/'result.json').read_text())
    if run['local_file_sha256']['nvngx_dlssnr.dll']!=DLL_SHA:
        raise ValueError('Unknown runtime: output argument offsets are build-specific.')
    records = [json.loads(line) for line in (trial/'nr-launch-contract.jsonl').read_text().splitlines()]
    records = [r for r in records if r.get('kind')=='launch' and r.get('frame')==1
               and r.get('name')=='cg2r_post_process_kernel']
    if len(records)!=1 or records[0]['status']!=0 or records[0]['param_bytes']!=376:
        raise ValueError('Requires exactly one successful first-frame post-process contract.')
    fields = records[0]['output_scalar_candidate']
    floats,ints,presence = (fields[k] for k in
        ('float_by_offset','int_by_offset','nonzero_64bit_by_offset'))
    expected = {288:1,316:0,320:1,324:float(np.float32(-.1)),328:0,
                332:-.25,336:float(np.float32(-.1)),340:0,344:0,348:0,
                352:0,356:0,360:0,364:0,368:0}
    if any(floats[str(k)]!=v for k,v in expected.items()):
        raise ValueError('This experiment supports only the observed grading controls.')
    if ints!={'292':1920,'296':1080,'300':0,'304':0,'308':0} or presence!={
            '64':True,'96':False,'128':False,'160':False,'192':False,'224':False,'256':False}:
        raise ValueError('Unsupported optional output path.')
    return fields


def observed_grade(image, fields, *, stage='all', round_input=False):
    """Exposure, smoothstep contrast and reduced HSL saturation, then FP16 store.

    With unchanged hue/lightness and saturation multiplied by a factor in [0,1],
    HSL conversion simplifies algebraically to L + factor*(RGB-L). This omits
    native RGB/HSL round-trip and neutral log/exp rounding, so is not bit exact.
    No constants were fitted to target images.
    """
    floats = fields['float_by_offset']
    x = np.clip(image,0,1).astype(np.float32)
    if round_input:x=x.astype(np.float16).astype(np.float32)
    if stage!='storage':
        x=np.clip(x*np.exp2(np.float32(floats['324'])),0,1)
    if stage in ('contrast','all'):
        # Evaluate multiply-adds with a single final float32 rounding here.
        square=x*x
        slope=(3.-2.*x.astype(np.float64)).astype(np.float32)
        delta=(square.astype(np.float64)*slope-x).astype(np.float32)
        x=np.clip((delta.astype(np.float64)*np.float32(floats['332'])+x).astype(np.float32),0,1)
    if stage=='all':
        lightness=(x.max(axis=-1,keepdims=True)+x.min(axis=-1,keepdims=True))*np.float32(.5)
        scale=np.float32(1)+np.float32(floats['336'])
        x=lightness+scale*(x-lightness)
    return np.clip(x,0,1).astype(np.float16).astype(np.float32)


def correlation(x,y):
    return float(np.corrcoef(x.reshape(-1).astype(np.float64),
                            y.reshape(-1).astype(np.float64))[0,1])


def metrics(image,target,source):
    error=image-target
    absolute=np.abs(error)
    mse=float(np.mean(error.astype(np.float64)**2))
    return {'rgb_mae':float(absolute.mean(dtype=np.float64)),
            'rgb_rmse':float(np.sqrt(mse)), 'rgb_max_abs':float(absolute.max()),
            'rgb_p99_abs':float(np.percentile(absolute,99)),
            'psnr_db':float(-10*np.log10(mse)) if mse else None,
            'effect_correlation':correlation(image-source,target-source)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract-trial',type=Path,required=True)
    parser.add_argument('--case',nargs=3,action='append',required=True,
                        metavar=('LABEL','CAPTURE_DIRECTORY','COMPARISON_DIRECTORY'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    fields=read_contract(args.contract_trial)
    controls,_,contract_hashes=read_capture(args.contract_trial/'capture')
    report={'schema':1,'quality_gate_passed':False,'target_achieved':False,
            'scope':'CPU output-composition comparison only; no new model or latency measurement.',
            'arithmetic':'Algebraic approximation; native HSL, texture and special-function rounding not reproduced.',
            'constants_fitted_to_images':False,'contract_capture_hashes':contract_hashes,
            'observed_postprocess_fields':fields,'cases':[]}
    labels=set()
    for label,capture_path,comparison_path in args.case:
        if label in labels:raise ValueError('Duplicate case label.')
        labels.add(label)
        paired_controls,images,hashes=read_capture(Path(capture_path))
        directory=Path(comparison_path)
        comparison=json.loads((directory/'comparison.json').read_text())
        if controls!=paired_controls or paired_controls!=comparison['controls']:
            raise ValueError('Controls differ between observed contract and paired reconstruction.')
        if hashes!=comparison['capture_hashes'] or comparison['noise_frame']!=0:
            raise ValueError('Reconstruction must match both paired texture hashes and reset noise.')
        image=np.load(directory/'reconstruction.npy',allow_pickle=False)
        if image.shape!=(1080,1920,3) or not np.isfinite(image).all():
            raise ValueError('Invalid saved reconstruction.')
        target,source=images['output'],images['color']
        variants={'unchanged':metrics(image,target,source)}
        for name,stage,round_input in [('float16_storage_only','storage',False),
                ('exposure_only','exposure',False),('exposure_and_contrast','contrast',False),
                ('all_observed_grading','all',False),('all_grading_input_fp16','all',True)]:
            variants[name]=metrics(observed_grade(image,fields,stage=stage,round_input=round_input),target,source)
        oracle=bool(comparison.get('native_pre_pool_diagnostic') or comparison.get('native_pre_stem_diagnostic'))
        report['cases'].append({'label':label,'capture_hashes':hashes,
            'reconstruction_sha256':sha256(directory/'reconstruction.npy'),
            'comparison_sha256':sha256(directory/'comparison.json'),
            'same_textures_as_contract_observation':hashes==contract_hashes,
            'parameter_reuse_inference':hashes!=contract_hashes,
            'uses_captured_native_activations':oracle,'metrics':variants})
        print(label,variants['unchanged']['rgb_mae'],
              variants['all_observed_grading']['rgb_mae'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
