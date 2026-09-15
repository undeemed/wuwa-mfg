# SPDX-License-Identifier: MIT
"""Audit a completed private camera collection; export numeric evidence only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_demo_views import HIDDEN_EXE_SHA256
from compare_output_grade import DLL_SHA,read_capture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--baseline-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads(args.collection.read_text())
    baseline=json.loads(args.baseline_result.read_text())
    if not (manifest['scene_restored'] and manifest['dll_restored']):
        raise ValueError('Collection did not report successful restoration.')
    training={h['color'] for h in baseline['training_capture_hashes']}
    validation={baseline['validation_capture_hashes']['color']}
    validation.update(v['capture_hashes']['color'] for v in baseline.get('extra_validation',[]))
    report={'schema':1,'quality_gate_passed':False,'scene_count':1,
            'collection_manifest_sha256':hashlib.sha256(args.collection.read_bytes()).hexdigest(),
            'viewset_sha256':manifest.get('viewset_sha256'),
            'existing_training_views':len(training),'unchanged_validation_views':len(validation),
            'views':[],'scope':'First-reset full-resolution training images in the existing scene; no temporal/cross-scene acceptance.'}
    for view in manifest['views']:
        label=view['label']
        if Path(label).name!=label or view['split']!='train':
            raise ValueError('This audit extends training only, using simple trial labels.')
        trial=args.collection.parent/'trials'/label
        run=json.loads((trial/'result.json').read_text())
        if not (run['local_file_sha256']['ngx_dlss_demo.exe']==HIDDEN_EXE_SHA256
                and run['local_file_sha256']['nvngx_dlssnr.dll']==DLL_SHA
                and run['local_file_sha256']['dxgi.dll']==manifest['capture_dll_sha256']):
            raise ValueError('Unexpected capture runtime or executable.')
        controls,images,hashes=read_capture(trial/'capture')
        if controls!=baseline['controls'] or hashes!=view['capture_hashes']:
            raise ValueError('Capture controls or content hashes changed.')
        if hashes['color'] in training|validation:
            raise ValueError('New training input duplicates existing training or validation.')
        training.add(hashes['color'])
        frames=[]
        for index in range(4):
            frame=json.loads((trial/'capture'/f'frame-{index}.json').read_text())
            if not (frame['complete'] and frame['gpu_completed'] and frame['evaluate_result']==1):
                raise ValueError('Incomplete fenced frame.')
            for resource in frame['resources'].values():
                name=resource['file']
                if Path(name).name!=name:raise ValueError('Unexpected resource path.')
                if (trial/'capture'/name).stat().st_size!=resource['rows']*resource['row_bytes']:
                    raise ValueError('Incomplete texture payload.')
            frames.append({'index':index,'reset':frame['controls']['DLSSNR.Reset']})
        stats={role:{'mean':float(image.mean()),'std':float(image.std()),
                     'min':float(image.min()),'max':float(image.max())} for role,image in images.items()}
        report['views'].append({**view,'complete_fenced_frames':frames,'rgb_statistics':stats})
    report['total_training_views']=len(training)
    report['validation_overlap']=False
    report['all_input_variances_positive']=all(v['rgb_statistics']['color']['std']>0 for v in report['views'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='views'},indent=2))


if __name__=='__main__':main()
