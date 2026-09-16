# SPDX-License-Identifier: Apache-2.0
"""Evaluate frozen students on private, previously unseen photo-plane captures.

No training or tuning occurs. Pixel scores are diagnostics, not perceptual or
temporal acceptance. Complete CUDA Graph timings exclude D3D12 integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from compare_output_grade import DLL_SHA, read_capture
from collect_demo_views import HIDDEN_EXE_SHA256
from fused_norm import FusedNorm
from output_grade import GradedStudent, grade_torch
from student_probe import HierarchicalStudent
from test_output_grade import graph_measure


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_capture(base, label):
    state = base / (label + '-state')
    record = read(state / 'result.json')
    run = read(base / 'trials' / label / 'result.json')
    assert all(record['restored'].values()) and record['capture_archived']
    assert record['first_reset_capture_complete'] and run['model_1920x1080_confirmed']
    assert run['local_file_sha256']['ngx_dlss_demo.exe'] == HIDDEN_EXE_SHA256
    assert run['local_file_sha256']['nvngx_dlssnr.dll'] == DLL_SHA
    assert run['local_file_sha256']['dxgi.dll'] == record['capture_dll_sha256']
    desktop = run['isolated_desktop_observations']
    assert len(desktop) >= 3 and all(x['separate_from_input_desktop'] and x['input_desktop_unchanged'] for x in desktop)
    assert any(any(w['title'] == 'NVIDIA NGX DLSS Sample (D3D12)' for w in x['windows']) for x in desktop)
    assert not any(any(w['visible_on_private_desktop'] for w in x['windows']) for x in desktop)
    frames = []
    for index in range(4):
        frame = read(state / 'capture' / f'frame-{index}.json')
        assert frame['complete'] and frame['gpu_completed'] and frame['evaluate_result'] == 1
        for resource in frame['resources'].values():
            assert Path(resource['file']).name == resource['file']
            assert (state / 'capture' / resource['file']).stat().st_size == resource['rows'] * resource['row_bytes']
        frames.append({'index': index, 'reset': frame['controls']['DLSSNR.Reset']})
    controls, images, hashes = read_capture(state / 'capture')
    assert hashes == record['capture_hashes'] and controls == record['controls']
    proof = {'label': label, 'capture_hashes': hashes, 'complete_fenced_frames': frames,
             'runtime_sha256': run['local_file_sha256'], 'restored': record['restored'],
             'inactive_desktop_observations': len(desktop),
             'all_demo_windows_hidden_and_off_input_desktop': True,
             'native_sparse_timing': run.get('summary'), 'native_retained_timing_count': run['retained_count']}
    return controls, images, hashes, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--plain', type=Path, required=True)
    parser.add_argument('--attention', type=Path, required=True)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Predictions must remain private.')
    selection, photos = read(args.photos / 'selection.json'), read(args.photos / 'manifest.json')
    assert [x['name'] for x in photos['cases']] == ['portrait', 'landscape', 'cat']
    for selected, prepared in zip(selection['selected_sources'], photos['cases']):
        assert all(prepared[key] == value for key, value in selected.items())
        assert prepared['split'] == 'validation'
        assert sha(args.photos / (prepared['name'] + '-original.jpg')) == prepared['source_sha256']
        assert sha(args.photos / (prepared['name'] + '-1080.png')) == prepared['texture_sha256']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backend = FusedNorm()
    models, information, training = {}, {}, {}
    for name in ('plain', 'attention', 'wide'):
        directory = getattr(args, name)
        record = read(directory / 'result.json')
        architecture = record['architecture']
        assert record['completed_steps'] == 4500 and record['data_split']['training_view_count'] == 30
        assert architecture['width'] == (32 if name == 'wide' else 16)
        assert architecture['variant'] == ('hierarchical-attention' if name == 'attention' else 'hierarchical')
        checkpoint = torch.load(directory / 'student-private.pt', map_location='cpu', weights_only=True)
        assert checkpoint['architecture'] == architecture
        network = HierarchicalStudent(architecture['width'], 2, attention=name == 'attention')
        model = GradedStudent(network, architecture['explicit_output_grading'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model = model.cuda().half().eval().to(memory_format=torch.channels_last)
        model.fused_backend = backend
        models[name] = model
        training[name] = {entry['color'] for entry in record['training_capture_hashes']}
        information[name] = {'architecture': architecture, 'record_sha256': sha(directory / 'result.json'),
                             'checkpoint_sha256': sha(directory / 'student-private.pt'),
                             'training_view_count': 30, 'steps': 4500}
        if name == 'plain':
            controls_expected = record['controls']
            grade = architecture['explicit_output_grading']
        assert record['controls'] == controls_expected
    args.output.mkdir(parents=True, exist_ok=False)
    rows, timings = [], {}
    with torch.inference_mode():
        for case in photos['cases']:
            name = case['name']
            controls, images, hashes, proof = audit_capture(args.base, 'teacher-photo-' + name)
            assert controls == controls_expected
            assert all(hashes['color'] not in values for values in training.values())
            source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
            target = torch.from_numpy(images['output']).permute(2, 0, 1).unsqueeze(0).cuda().float()

            def metrics(value):
                delta = value.float() - target
                return {'mae': float(delta.abs().mean()), 'rmse': float(delta.square().mean().sqrt()),
                        'p99': float(torch.quantile(delta.abs().flatten(), .99)),
                        'gradient_error': float((delta[:,:,1:]-delta[:,:,:-1]).abs().mean()
                                               + (delta[:,:,:,1:]-delta[:,:,:,:-1]).abs().mean())}

            row = {'name': name, 'capture_proof': proof, 'input_statistics': {
                'mean': float(images['color'].mean()), 'std': float(images['color'].std()),
                'fraction_at_least_one': float(np.mean(images['color'] >= 1)),
                'fraction_at_most_zero': float(np.mean(images['color'] <= 0))},
                'identity': metrics(source), 'grade_only': metrics(grade_torch(source, grade)), 'models': {}}
            for model_name, model in models.items():
                prediction = model(source)
                array = prediction[0].permute(1, 2, 0).float().cpu().numpy()
                assert np.isfinite(array).all()
                row['models'][model_name] = metrics(prediction)
                np.save(args.output / f'{model_name}-{name}.npy', array)
                if model_name not in timings:
                    timings[model_name], graph = graph_measure(model, source)
                    timings[model_name]['graph_matches_eager'] = bool(torch.equal(graph, prediction))
                    assert timings[model_name]['graph_matches_eager']
            rows.append(row)
            print(json.dumps({'case': name, 'models': row['models'], 'grade_only': row['grade_only']}), flush=True)
    report = {'schema': 1, 'target_achieved': False, 'quality_gate_passed': False,
              'selection_sha256': sha(args.photos / 'selection.json'), 'photo_manifest': photos,
              'models': information, 'cases': rows, 'complete_graph_timings': timings,
              'mean_mae': {name: statistics.mean(row['models'][name]['mae'] for row in rows) for name in models},
              'mean_grade_only_mae': statistics.mean(row['grade_only']['mae'] for row in rows),
              'limitations': ['Three unseen photographs rendered on a static emissive plane; not representative game scenes or temporal validation.',
                  'Images are cropped/resampled to 1080p, then rendered/filtered/exposed by the sample. Targets use the actual captured model input.',
                  'Some captured channels reach or exceed one; the recorded >=1 fraction is not a count of exactly clipped values. These are not source-photo fidelity tests.',
                  'Native sparse GPU timings and full Torch graph timings are different scopes; no D3D12 student integration or native speedup.',
                  'Frozen students were fitted only on 30 Sponza views. No training occurs here; these results can inform future work but are not a final quality gate.']}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'mean_mae': report['mean_mae'], 'mean_grade_only_mae': report['mean_grade_only_mae'],
                      'graph_ms': {name: value['median_ms'] for name, value in timings.items()}}, indent=2))


if __name__ == '__main__':
    main()
