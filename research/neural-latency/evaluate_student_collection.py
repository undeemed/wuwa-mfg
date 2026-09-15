# SPDX-License-Identifier: Apache-2.0
"""Compare private student checkpoints on older and newly reserved validation.

No training occurs here. Every evaluated color hash must be absent from every
model's training set. Predictions stay private; export numerical evidence only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np
import torch

from compare_output_grade import read_capture
from fused_norm import FusedNorm
from output_grade import GradedStudent, grade_torch
from student_probe import HierarchicalStudent
from test_output_grade import graph_measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--baseline-result', type=Path, required=True)
    parser.add_argument('--base-trials', type=Path, required=True)
    parser.add_argument('--model', nargs=2, action='append', required=True, metavar=('NAME', 'DIRECTORY'))
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--photo-collection', type=Path, help='Optional private audited photo extension manifest.')
    parser.add_argument('--image-collection', type=Path, help='Optional diverse image extension, requiring --photo-collection.')
    args = parser.parse_args()
    if args.image_collection and not args.photo_collection:
        raise ValueError('The image extension requires its preceding photo collection.')
    repo = Path(__file__).resolve().parents[2]
    if args.output_directory.resolve().is_relative_to(repo):
        raise ValueError('Predictions must stay in a private directory outside the repository.')
    if not 1 <= len(args.model) <= 4:
        raise ValueError('Compare one to four models.')
    collection = json.loads(args.collection.read_text())
    audit = json.loads(args.audit.read_text())
    baseline = json.loads(args.baseline_result.read_text())
    assert len(baseline['training_capture_hashes']) == 18 and len(baseline['extra_validation']) == 1
    assert hashlib.sha256(args.collection.read_bytes()).hexdigest() == audit['collection_manifest_sha256']
    assert collection['scene_restored'] and collection['dll_restored'] and not audit['validation_overlap']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backend = FusedNorm()
    models, previous, model_info = {}, {}, {}
    for name, directory_name in args.model:
        if not re.fullmatch('[a-z0-9-]+', name) or name in models:
            raise ValueError('Use unique simple model names.')
        directory = Path(directory_name)
        record = json.loads((directory / 'result.json').read_text())
        assert record['controls'] == baseline['controls'] and record['native_shape'] == [1080, 1920]
        assert record['completed_steps'] == 4500
        architecture = record['architecture']
        assert architecture['variant'] in ('hierarchical', 'hierarchical-attention')
        assert architecture['width'] in (16, 32) and architecture['blocks'] == 2 and architecture['noise_channels'] == 0
        checkpoint = torch.load(directory / 'student-private.pt', map_location='cpu', weights_only=True)
        assert checkpoint['architecture'] == architecture
        network = HierarchicalStudent(architecture['width'], 2, attention=architecture['variant'] == 'hierarchical-attention')
        model = GradedStudent(network, architecture['explicit_output_grading'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model = model.cuda().half().eval().to(memory_format=torch.channels_last)
        model.fused_backend = backend
        models[name] = model
        saved = {record['validation_capture_hashes']['color']: directory / 'validation-output.npy'}
        saved.update({view['capture_hashes']['color']: directory / f'extra-validation-{index}.npy'
                      for index, view in enumerate(record['extra_validation'])})
        previous[name] = {'saved': saved, 'training': {entry['color'] for entry in record['training_capture_hashes']}}
        model_info[name] = {'architecture': architecture, 'training_view_count': len(previous[name]['training']),
                            'completed_steps': record['completed_steps'],
                            'result_sha256': hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()}
    cases = [('north', args.base_trials / 'model-capture-natural-north/capture', baseline['validation_capture_hashes'], 'previous'),
             ('shifted-north', args.base_trials / 'teacher-natural-shifted-north/capture', baseline['extra_validation'][0]['capture_hashes'], 'previous')]
    for view in collection['views']:
        if view['split'] == 'validation':
            assert Path(view['label']).name == view['label']
            cases.append((view['label'], args.collection.parent / 'trials' / view['label'] / 'capture', view['capture_hashes'], 'new'))
    assert len(cases) == audit['total_validation_views'] and len(cases) > 2
    photo_proofs, image_proofs = [], []
    if args.photo_collection:
        from collect_demo_photo_training import audited_photos
        for row in audited_photos(args.photo_collection, args.base_trials.parent, baseline['controls']):
            if row['split'] == 'validation':
                cases.append((row['label'], args.base_trials.parent / (row['label'] + '-state') / 'capture',
                              row['capture_hashes'], 'photos-emittance-' + str(row['emittance'])))
                photo_proofs.append(row)
    if args.image_collection:
        from collect_diverse_images import audited_images
        for row in audited_images(args.image_collection, args.base_trials.parent, baseline['controls'], args.photo_collection):
            if row['split']=='validation':
                cases.append((row['label'], args.base_trials.parent / (row['label'] + '-state') / 'capture',
                              row['capture_hashes'], 'diverse-extension'))
                image_proofs.append(row)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    rows, timings = [], {}
    with torch.inference_mode():
        for label, capture, expected_hashes, group in cases:
            controls, images, hashes = read_capture(capture)
            assert controls == baseline['controls'] and hashes == expected_hashes
            source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
            target = torch.from_numpy(images['output']).permute(2, 0, 1).unsqueeze(0).cuda().float()
            row = {'label': label, 'validation_group': group, 'capture_hashes': hashes,
                   'input_mean': float(images['color'].mean()), 'input_std': float(images['color'].std()),
                   'input_min': float(images['color'].min()), 'input_max': float(images['color'].max()),
                   'input_fraction_equal_one': float(np.mean(images['color'] == 1)),
                   'input_fraction_above_one': float(np.mean(images['color'] > 1)),
                   'models': {}}

            def metrics(output):
                delta = output.float() - target
                error = delta.abs()
                detail = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
                detail += (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                return {'mae': float(error.mean()), 'rmse': float(delta.square().mean().sqrt()),
                        'p99': float(torch.quantile(error.flatten(), .99)), 'gradient_error': float(detail)}

            row['identity'] = metrics(source.clamp(0, 1))
            row['grade_only'] = metrics(grade_torch(source, baseline['architecture']['explicit_output_grading']))
            for name, model in models.items():
                assert hashes['color'] not in previous[name]['training'], 'Validation image was used in training.'
                prediction = model(source)
                produced = prediction[0].permute(1, 2, 0).float().cpu().numpy()
                assert np.isfinite(produced).all()
                score = metrics(prediction)
                saved = previous[name]['saved'].get(hashes['color'])
                if saved is not None:
                    score['saved_output_reproduced_exactly'] = bool(np.array_equal(np.load(saved, allow_pickle=False), produced))
                    assert score['saved_output_reproduced_exactly']
                row['models'][name] = score
                np.save(args.output_directory / f'{name}-{label}.npy', produced)
                if name not in timings:
                    timings[name], graph_output = graph_measure(model, source)
                    timings[name]['graph_matches_eager'] = bool(torch.equal(graph_output, prediction))
                    assert timings[name]['graph_matches_eager']
            rows.append(row)
            print(json.dumps({'label': label, 'mae': {name: score['mae'] for name, score in row['models'].items()}}), flush=True)
    summary = {group: {name: statistics.mean(row['models'][name]['mae'] for row in rows
                                           if group == 'all' or row['validation_group'] == group)
                       for name in models} for group in ['previous', 'new', 'all'] + sorted({r['validation_group'] for r in rows} - {'previous', 'new'})}
    report = {'schema': 1, 'target_achieved': False, 'quality_gate_passed': False,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'models': model_info, 'validation': rows, 'mean_validation_mae': summary,
              'complete_graph_timings': timings,
              'photo_validation_proofs': photo_proofs,
              'image_validation_proofs': image_proofs,
              'timing_scope': 'Full-1080p FP16 network plus output grade; no D3D12/application integration.',
              'limitations': ['One 3D scene, optionally supplemented by static photograph planes; no representative game or temporal acceptance.',
                              'Validation identities stay out of fitting, but previous validation scores have informed research choices; not an independent final test set.',
                              'Pixel metrics do not establish preserved perceptual quality. No replacement is accepted.']}
    (args.output_directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'mean_validation_mae': summary, 'median_ms': {name: value['median_ms'] for name, value in timings.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
