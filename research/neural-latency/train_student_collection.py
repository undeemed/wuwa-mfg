# SPDX-License-Identifier: MIT
"""Train a bounded student using the original 18 views plus an audited collection.

The collection's explicit split determines which additional captures can enter
training. Both older validation views remain excluded. Inputs and checkpoints
must be kept outside the public repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def original_training_names():
    names = ['model-capture-natural', 'model-capture-natural-west',
             'teacher-natural-northeast', 'teacher-natural-northwest',
             'teacher-natural-southeast', 'teacher-natural-southwest']
    poses = json.loads((Path(__file__).parent / 'capture-viewsets/translated-training.json').read_text())
    return names + ['teacher-translated-' + pose['name'] for pose in poses]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--baseline-result', type=Path, required=True)
    parser.add_argument('--base-trials', type=Path, required=True)
    parser.add_argument('--grade-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--architecture', choices=['hierarchical', 'hierarchical-attention', 'hierarchical-film','hierarchical-latent'], required=True)
    parser.add_argument('--width', type=int, choices=[16, 32], default=16)
    parser.add_argument('--photo-collection', type=Path, help='Optional private audited photo extension manifest.')
    parser.add_argument('--image-collection', type=Path, help='Optional diverse image extension, requiring --photo-collection.')
    parser.add_argument('--initial-lr', type=float, default=.002)
    parser.add_argument('--final-lr', type=float, default=.00002)
    parser.add_argument('--initialize-from', type=Path, help='Optional private matching student run for weights-only initialization.')
    parser.add_argument('--feature-targets', type=Path, help='Optional private native feature target manifest.')
    parser.add_argument('--feature-weight', type=float, default=.01)
    parser.add_argument('--paired-gradient',choices=['mean','pcgrad'],help='Matched two-domain training; requires the diverse image collection.')
    args = parser.parse_args()
    if args.image_collection and not args.photo_collection:
        raise ValueError('The image extension requires its preceding photo collection.')
    if args.paired_gradient and not args.image_collection:
        raise ValueError('Paired training requires the fixed 46-frame collection.')
    repo = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repo) or args.output.exists():
        raise ValueError('Use a fresh private output directory outside the repository.')
    collection = json.loads(args.collection.read_text())
    audit = json.loads(args.audit.read_text())
    baseline = json.loads(args.baseline_result.read_text())
    if not collection['scene_restored'] or not collection['dll_restored']:
        raise ValueError('Collection has not completed restoration.')
    if hashlib.sha256(args.collection.read_bytes()).hexdigest() != audit['collection_manifest_sha256']:
        raise ValueError('Collection changed after its audit.')
    if audit['validation_overlap'] or not audit['all_input_variances_positive']:
        raise ValueError('Collection audit failed.')

    def verify_capture(path, expected_hashes):
        metadata = json.loads((path / 'frame-0.json').read_text())
        assert metadata['complete'] and metadata['gpu_completed'] and metadata['evaluate_result'] == 1
        assert metadata['controls'] == baseline['controls']
        for role in ('color', 'output'):
            resource = metadata['resources'][role]
            assert Path(resource['file']).name == resource['file']
            assert hashlib.sha256((path / resource['file']).read_bytes()).hexdigest() == expected_hashes[role]
        return path

    names = original_training_names()
    assert len(names) == len(baseline['training_capture_hashes']) == 18
    train = [verify_capture(args.base_trials / name / 'capture', hashes)
             for name, hashes in zip(names, baseline['training_capture_hashes'])]
    validation = [verify_capture(args.base_trials / 'model-capture-natural-north/capture', baseline['validation_capture_hashes']),
                  verify_capture(args.base_trials / 'teacher-natural-shifted-north/capture', baseline['extra_validation'][0]['capture_hashes'])]
    audited = {view['label']: view for view in audit['views']}
    assert len(audited) == len(collection['views'])
    for view in collection['views']:
        label = view['label']
        assert Path(label).name == label
        assert view['capture_hashes'] == audited[label]['capture_hashes'] and view['split'] == audited[label]['split']
        path = verify_capture(args.collection.parent / 'trials' / label / 'capture', view['capture_hashes'])
        if view['split'] == 'train':
            train.append(path)
        elif view['split'] == 'validation':
            validation.append(path)
        else:
            raise ValueError('Unrecognized data split.')
    assert len(train) == audit['total_training_views']
    assert len(validation) == audit['total_validation_views']
    if args.photo_collection:
        from collect_demo_photo_training import audited_photos
        for row in audited_photos(args.photo_collection, args.base_trials.parent, baseline['controls']):
            path = verify_capture(args.base_trials.parent / (row['label'] + '-state') / 'capture', row['capture_hashes'])
            (train if row['split'] == 'train' else validation).append(path)
    if args.image_collection:
        from collect_diverse_images import audited_images
        for row in audited_images(args.image_collection, args.base_trials.parent, baseline['controls'], args.photo_collection):
            path = verify_capture(args.base_trials.parent / (row['label'] + '-state') / 'capture', row['capture_hashes'])
            (train if row['split'] == 'train' else validation).append(path)
    command = [sys.executable, str(Path(__file__).parent / 'student_probe.py'),
               '--capture', str(train[0]), '--validation-capture', str(validation[0])]
    for path in train[1:]: command.extend(['--extra-train-capture', str(path)])
    for path in validation[1:]: command.extend(['--extra-validation-capture', str(path)])
    command.extend(['--output', str(args.output), '--steps', '4500', '--max-seconds', '600',
                    '--width', str(args.width), '--blocks', '2', '--loss-border', '0', '--batch', '1', '--cosine-lr',
                    '--architecture', args.architecture, '--whole-frame', '--output-grade-contract', str(args.grade_contract),
                    '--evaluate-all-training'])
    command.extend(['--initial-lr', str(args.initial_lr), '--final-lr', str(args.final_lr)])
    if args.initialize_from:
        command.extend(['--initialize-from', str(args.initialize_from)])
    if args.feature_targets:
        command.extend(['--feature-targets', str(args.feature_targets), '--feature-weight', str(args.feature_weight)])
    if args.paired_gradient:
        assert len(train)==46 and len(validation)==16
        command.extend(['--paired-gradient',args.paired_gradient])
    if args.image_collection:
        command.extend(['--data-description', 'One Sponza scene plus sixteen training photo identities; seven different photo identities stay in validation, three at two emissions. Image identities and splits were fixed before the new captures and fitting. First-reset static frames only, not representative game or temporal validation.'])
    elif args.photo_collection:
        command.extend(['--data-description', 'One Sponza scene plus four training photo identities; three different photo identities stay in validation at two emissions. First-reset frames only, not representative game or temporal validation.'])
    print(json.dumps({'architecture': args.architecture, 'width': args.width, 'training_views': len(train),
                      'validation_views': len(validation), 'steps': 4500}), flush=True)
    subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


if __name__ == '__main__':
    main()
