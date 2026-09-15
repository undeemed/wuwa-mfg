# SPDX-License-Identifier: Apache-2.0
"""Evaluate a frozen-base refinement and time both stages at full 1080p.

Predictions and checkpoints must remain outside the repository. Existing
validation scores informed research choices; these are not final acceptance.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from collect_demo_photo_training import audited_photos
from collect_diverse_images import audited_images
from compare_output_grade import read_capture
from fused_norm import FusedNorm
from progressive_student import FrozenStudentRefinement, configure_progressive, load_first
from student_training_pairs import read, sha
from test_student_output import configure, same


def validation_cases(base, lab, photos, images, controls):
    collection_path = base / 'teacher-illumination-manifest.json'
    collection = read(collection_path)
    audit = read(lab / 'native-output-inspection/teacher-illumination-audit.json')
    initial = read(lab / 'student-natural-hierarchical-eighteenviews-grade-long/result.json')
    assert sha(collection_path) == audit['collection_manifest_sha256']
    assert collection['scene_restored'] and collection['dll_restored'] and not audit['validation_overlap']
    assert initial['controls'] == controls
    cases = [
        ('north', base / 'trials/model-capture-natural-north/capture', initial['validation_capture_hashes'], 'scene'),
        ('shifted-north', base / 'trials/teacher-natural-shifted-north/capture', initial['extra_validation'][0]['capture_hashes'], 'scene'),
    ]
    for row in collection['views']:
        if row['split'] == 'validation':
            assert Path(row['label']).name == row['label']
            cases.append((row['label'], base / 'trials' / row['label'] / 'capture', row['capture_hashes'], 'scene'))
    assert len(cases) == audit['total_validation_views'] == 6
    for rows, group in [(audited_photos(photos, base, controls), 'previous-photos'),
                        (audited_images(images, base, controls, photos), 'diverse-photos')]:
        for row in rows:
            if row['split'] == 'validation':
                cases.append((row['label'], base / (row['label'] + '-state') / 'capture', row['capture_hashes'], group))
    assert len(cases) == len({row[2]['color'] for row in cases}) == 16
    return cases


def metrics(output, target):
    delta = output.float() - target
    error = delta.abs()
    detail = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
    detail += (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return {'mae': float(error.mean()), 'rmse': float(delta.square().mean().sqrt()),
            'p99': float(torch.quantile(error.flatten(), .99)), 'gradient_error': float(detail)}


def paired_timing(models, source):
    graphs, outputs, references = {}, {}, {}
    for name, model in models.items():
        references[name] = model(source).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                model(source)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(source)
        graphs[name], outputs[name] = graph, output
    for _ in range(40):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    assert all(same(outputs[name], references[name]) for name in models)
    rows = []
    for index in range(30):
        order = list(models) if index % 2 == 0 else list(reversed(models))
        times = {}
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name] = start.elapsed_time(end) / 10
        rows.append({'order': order, 'ms_per_replay': times})
    assert all(same(outputs[name], references[name]) for name in models)
    return {'measurements': rows, 'all_graphs_match_eager_bitwise': True,
            'warmup_replays_per_model': 40, 'replays_per_interval': 10, 'alternating_pairs': 30,
            'summary': {name: {'median_ms': statistics.median(row['ms_per_replay'][name] for row in rows),
                               'p95_interval_ms': float(np.percentile([row['ms_per_replay'][name] for row in rows], 95))}
                        for name in models}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'lab', 'photos', 'images', 'first', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    first, record = load_first(args.first)
    second_first, _ = load_first(args.first)
    candidate_record = read(args.candidate / 'result.json')
    assert candidate_record['complete'] and candidate_record['completed_steps'] == 4500
    assert candidate_record['first_stage_unchanged'] and not candidate_record['first_stage_has_gradients']
    assert candidate_record['first_checkpoint_sha256'] == sha(args.first / 'student-private.pt')
    assert candidate_record['first_result_sha256'] == sha(args.first / 'result.json')
    assert candidate_record['controls'] == record['controls'] and candidate_record['native_shape'] == [1080, 1920]
    assert candidate_record['training_capture_hashes'] == record['training_capture_hashes']
    candidate = FrozenStudentRefinement(second_first)
    state = torch.load(args.candidate / 'student-private.pt', map_location='cpu', weights_only=True)
    assert state['grade_parameters'] == first.grade_parameters
    candidate.load_state_dict(state['state_dict'], strict=True)
    assert all(torch.equal(value, first.state_dict()[key]) for key, value in candidate.first.state_dict().items())
    models = {name: model.cuda().half().eval().to(memory_format=torch.channels_last)
              for name, model in [('first', first), ('refined', candidate)]}
    kernel = FusedNorm()
    first.fused_backend = kernel
    cases = validation_cases(args.base, args.lab, args.photos, args.images, record['controls'])
    training = {row['color'] for row in record['training_capture_hashes']}
    expected_validation = {record['validation_capture_hashes']['color']} | {row['capture_hashes']['color'] for row in record['extra_validation']}
    assert len(training) == 62 and {row[2]['color'] for row in cases} == expected_validation
    assert not training & expected_validation
    saved = {record['validation_capture_hashes']['color']: args.first / 'validation-output.npy'}
    saved.update({row['capture_hashes']['color']: args.first / f'extra-validation-{index}.npy'
                  for index, row in enumerate(record['extra_validation'])})
    args.output.mkdir(parents=True)
    rows, timing_source = [], None
    with torch.inference_mode():
        for label, path, expected_hashes, group in cases:
            controls, arrays, hashes = read_capture(path)
            assert controls == record['controls'] and hashes == expected_hashes
            assert arrays['color'].shape == arrays['output'].shape == (1080, 1920, 3)
            source = torch.from_numpy(arrays['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
            target = torch.from_numpy(arrays['output']).permute(2, 0, 1).unsqueeze(0).cuda().float()
            if timing_source is None:
                timing_source = source.clone()
            configure(first, 'baseline', kernel)
            configure_progressive(candidate, 'baseline', kernel)
            references = {name: model(source) for name, model in models.items()}
            produced_first = references['first'][0].permute(1, 2, 0).float().cpu().numpy()
            assert np.array_equal(produced_first, np.load(saved[hashes['color']], allow_pickle=False))
            configure(first, 'all-conditioning', kernel)
            configure_progressive(candidate, 'all-conditioning', kernel)
            assert all(same(model(source), references[name]) for name, model in models.items())
            produced = references['refined'][0].permute(1, 2, 0).float().cpu().numpy()
            assert np.isfinite(produced).all()
            np.save(args.output / f'refined-{label}.npy', produced)
            row = {'label': label, 'group': group, 'capture_hashes': hashes,
                   'first_saved_output_reproduced_exactly': True, 'existing_fusions_match_unfused_bitwise': True,
                   'models': {name: metrics(output, target) for name, output in references.items()}}
            rows.append(row)
            print(json.dumps({'label': label, 'mae': {name: score['mae'] for name, score in row['models'].items()}}), flush=True)
        timings = paired_timing(models, timing_source)
    summary = {}
    for group in ['all', 'scene', 'previous-photos', 'diverse-photos']:
        subset = [row for row in rows if group == 'all' or row['group'] == group]
        means = {name: {metric: statistics.mean(row['models'][name][metric] for row in subset)
                         for metric in ('mae', 'rmse', 'p99', 'gradient_error')} for name in models}
        summary[group] = {'count': len(subset), 'means': means,
                          'improved_mae_count': sum(row['models']['refined']['mae'] < row['models']['first']['mae'] for row in subset),
                          'mae_change_percent': 100 * (means['refined']['mae'] / means['first']['mae'] - 1)}
    report = {'complete': True, 'target_achieved': False, 'quality_gate_passed': False,
              'native_shape': [1080, 1920], 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'first_checkpoint_sha256': sha(args.first / 'student-private.pt'),
              'candidate_checkpoint_sha256': sha(args.candidate / 'student-private.pt'),
              'candidate_result_sha256': sha(args.candidate / 'result.json'),
              'frozen_first_state_exact': True, 'training_count': 62, 'validation_count': 16, 'validation_training_overlap': False,
              'validation': rows, 'summary': summary, 'timing': timings,
              'timing_scope': 'FP16 full-1920x1080 input through both models, intermediate concatenation, and one final grade. No cached first output during inference. Existing fusions reused unchanged.',
              'limitations': ['No D3D12 integration or game/temporal acceptance. Native runtime is unchanged.',
                              'p95 is over ten-replay interval averages, not individual frame tails.',
                              'Previously reserved validation images remain excluded from training, but their past scores informed research; this is not an independent final acceptance set.',
                              'The second stage adds 4500 updates and 9000 image examples on top of the existing trained base; this is not an equal-total-training-compute architecture comparison.',
                              'Pixel metrics do not establish perceptual equivalence. No replacement is accepted.']}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': summary, 'timing': timings['summary']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
