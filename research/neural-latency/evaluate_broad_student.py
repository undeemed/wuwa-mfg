# SPDX-License-Identifier: Apache-2.0
"""Compare the unchanged routed architecture before/after broad-data fitting.

The 56 development images are not final gameplay trials. Checkpoints, captures
and optional preview arrays remain private. Timings cover complete standalone
models, not D3D12 conversion, synchronization or composition.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
import torch

from compare_output_grade import read_capture
from evaluate_progressive_student import metrics, paired_timing, validation_cases
from fused_norm import FusedNorm
from progressive_student import load_first
from region_context_student import RegionFeatureRefinement, architecture
from shared_feature_student import configure_shared
from student_training_pairs import read, sha
from test_student_output import configure, same


def summarize(rows, names):
    result = {}
    groups = ['all', 'old-development', *dict.fromkeys(r['group'] for r in rows)]
    for group in groups:
        subset = [r for r in rows if group == 'all' or r['group'] == group
                  or (group == 'old-development' and r['group'] != 'new-photos')]
        means = {name: {metric: statistics.mean(r['models'][name][metric] for r in subset)
                        for metric in ('mae', 'rmse', 'p99', 'gradient_error')} for name in names}
        result[group] = {'count': len(subset), 'means': means,
                         'improved_mae_count': sum(r['models']['broad']['mae'] < r['models']['previous']['mae'] for r in subset),
                         'mae_change_percent': 100*(means['broad']['mae']/means['previous']['mae']-1)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'lab', 'photos', 'images', 'first', 'previous', 'previous-evaluation',
                 'candidate', 'sources', 'teacher', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--save-case', action='append', default=[], help='Save private arrays for named cases only.')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    running = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if running.stdout.strip() != '0':
        raise RuntimeError('Close the game and finish capture before benchmarking.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    first, record = load_first(args.first)
    fitted = read(args.candidate/'result.json')
    previous_record = read(args.previous/'result.json')
    previous_eval = read(args.previous_evaluation)
    sources, teacher = read(args.sources), read(args.teacher)
    assert sources['complete'] and teacher['complete'] and len(teacher['cases']) == 240
    assert teacher['prepared_manifest_sha256'] == sha(args.sources) == fitted['source_manifest_sha256']
    assert fitted['teacher_manifest_sha256'] == sha(args.teacher)
    assert fitted['complete'] and fitted['completed_steps'] == 6000 and fitted['training_frames'] == 262
    assert fitted['region_context'] == previous_record['region_context'] == architecture('routed')
    assert fitted['first_stage_unchanged'] and not fitted['first_stage_has_gradients']
    assert fitted['first_checkpoint_sha256'] == previous_record['first_checkpoint_sha256'] == sha(args.first/'student-private.pt')
    assert fitted['first_result_sha256'] == previous_record['first_result_sha256'] == sha(args.first/'result.json')
    assert fitted['starting_checkpoint_sha256'] == previous_eval['candidate_checkpoint_sha256'] == sha(args.previous/'student-private.pt')
    assert fitted['starting_result_sha256'] == previous_eval['candidate_result_sha256'] == sha(args.previous/'result.json')
    assert fitted['checkpoint_sha256'] == sha(args.candidate/'student-private.pt')
    assert fitted['controls'] == previous_record['controls'] == record['controls']
    assert fitted['native_shape'] == [1080, 1920] and fitted['parameters'] == 381600
    models = {'first': first}
    for name, directory in [('previous', args.previous), ('broad', args.candidate)]:
        base, _ = load_first(args.first)
        model = RegionFeatureRefinement(base, 'routed')
        state = torch.load(directory/'student-private.pt', map_location='cpu', weights_only=True)
        assert state['grade_parameters'] == first.grade_parameters
        model.load_state_dict(state['state_dict'], strict=True)
        assert all(torch.equal(value, first.state_dict()[key]) for key, value in model.first.state_dict().items())
        models[name] = model
    cases = validation_cases(args.base, args.lab, args.photos, args.images, record['controls'])
    old_validation = {record['validation_capture_hashes']['color']} | {r['capture_hashes']['color'] for r in record['extra_validation']}
    assert {r[2]['color'] for r in cases} == old_validation
    source_rows = {r['id']: r for r in sources['cases']}
    assert len(source_rows) == 240 and {r['id'] for r in teacher['cases']} == set(source_rows)
    assert Counter(r['split'] for r in teacher['cases']) == Counter({'train': 200, 'development': 40})
    expected_training = {r['color'] for r in record['training_capture_hashes']}
    for row in teacher['cases']:
        original = source_rows[row['id']]
        assert row['split'] == original['split'] and row['source_sha256'] == original['source_sha256']
        assert row['texture_sha256'] == original['texture_sha256'] and row['four_fenced_frames_rehashed']
        assert row['controls'] == record['controls'] and Path(row['label']).name == row['label']
        if row['split'] == 'development':
            cases.append((row['id'], args.base/(row['label']+'-state')/'capture', row['capture_hashes'], 'new-photos'))
        else:
            expected_training.add(row['capture_hashes']['color'])
    training = {r['capture_hashes']['color'] for r in fitted['training_pairs']}
    validation = {r[2]['color'] for r in cases}
    assert len(training) == 262 and training == expected_training
    assert len(cases) == len(validation) == 56 and not training & validation
    assert set(args.save_case) <= {r[0] for r in cases} and len(args.save_case) <= 8
    saved = {record['validation_capture_hashes']['color']: args.first/'validation-output.npy'}
    saved.update({r['capture_hashes']['color']: args.first/f'extra-validation-{i}.npy'
                  for i, r in enumerate(record['extra_validation'])})
    old_scores = {r['capture_hashes']['color']: r for r in previous_eval['validation']}
    assert set(old_scores) == old_validation
    models = {name: model.cuda().half().eval().to(memory_format=torch.channels_last) for name, model in models.items()}
    kernel = FusedNorm()
    models['first'].fused_backend = kernel
    args.output.mkdir(parents=True)
    rows, timing_source = [], None
    with torch.inference_mode():
        for label, path, expected_hashes, group in cases:
            controls, arrays, hashes = read_capture(path)
            assert controls == record['controls'] and hashes == expected_hashes
            source = torch.from_numpy(arrays['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
            target = torch.from_numpy(arrays['output']).permute(2, 0, 1).unsqueeze(0).cuda().float()
            if timing_source is None:
                timing_source = source.clone()
            configure(models['first'], 'baseline', kernel)
            for name in ('previous', 'broad'):
                configure_shared(models[name], 'baseline', kernel)
            references = {name: model(source) for name, model in models.items()}
            assert all(bool(torch.isfinite(output).all()) for output in references.values())
            old_case = hashes['color'] in saved
            if old_case:
                output = references['first'][0].permute(1, 2, 0).float().cpu().numpy()
                assert np.array_equal(output, np.load(saved[hashes['color']], allow_pickle=False))
            configure(models['first'], 'all-conditioning', kernel)
            for name in ('previous', 'broad'):
                configure_shared(models[name], 'all-conditioning', kernel)
            assert all(same(model(source), references[name]) for name, model in models.items())
            scores = {name: metrics(output, target) for name, output in references.items()}
            if old_case:
                prior = old_scores[hashes['color']]
                assert prior['capture_hashes'] == hashes
                for name, old_name in [('first', 'first'), ('previous', 'refined')]:
                    assert all(abs(value-prior['models'][old_name][metric]) <= 1e-7 for metric, value in scores[name].items())
            if label in args.save_case:
                for name, output in references.items():
                    np.save(args.output/f'{label}-{name}.npy', output[0].permute(1, 2, 0).float().cpu().numpy())
                for role in ('color', 'output'):
                    np.save(args.output/f'{label}-{role}.npy', arrays[role])
            rows.append({'label': label, 'group': group, 'capture_hashes': hashes,
                         'existing_fusions_match_unfused_bitwise': True,
                         'old_first_saved_output_exact': True if old_case else None,
                         'old_metrics_reproduced_within_1e_7': True if old_case else None, 'models': scores})
            print(json.dumps({'evaluated': len(rows), 'label': label,
                              'mae': {name: score['mae'] for name, score in scores.items()}}), flush=True)
        timing = paired_timing(models, timing_source)
    summary = summarize(rows, models)
    result = {'complete': True, 'target_achieved': False, 'quality_gate_passed': False,
              'variant': 'region-context-broad-warm-start', 'region_context': architecture('routed'),
              'native_shape': [1080, 1920], 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'first_checkpoint_sha256': sha(args.first/'student-private.pt'),
              'previous_checkpoint_sha256': sha(args.previous/'student-private.pt'),
              'candidate_checkpoint_sha256': sha(args.candidate/'student-private.pt'),
              'candidate_result_sha256': sha(args.candidate/'result.json'),
              'source_manifest_sha256': sha(args.sources), 'teacher_manifest_sha256': sha(args.teacher),
              'frozen_first_state_exact': True, 'training_count': 262, 'development_count': 56,
              'development_training_overlap': False, 'final_gameplay_test_count': 0,
              'validation': rows, 'summary': summary, 'timing': timing, 'saved_cases': args.save_case,
              'timing_scope': 'Complete FP16 model at 1920x1080, both stages including final grade. Existing fusions unchanged; no inference feature cache.',
              'limitations': ['Development comparisons only; no independent final gameplay or temporal validation.',
                              'Additional warm-start training is not an equal-training-compute architecture comparison.',
                              'Standalone CUDA graph excludes application conversion, synchronization and composition.',
                              'p95 is over ten-replay interval averages, not individual-frame tails.',
                              'Pixel error does not establish perceptual equivalence. No model has been installed.']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summary': summary, 'timing': timing['summary']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
