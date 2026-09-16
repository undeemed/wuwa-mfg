# SPDX-License-Identifier: MIT
"""Summarize whitelisted native-launch metadata; never publish packed arguments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def compare_captures(reference, candidate):
    """Require matching prior frames before treating a temporal output as paired."""
    def scalars(trial):
        rows = [json.loads(line) for line in (trial / 'nr-launch-contract.jsonl').read_text().splitlines()]
        return {r['frame'] - 1: r['preprocessor_scalar_candidate'] for r in rows
                if 'preprocessor_scalar_candidate' in r}
    left_scalars, right_scalars = scalars(reference), scalars(candidate)
    prior_inputs_match = False
    frames = []
    for index in range(4):
        manifests = []
        hashes = []
        formats = []
        for trial in (reference, candidate):
            folder = trial / 'capture'
            path = folder / f'frame-{index}.json'
            if not path.exists():
                return {'reference': reference.name, 'candidate': candidate.name,
                        'comparison_available': False, 'reason': 'Four captures required'}
            manifest = json.loads(path.read_text())
            assert manifest['complete'] and manifest['gpu_completed'] and manifest['evaluate_result'] == 1
            resources = manifest['resources']
            assert set(resources) == {'color', 'depth', 'motion', 'output'}
            texture_hashes, texture_formats = {}, {}
            for role, desc in resources.items():
                assert Path(desc['file']).name == desc['file']
                payload = (folder / desc['file']).read_bytes()
                assert len(payload) == desc['row_bytes'] * desc['rows']
                texture_hashes[role] = hashlib.sha256(payload).hexdigest()
                texture_formats[role] = {k: v for k, v in desc.items() if k != 'file'}
            manifests.append(manifest)
            hashes.append(texture_hashes)
            formats.append(texture_formats)
        controls_match = manifests[0]['controls'] == manifests[1]['controls']
        scalars_match = left_scalars[index] == right_scalars[index]
        texture_matches = {r: hashes[0][r] == hashes[1][r] for r in hashes[0]}
        if controls_match and manifests[0]['controls']['DLSSNR.Reset']:
            prior_inputs_match = True
        inputs_match = (controls_match and scalars_match and formats[0] == formats[1]
                        and all(texture_matches[r] for r in ('color', 'depth', 'motion')))
        paired = prior_inputs_match and inputs_match
        frames.append({'index': index, 'controls_match': controls_match,
                       'recorded_scalars_match': scalars_match,
                       'texture_formats_match': formats[0] == formats[1],
                       'texture_bytes_match': texture_matches,
                       'input_and_recorded_history_match': paired,
                       'paired_output_bytes_match': texture_matches['output'] if paired else None})
        prior_inputs_match = paired
    return {'reference': reference.name, 'candidate': candidate.name,
            'comparison_available': True, 'frames': frames, 'quality_gate_passed': False,
            'limitation': 'First four frames of one static scene; later unmatched inputs are excluded.'}


def summarize(trial):
    records = [json.loads(line) for line in
               (trial / 'nr-launch-contract.jsonl').read_text().splitlines() if line.strip()]
    modules = [r for r in records if r['kind'] == 'module']
    launches = [r for r in records if r['kind'] == 'launch']
    assert launches, 'No observed launches'
    grouped = defaultdict(list)
    for row in launches:
        assert row['status'] == 0, 'A recorded launch failed'
        grouped[row['frame']].append(row)
    reference = None
    frames = []
    for frame, rows in sorted(grouped.items()):
        order = [(r['name'], r['api'], r['param_bytes'], r['grid'], r['block']) for r in rows]
        if reference is None:
            reference = order
        assert order == reference, 'Observed launch order or geometry changed'
        chains = defaultdict(list)
        for row in rows:
            chains[row['chain']].append(row)
        assert sorted(chains) == list(range(1, len(chains) + 1)), 'Incomplete chain sequence'
        times = []
        for chain_rows in chains.values():
            assert [r['kernel'] for r in chain_rows] == list(range(len(chain_rows)))
            # This duration is repeated for every kernel in a chain. Count it once.
            assert len({r['chain_cpu_us'] for r in chain_rows}) == 1
            times.append(chain_rows[0]['chain_cpu_us'])
        scalar_rows = [r for r in rows if 'preprocessor_scalar_candidate' in r]
        assert len(scalar_rows) == 1, 'Expected exactly one whitelisted preprocessor'
        scalar = scalar_rows[0]['preprocessor_scalar_candidate']
        sanity = None
        capture = trial / 'capture' / f'frame-{frame - 1}.json'
        if capture.exists():
            manifest = json.loads(capture.read_text())
            assert manifest['complete'] and manifest['gpu_completed']
            assert manifest['evaluate_result'] == 1 and manifest['index'] == frame - 1
            c = manifest['controls']
            checks = {
                'extent_matches_capture': [scalar['width'], scalar['height']] ==
                                         [c['DLSSNR.Width'], c['DLSSNR.Height']],
                'style_matches_normalized_control': scalar['style'] == c['DLSSNR.Style'] / 128,
                'tone_matches_control': scalar['tone'] == c['DLSSNR.LocalToneStrength'],
                'structure_matches_control': scalar['structure'] == c['DLSSNR.LocalStructureStrength'],
                'mask_matches_control': scalar['auto_enabled'] == c['DLSSNR.UseAutoMask'],
                'depth_direction_matches_control': scalar['depth_inverted'] == c['DLSSNR.DepthInverted'],
            }
            assert all(checks.values()), checks
            sanity = {'completed_fenced_capture': True, 'checks': checks,
                      'reset': c['DLSSNR.Reset']}
        frames.append({'evaluation': frame, 'kernels': len(rows), 'chains': len(chains),
                       'kernels_per_chain': dict(Counter(len(v) for v in chains.values())),
                       'native_call_cpu_us_sum': round(sum(times), 4),
                       'native_call_cpu_us_median': statistics.median(times),
                       'native_call_cpu_us_max': max(times),
                       'preprocessor_scalars': scalar, 'capture_sanity': sanity})
    module_keys = ['bytes', 'status', 'architecture_selection', 'fatbin_architectures',
                   'fatbin_walk_complete', 'force_sm89_requested', 'filtered_to_sm89',
                   'submitted_bytes', 'fatbin_version', 'fatbin_header_bytes',
                   'fatbin_payload_bytes', 'trailing_bytes', 'trailing_bytes_all_zero']
    successful = [r for r in modules if r['status'] == 0 and r['bytes']]
    result = json.loads((trial / 'result.json').read_text())
    return {
        'schema': 1, 'trial': trial.name,
        'successful_modules': len(successful),
        'successful_sm89_filtered_modules': sum(r.get('filtered_to_sm89', False) for r in successful),
        'modules': [{k: r[k] for k in module_keys if k in r} for r in modules],
        'apis': dict(Counter(r['api'] for r in launches)),
        'all_launches_succeeded': True,
        'all_observed_launches_use_packed_parameters': all(r['has_packed_params'] for r in launches),
        'observed_launch_order_and_geometry_consistent': True,
        'launch_order_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        'frames': frames,
        'model_1920x1080_confirmed': result['model_1920x1080_confirmed'],
        'sparse_gpu_intervals': {'retained_count': result['retained_count'],
                                 'summary': result.get('summary')},
        'limitations': [
            'CPU durations enclose only the original NVAPI call; they are not GPU execution times.',
            'Capture and metadata instrumentation are active. Sparse intervals do not prove a speedup.',
            'A zero-length module probe may fail independently of successful real modules.',
            'Scalar sanity checks validate only the listed controls, not every inferred parameter offset.',
            'Different demo launches are not guaranteed identical inputs or temporal history.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 1, 'trials': [summarize(p) for p in args.trials],
              'matched_capture_comparisons': [compare_captures(args.trials[0], p) for p in args.trials[1:]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps([{'trial': r['trial'], 'modules': r['successful_modules'],
                       'sm89_filtered': r['successful_sm89_filtered_modules'],
                       'frames': len(r['frames']), 'timing': r['sparse_gpu_intervals']}
                      for r in report['trials']], indent=2))
