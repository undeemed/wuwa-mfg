# SPDX-License-Identifier: Apache-2.0
"""Inspect training-only correction size, spatial variation and learned features.

Only scalar summaries are written. No activations, weights or pixel arrays are
exported. These diagnostics are evidence of behavior, not a causal proof.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from collect_training_brightness import audited_brightness
from progressive_student import FrozenStudentRefinement, load_first
from student_training_pairs import read, sha, training_pairs, load_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'photos', 'images', 'brightness', 'baseline', 'first', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shared-features', action='store_true', help='Inspect a shared-feature correction decoder, including a spatial-feature counterfactual.')
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first, record = load_first(args.first)
    fitted = read(args.candidate / 'result.json')
    assert fitted['first_checkpoint_sha256'] == sha(args.first / 'student-private.pt')
    pairs = training_pairs(args.base, args.photos, args.images, read(args.baseline / 'result.json'))
    for row in audited_brightness(args.brightness, args.base, record['controls'], args.photos, args.images):
        pairs.append({'label': row['label'], 'group': 'bright-photos',
                      'path': args.base / (row['label'] + '-state') / 'capture', 'capture_hashes': row['capture_hashes']})
    assert len(pairs) == 62 and [row['capture_hashes'] for row in pairs] == fitted['training_capture_hashes']
    validation = {record['validation_capture_hashes']['color']} | {row['capture_hashes']['color'] for row in record['extra_validation']}
    assert not validation & {row['capture_hashes']['color'] for row in pairs}
    torch.manual_seed(28411)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.shared_features:
        from shared_feature_student import SharedFeatureRefinement
        assert fitted['variant'] == 'shared-features'
        model = SharedFeatureRefinement(first)
        prefix, input_layer = '', 'local'
    else:
        assert fitted.get('variant', 'rgb-cascade') == 'rgb-cascade'
        model = FrozenStudentRefinement(first)
        prefix, input_layer = 'network.', 'stem'
    initial = {name: value.clone() for name, value in model.refinement.state_dict().items()}
    state = torch.load(args.candidate / 'student-private.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'], strict=True)
    changes = {name: float((value - initial[name]).square().mean().sqrt())
               for name, value in model.refinement.state_dict().items()}
    parameter_changes = {'changed_tensor_count': sum(value > 0 for value in changes.values()),
                         'total_tensor_count': len(changes),
                         input_layer+'_weight_rms_change': changes[prefix+input_layer+'.weight'],
                         'head_weight_rms_change': changes[prefix+'head.weight'],
                         'head_bias_rms_change': changes[prefix+'head.bias']}
    model = model.cuda().float().eval().to(memory_format=torch.channels_last)
    captured = {}
    def before_head(module, inputs):
        captured['features'] = inputs[0].detach()
    def after_head(module, inputs, output):
        captured['head'] = output.detach()
    head_layer = model.refinement.head if args.shared_features else model.refinement.network.head
    handles = [head_layer.register_forward_pre_hook(before_head), head_layer.register_forward_hook(after_head)]
    rows, head_means = [], []
    with torch.inference_mode():
        for pair, expected in zip(pairs, fitted['training_image_metrics']):
            source, target = load_pair(pair, record['controls'])
            base = model.first(source)
            refined = model(source)
            assert float((refined - target).abs().mean()) == expected['refined_mae']
            desired, correction = target - base, refined - base
            head = captured['head']
            features = captured['features']
            means = head.mean(dim=(0, 2, 3))
            head_means.append(means.cpu())
            centered = head - means[None, :, None, None]
            feature_std = features.std(dim=(0, 2, 3), unbiased=False)
            denom = correction.square().sum().sqrt() * desired.square().sum().sqrt()
            row = {'label': pair['label'], 'group': pair['group'], 'capture_hashes': pair['capture_hashes'],
                   'first_mae': float(desired.abs().mean()), 'refined_mae': expected['refined_mae'],
                   'adjustment_mae': float(correction.abs().mean()),
                   'adjustment_to_error_mae_ratio': float(correction.abs().mean() / desired.abs().mean()),
                   'adjustment_desired_cosine': float((correction * desired).sum() / denom.clamp_min(1e-30)),
                   'head_rms': float(head.square().mean().sqrt()),
                   'head_spatial_rms': float(centered.square().mean().sqrt()),
                   'head_spatial_energy_fraction': float(centered.square().mean() / head.square().mean().clamp_min(1e-30)),
                   'head_phase_mean_std': float(means.reshape(3, 16).std(dim=1, unbiased=False).mean()),
                   'head_feature_spatial_std_mean': float(feature_std.mean()),
                   'head_feature_nearly_constant_count': int((feature_std < 1e-7).sum())}
            if args.shared_features:
                ungraded, features = model.extract(source)
                flipped = tuple(torch.flip(feature, dims=(-1,)) for feature in features)
                counterfactual = model.refinement(source, ungraded, flipped)
                row['flipped_features_mae'] = float((counterfactual-target).abs().mean())
                row['flipped_feature_output_difference_mae'] = float((counterfactual-refined).abs().mean())
            rows.append(row)
    for handle in handles:
        handle.remove()
    # The RGB output head maps each of its 48 channels to a 4x4 pixel-shuffle
    # phase. Its channel means stay private; only their scalar dispersion is kept.
    means = torch.stack(head_means)
    summary = {}
    fields = [key for key, value in rows[0].items() if isinstance(value, (float, int))]
    for group in ['all', 'scene', 'photos', 'bright-photos']:
        selected = [row for row in rows if group == 'all' or row['group'] == group]
        summary[group] = {'count': len(selected), **{key: statistics.mean(row[key] for row in selected) for key in fields}}
    report = {'complete': True, 'target_achieved': False, 'quality_gate_passed': False,
              'variant': 'shared-features' if args.shared_features else 'rgb-cascade',
              'scope': 'FP32 training-only evaluation; no optimizer step, validation inference or parameter change.',
              'candidate_checkpoint_sha256': sha(args.candidate / 'student-private.pt'),
              'validation_count_excluded': len(validation), 'training_count': len(rows),
              'all_training_predictions_reproduced_exactly': True, 'parameter_changes': parameter_changes,
              'across_images_head_mean_std': float(means.std(dim=0, unbiased=False).mean()),
              'within_image_mean_head_phase_std': statistics.mean(row['head_phase_mean_std'] for row in rows),
              'summary': summary, 'training': rows,
              'limitations': ['Scalar variation and alignment describe the correction but do not prove why training converged there.',
                              'For shared-feature models, horizontal feature flips preserve the original RGB input and base output but intentionally create inconsistent internal features. This is a sensitivity diagnostic, not a quality acceptance input.',
                              'The head includes 4x4 pixel-shuffle phases; per-channel mean offsets need not be a spatially uniform RGB correction.',
                              'No claim that a small correction establishes an irreducible quality limit.']}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ['parameter_changes', 'across_images_head_mean_std', 'within_image_mean_head_phase_std', 'summary']}, indent=2))


if __name__ == '__main__':
    main()
