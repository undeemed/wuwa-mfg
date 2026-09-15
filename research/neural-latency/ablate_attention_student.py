# SPDX-License-Identifier: Apache-2.0
"""Measure a fitted attention branch without changing its surrounding weights.

Private checkpoints and matched captures are required. The baseline is the
previous independently trained hierarchical student, not native intermediate
features. This is an offline diagnostic, not application integration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fused_norm import FusedNorm
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from test_output_grade import graph_measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--case', nargs=3, action='append', required=True,
                        metavar=('LABEL', 'CAPTURE', 'SAVED_PREDICTION_NAME'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = {name: json.loads((path / 'result.json').read_text())
               for name, path in [('baseline', args.baseline), ('attention', args.student)]}
    for key in ('controls', 'native_shape', 'training_capture_hashes',
                'validation_capture_hashes', 'data_split', 'optimization', 'completed_steps'):
        assert records['baseline'][key] == records['attention'][key], key
    assert records['baseline']['extra_validation'][0]['capture_hashes'] == records['attention']['extra_validation'][0]['capture_hashes']
    assert records['baseline']['architecture']['variant'] == 'hierarchical'
    assert records['attention']['architecture']['variant'] == 'hierarchical-attention'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    backend = FusedNorm()

    def load_model(directory):
        checkpoint = torch.load(directory / 'student-private.pt', map_location='cpu', weights_only=True)
        architecture = checkpoint['architecture']
        network = HierarchicalStudent(architecture['width'], architecture['blocks'],
                                      attention=architecture['variant'] == 'hierarchical-attention')
        model = GradedStudent(network, architecture['explicit_output_grading'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model = model.cuda().half().eval().to(memory_format=torch.channels_last)
        model.fused_backend = backend
        return model

    baseline = load_model(args.baseline)
    candidate = load_model(args.student)
    branch = candidate.network.global_attention
    cases = []
    timing = {}
    labels = set()
    with torch.inference_mode():
        for label, capture_name, prediction_name in args.case:
            if label in labels or Path(prediction_name).name != prediction_name:
                raise ValueError('Use unique labels and a saved prediction basename.')
            labels.add(label)
            capture = Path(capture_name)
            metadata = json.loads((capture / 'frame-0.json').read_text())
            assert metadata['complete'] and metadata['gpu_completed'] and metadata['evaluate_result'] == 1
            assert metadata['controls'] == records['attention']['controls']
            images, hashes = {}, {}
            for role in ('color', 'output'):
                resource = metadata['resources'][role]
                assert resource['format'] == 10 and resource['row_bytes'] == resource['width'] * 8
                assert Path(resource['file']).name == resource['file']
                raw = (capture / resource['file']).read_bytes()
                assert len(raw) == resource['height'] * resource['row_bytes']
                image = np.frombuffer(raw, dtype='<f2').reshape(resource['height'], resource['width'], 4)[..., :3].copy()
                assert np.isfinite(image).all()
                images[role] = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
                hashes[role] = hashlib.sha256(raw).hexdigest()
            source = images['color'].contiguous(memory_format=torch.channels_last)
            target = images['output'].float()

            def metrics(output):
                delta = output.float() - target
                absolute = delta.abs()
                detail = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
                detail = detail + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
                return {'mae': float(absolute.mean()), 'rmse': float(delta.square().mean().sqrt()),
                        'p99': float(torch.quantile(absolute.flatten(), .99)),
                        'gradient_error': float(detail)}

            baseline_output, complete_output = baseline(source), candidate(source)
            reproduced = {}
            for name, directory, output in [('baseline', args.baseline, baseline_output),
                                            ('attention', args.student, complete_output)]:
                saved = np.load(directory / prediction_name, allow_pickle=False)
                produced = output[0].permute(1, 2, 0).float().cpu().numpy()
                reproduced[name] = bool(np.array_equal(saved, produced))
                assert reproduced[name], f'{label}: saved {name} output differs.'
            candidate.network.global_attention = None
            try:
                disabled_output = candidate(source)
            finally:
                candidate.network.global_attention = branch
            cases.append({'label': label, 'capture_hashes': hashes, 'saved_outputs_exact': reproduced,
                          'baseline': metrics(baseline_output), 'attention_enabled': metrics(complete_output),
                          'attention_disabled_same_weights': metrics(disabled_output),
                          'branch_rgb_change_mae': float((complete_output - disabled_output).abs().float().mean())})
            if not timing:
                for name, model, expected in [('baseline', baseline, baseline_output),
                                              ('attention_enabled', candidate, complete_output)]:
                    timing[name], graph_output = graph_measure(model, source)
                    timing[name]['graph_matches_eager'] = bool(torch.equal(graph_output, expected))
                    assert timing[name]['graph_matches_eager']
                candidate.network.global_attention = None
                try:
                    timing['attention_disabled_same_weights'], graph_output = graph_measure(candidate, source)
                    timing['attention_disabled_same_weights']['graph_matches_eager'] = bool(torch.equal(graph_output, disabled_output))
                    assert timing['attention_disabled_same_weights']['graph_matches_eager']
                finally:
                    candidate.network.global_attention = branch
    report = {'schema': 1, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'quality_gate_passed': False, 'target_achieved': False,
              'cases': cases, 'complete_graph_timings': timing,
              'timing_scope': 'FP16 full-1080p network plus fused output grade; excludes application integration.',
              'limitations': ['Disabling the branch is an ablation, not a separately retrained model.',
                              'Repeatedly consulted validation views from one scene; not an independent final test.',
                              'First-reset images only; no temporal or perceptual acceptance established.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cases': cases, 'median_ms': {key: value['median_ms'] for key, value in timing.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
