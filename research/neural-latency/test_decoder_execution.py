# SPDX-License-Identifier: Apache-2.0
"""Test optional decoder reordering/fusion against complete unchanged students.

Projection at the smaller feature extent is mathematically equivalent, but
backend rounding must be checked. The path stays disabled by default. This
does not modify the native renderer or validate student perceptual quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_demo_photo_training import audited_photos
from compare_output_grade import read_capture
from fused_norm import FusedNorm
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from test_output_grade import graph_measure


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


MODES = ('baseline', 'reordered', 'fused', 'fused-keep-last-projection')


def configure(model, mode, kernel):
    model.network.reorder_decoder = mode != 'baseline'
    model.network.decoder_reorder_stages = (1, 2) if mode == 'fused-keep-last-projection' else (0, 1, 2)
    model.network.fused_decoder_backend = kernel if mode.startswith('fused') else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--model', nargs=2, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output.exists():
        raise ValueError('Use a fresh private output file.')
    if not 1 <= len(args.model) <= 4 or len({n for n, _ in args.model}) != len(args.model):
        raise ValueError('Use one to four uniquely named models.')
    torch.manual_seed(53017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    kernel = FusedNorm()
    tests, guards, timings, results, information, projection_checks = [], [], {}, [], {}, {}
    with torch.inference_mode():
        for shape in [(1, 7, 3, 5), (2, 16, 5, 7), (1, 64, 34, 60), (1, 32, 68, 120), (1, 16, 136, 240)]:
            b, c, h, w = shape
            for layout in ('contiguous', 'channels-last', 'strided'):
                x = torch.randn((b, c, h, w), device='cuda', dtype=torch.float16) * 8
                skip = torch.randn((b, c, h*2, w*2), device='cuda', dtype=torch.float16) * 8
                if layout == 'channels-last':
                    x, skip = (v.contiguous(memory_format=torch.channels_last) for v in (x, skip))
                if layout == 'strided':
                    x = torch.stack((x, x), dim=-1)[..., 0]
                    skip = torch.stack((skip, skip), dim=-1)[..., 1]
                expected = F.interpolate(x, scale_factor=2, mode='nearest') + skip
                actual = kernel.decoder_upscale_add(x, skip)
                exact = bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
                assert exact
                tests.append({'shape': list(shape), 'layout': layout, 'bit_equal': exact})
        x = torch.zeros((1, 16, 4, 6), device='cuda', dtype=torch.float16)
        skip = torch.zeros((1, 16, 8, 12), device='cuda', dtype=torch.float16)
        invalid = [('dtype', x.float(), skip), ('spatial-ratio', x, skip[..., :-1]),
                   ('channels', x[:, :8], skip), ('rank', x[0], skip),
                   ('empty', x[:, :0], skip[:, :0]), ('cpu', x.cpu(), skip.cpu())]
        for name, a, b in invalid:
            try:
                kernel.decoder_upscale_add(a, b)
            except ValueError:
                guards.append(name)
            else:
                raise AssertionError('Invalid decoder input accepted: ' + name)
        # A real decoder stage: include projection, resize and addition in timing.
        stage = torch.nn.Conv2d(96, 64, 1).cuda().half().eval()
        x = torch.randn((1, 96, 34, 60), device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        skip = torch.randn((1, 64, 68, 120), device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        stage_functions = {
            'baseline': lambda value: stage(F.interpolate(value, scale_factor=2, mode='nearest')) + skip,
            'reordered': lambda value: F.interpolate(stage(value), scale_factor=2, mode='nearest') + skip,
            'fused': lambda value: kernel.decoder_upscale_add(stage(value), skip)}
        stage_expected = stage_functions['baseline'](x)
        for name, function in stage_functions.items():
            output = function(x)
            timing, graph = graph_measure(function, x)
            assert torch.equal(graph, output)
            timings['stage-' + name] = {**timing, 'graph_matches_eager': True,
                'matches_baseline': bool(torch.equal(output, stage_expected)),
                'max_abs_difference': float((output-stage_expected).abs().max())}

        models = {}
        for name, folder in args.model:
            directory = Path(folder)
            record = read(directory / 'result.json')
            architecture = record['architecture']
            assert architecture['variant'] == 'hierarchical' and architecture['noise_channels'] == 0
            assert architecture['width'] in (16, 32) and architecture['blocks'] == 2
            checkpoint = torch.load(directory / 'student-private.pt', map_location='cpu', weights_only=True)
            assert checkpoint['architecture'] == architecture
            network = HierarchicalStudent(architecture['width'], 2)
            model = GradedStudent(network, architecture['explicit_output_grading'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model = model.cuda().half().eval().to(memory_format=torch.channels_last)
            model.fused_backend = kernel
            models[name] = model
            information[name] = {'architecture': architecture, 'result_sha256': sha(directory/'result.json'),
                                 'checkpoint_sha256': sha(directory/'student-private.pt')}
            if name == args.model[0][0]:
                controls_expected = record['controls']
            assert record['controls'] == controls_expected
        illumination = read(args.base / 'teacher-illumination-manifest.json')
        labels = ['model-capture-natural-north', 'teacher-natural-shifted-north']
        labels += [r['label'] for r in illumination['views'] if r['split'] == 'validation']
        cases = [(label, args.base/'trials'/label/'capture') for label in labels]
        cases += [(r['label'], args.base/(r['label']+'-state')/'capture')
                  for r in audited_photos(args.photos, args.base, controls_expected) if r['split']=='validation']
        assert len(cases) == 12
        for label, path in cases:
            controls, images, hashes = read_capture(path)
            assert controls == controls_expected
            source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
            row = {'label': label, 'capture_hashes': hashes, 'models': {}}
            for name, model in models.items():
                configure(model, 'baseline', kernel)
                snapshots, handles = {}, []
                if name not in projection_checks:
                    def capture_stage(index):
                        def hook(module, inputs, output):
                            snapshots[index] = (inputs[0].clone(), output.clone())
                        return hook
                    handles = [layer.register_forward_hook(capture_stage(index))
                               for index, layer in enumerate(model.network.up)]
                expected = model(source)
                for handle in handles:
                    handle.remove()
                if snapshots:
                    projection_checks[name] = []
                    for index, (large_input, original_projection) in sorted(snapshots.items()):
                        small_input = large_input[:, :, ::2, ::2].contiguous(memory_format=torch.channels_last)
                        assert torch.equal(F.interpolate(small_input, scale_factor=2, mode='nearest'), large_input)
                        projected = model.network.up[index](small_input)
                        alternative = F.interpolate(projected, scale_factor=2, mode='nearest')
                        difference = alternative.float()-original_projection.float()
                        projection_checks[name].append({'stage': index, 'small_input_shape': list(small_input.shape),
                            'projection_output_channels': projected.shape[1],
                            'same_preprojection_values': True,
                            'bit_equal': bool(torch.equal(alternative.view(torch.int16), original_projection.view(torch.int16))),
                            'max_abs_difference': float(difference.abs().max()),
                            'changed_fraction': float((difference != 0).float().mean())})
                    snapshots.clear()
                outputs = {'baseline': expected}
                for mode in MODES[1:]:
                    configure(model, mode, kernel)
                    outputs[mode] = model(source)
                row['models'][name] = {mode: {
                    'bit_equal': bool(torch.equal(output.view(torch.int16), expected.view(torch.int16))),
                    'max_abs_difference': float((output-expected).abs().max()),
                    'mae_difference': float((output.float()-expected.float()).abs().mean())}
                    for mode, output in outputs.items() if mode != 'baseline'}
                row['models'][name]['fused']['matches_reordered_bitwise'] = bool(
                    torch.equal(outputs['fused'].view(torch.int16), outputs['reordered'].view(torch.int16)))
                assert row['models'][name]['fused']['matches_reordered_bitwise']
                if name not in timings:
                    timings[name] = {}
                    for mode in MODES:
                        configure(model, mode, kernel)
                        timing, graph = graph_measure(model, source)
                        assert torch.equal(graph, outputs[mode])
                        timings[name][mode] = {**timing, 'graph_matches_eager': True}
                configure(model, 'baseline', kernel)
            results.append(row)
            print(json.dumps({'case': label, 'models': row['models']}), flush=True)
    report = {'schema': 1, 'target_achieved': False, 'quality_gate_passed': False,
              'native_runtime_accelerated': False, 'kernel_tests': tests, 'rejected_inputs': guards,
              'models': information, 'cases': results, 'timings': timings,
              'projection_checks_on_first_image': projection_checks,
              'decoder_enabled_by_default': False,
              'limitations': ['Only student execution is tested; these students do not preserve native image quality.',
                  'Projection reordering can alter backend rounding; all measured discrepancies are retained.',
                  'Complete 1080p student graph and isolated decoder stage are separately reported; neither is a native app speedup.',
                  'No temporal or game acceptance, no native runtime change or replacement installation.']}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'kernel_cases': len(tests), 'guards': guards,
        'all_full_outputs_bit_equal': all(s['bit_equal'] for r in results for m in r['models'].values() for s in m.values()),
        'complete_graph_ms': {n: {k: v['median_ms'] for k, v in timings[n].items()} for n in models}}, indent=2))


if __name__ == '__main__':
    main()
