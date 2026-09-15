# SPDX-License-Identifier: Apache-2.0
"""Check fused student output rounding and complete-image/graph equivalence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from collect_demo_photo_training import audited_photos
from compare_output_grade import read_capture
from fused_norm import FusedNorm
from output_grade import GradedStudent, grade_torch
from student_probe import HierarchicalStudent, ResidualBlock
from test_output_grade import graph_measure


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def configure(model, mode, kernel):
    conditioning=mode in ('conditioning','all-conditioning')
    mode='all' if mode=='all-conditioning' else 'baseline' if mode=='conditioning' else mode
    model.network.fused_conditioning_backend=kernel if conditioning else None
    model.fused_student_output = mode in ('output', 'combined', 'all')
    model.network.reorder_decoder = mode in ('decoder', 'combined', 'all')
    model.network.decoder_reorder_stages = (1, 2) if model.network.head.in_channels == 32 else (0, 1, 2)
    model.network.fused_decoder_backend = kernel if model.network.reorder_decoder else None
    for block in model.network.modules():
        if isinstance(block,ResidualBlock):
            block.fused_residual_backend=kernel if mode in ('residual','all') else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--image-collection', type=Path, help='Optional diverse image extension manifest.')
    parser.add_argument('--residual-fusion', action='store_true', help='Also check residual-only and all-fusion modes.')
    parser.add_argument('--conditioning-fusion',action='store_true',help='Also check conditioning-only and all-plus-conditioning modes.')
    parser.add_argument('--model', nargs=2, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output file.')
    if len(args.model)>4 or len({n for n, _ in args.model}) != len(args.model):
        raise ValueError('Use up to four uniquely named models.')
    torch.manual_seed(95431)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = FusedNorm()
    modes=['baseline','output','decoder','combined']+(['residual','all'] if args.residual_fusion else [])
    if args.conditioning_fusion:
        if not args.residual_fusion:raise ValueError('Conditioning comparison requires the previous all-fusion baseline.')
        modes+=['conditioning','all-conditioning']
    conditioning_tests=None
    if args.conditioning_fusion:
        from test_conditioning_fusion import operator_tests as conditioning_operator_tests
        conditioning_tests=conditioning_operator_tests(kernel)
    residual_tests=None
    if args.residual_fusion:
        from test_student_residual import operator_tests
        residual_tests=operator_tests(kernel)
    controls = (.9330329895019531, -.25, .8999999761581421)
    tests, rejected, rows, timings, model_info = [], [], [], {}, {}
    with torch.inference_mode():
        def reference(source, head, parameters=controls):
            height, width = source.shape[-2:]
            residual = F.pixel_shuffle(head, 4)[:, :, :height, :width]
            return kernel.output_grade((source+.25*residual).clamp(0, 1), parameters)

        def check(label, source, head, parameters=controls):
            expected, actual = reference(source, head, parameters), kernel.student_output(source, head, parameters)
            assert same(actual, expected), label
            tests.append({'label': label, 'shape': list(source.shape), 'head_shape': list(head.shape),
                          'source_stride': list(source.stride()), 'head_stride': list(head.stride()),
                          'parameters': list(parameters), 'bit_equal': True})

        for batch, height, width in [(1, 19, 37), (2, 32, 64), (1, 1080, 1920)]:
            source = torch.randn((batch, 3, height, width), device='cuda', dtype=torch.float16)*.7+.5
            head = torch.randn((batch, 48, (height+31)//32*8, (width+31)//32*8), device='cuda', dtype=torch.float16)*4
            for layout in ['planar', 'channels-last', 'strided']:
                a, b = source, head
                if layout == 'channels-last':
                    a, b = (x.contiguous(memory_format=torch.channels_last) for x in (a, b))
                elif layout == 'strided':
                    a, b = (torch.stack((x, x), dim=-1)[..., 1] for x in (a, b))
                check(layout, a, b)
        for parameters in [(1., 0., 1.), (2., 1., 0.), (.25, -1., .5)]:
            check('control-boundary', source, head, parameters)
        values = np.arange(65536, dtype=np.uint16).view(np.float16)
        values = torch.from_numpy(values[np.isfinite(values)].copy()).cuda()
        source = torch.stack((values, values.roll(1), values.roll(173))).reshape(1, 3, 1, -1)
        head_shape = (1, 48, 8, (source.shape[-1]+31)//32*8)
        head = values.repeat((int(np.prod(head_shape))+len(values)-1)//len(values))[:int(np.prod(head_shape))].reshape(head_shape)
        check('all-finite-half-patterns', source, head)
        check('broadcast-source-and-head', source[:, :1].expand_as(source), head[:, :1].expand_as(head))
        a = torch.zeros((1, 3, 32, 32), device='cuda', dtype=torch.float16)
        b = torch.zeros((1, 48, 8, 8), device='cuda', dtype=torch.float16)
        invalid = [('dtype', a.float(), b, controls), ('rgb', a[:, :2], b, controls),
                   ('head-shape', a, b[:, :32], controls), ('rank', a[0], b, controls),
                   ('cpu', a.cpu(), b.cpu(), controls), ('empty-batch', a[:0], b[:0], controls),
                   ('nonfinite-control', a, b, (float('nan'), 0., 1.)), ('control-range', a, b, (1., 0., 2.))]
        for label, x, y, parameters in invalid:
            try:
                kernel.student_output(x, y, parameters)
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError('Invalid input accepted: '+label)
        models = {}
        for name, folder in args.model:
            folder = Path(folder)
            record = read(folder/'result.json'); architecture = record['architecture']
            assert architecture['variant'] in ('hierarchical','hierarchical-film') and architecture['width'] in (16, 32)
            assert architecture['blocks']==2 and architecture['noise_channels']==0
            checkpoint = torch.load(folder/'student-private.pt', map_location='cpu', weights_only=True)
            assert checkpoint['architecture']==architecture
            model = GradedStudent(HierarchicalStudent(architecture['width'], 2, conditioned=architecture['variant']=='hierarchical-film'), architecture['explicit_output_grading'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model = model.cuda().half().eval().to(memory_format=torch.channels_last)
            model.fused_backend = kernel
            models[name] = model
            model_info[name] = {'architecture': architecture, 'result_sha256': sha(folder/'result.json'),
                                'checkpoint_sha256': sha(folder/'student-private.pt')}
            if len(models)==1:
                expected_controls = record['controls']
            assert record['controls']==expected_controls
        if models:
            collection = read(args.base/'teacher-illumination-manifest.json')
            labels = ['model-capture-natural-north', 'teacher-natural-shifted-north']
            labels += [r['label'] for r in collection['views'] if r['split']=='validation']
            cases = [(label, args.base/'trials'/label/'capture') for label in labels]
            cases += [(r['label'], args.base/(r['label']+'-state')/'capture')
                      for r in audited_photos(args.photos, args.base, expected_controls) if r['split']=='validation']
            assert len(cases)==12
            if args.image_collection:
                from collect_diverse_images import audited_images
                cases += [(r['label'], args.base/(r['label']+'-state')/'capture')
                          for r in audited_images(args.image_collection, args.base, expected_controls, args.photos)
                          if r['split']=='validation']
                assert len(cases)==16
            for label, path in cases:
                control_values, images, hashes = read_capture(path)
                assert control_values==expected_controls
                source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
                row = {'label': label, 'capture_hashes': hashes, 'models': {}}
                for name, model in models.items():
                    outputs = {}
                    for mode in modes:
                        configure(model, mode, kernel)
                        outputs[mode] = model(source)
                        assert same(outputs[mode], outputs['baseline']), (name, label, mode)
                        if name not in timings or mode not in timings[name]:
                            timing, graph = graph_measure(model, source)
                            assert same(graph, outputs[mode])
                            timings.setdefault(name, {})[mode] = {**timing, 'graph_matches_eager_bitwise': True}
                    row['models'][name] = {'all_four_modes_bit_equal': True, 'all_checked_modes_bit_equal': True}
                    configure(model, 'baseline', kernel)
                rows.append(row)
    report = {'schema': 1, 'complete': True, 'target_achieved': False, 'quality_gate_passed': False,
              'native_runtime_accelerated': False, 'enabled_by_default': False,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'kernel_tests': tests, 'rejected_inputs': rejected, 'models': model_info,
              'checked_modes':modes, 'residual_tests':residual_tests,
              'conditioning_tests':conditioning_tests,
              'cases': rows, 'timings': timings,
              'limitations': ['Finite-input equivalence to existing student arithmetic, not native quality.',
                  'Complete student graph timings exclude D3D12 integration and do not accelerate native NVIDIA execution.',
                  'Exact full-image results cover only the checked checkpoints, inputs and environment.']}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'tests': len(tests), 'guards': rejected, 'complete_images': len(rows),
                      'graph_medians': {n: {m: t['median_ms'] for m, t in modes.items()} for n, modes in timings.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
