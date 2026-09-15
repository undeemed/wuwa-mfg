# SPDX-License-Identifier: Apache-2.0
"""Export and independently reload the private, full-1080p routed student.

This is a PyTorch 2.7 export/parity experiment, not an NGX replacement DLL.
Artifacts include private learned weights and must stay outside the repository.
No model training, application launch or new custom kernel is performed.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_from_capture(path, expected, controls):
    from compare_output_grade import read_capture
    actual_controls, arrays, hashes = read_capture(path)
    if hashes != expected or actual_controls != controls:
        raise ValueError('Capture identity or controls changed.')
    source = torch.from_numpy(arrays['color']).permute(2, 0, 1).unsqueeze(0)
    return source.cuda().half().contiguous(memory_format=torch.channels_last)


def graph_operators(program):
    return dict(sorted(Counter(
        str(node.target) for module in program.graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes if node.op == 'call_function'
    ).items()))


def verify(directory):
    # Deliberately do not import any original model or training definitions.
    manifest = read(directory / 'manifest-private.json')
    result_path = directory / 'reload-result.json'
    if result_path.exists():
        raise ValueError('Reload result already exists; preserve the prior result.')
    artifact = directory / 'student-private.pt2'
    if manifest['artifact_sha256'] != sha(artifact):
        raise ValueError('Exported artifact changed.')
    if manifest['torch_version'] != torch.__version__:
        raise ValueError('Use the export experiment\'s exact PyTorch version.')
    program = torch.export.load(artifact)
    model = program.module()
    if graph_operators(program) != manifest['operators']:
        raise ValueError('Reloaded graph operators changed.')
    rows = []
    with torch.inference_mode():
        for row in manifest['cases']:
            source = source_from_capture(Path(row['capture']), row['capture_hashes'], manifest['controls'])
            output = model(source)
            digest = tensor_sha(output)
            if digest != row['output_sha256'] or output.shape != source.shape or not torch.isfinite(output).all():
                raise ValueError('Reloaded output differs from the original candidate: ' + row['label'])
            rows.append({'label': row['label'], 'bitwise_equal': True})
            print(json.dumps(rows[-1]), flush=True)
    forbidden = ('region_context_student', 'shared_feature_student', 'student_probe', 'progressive_student')
    if any(name in sys.modules for name in forbidden):
        raise ValueError('The independent loader unexpectedly imported model definitions.')
    result = {'complete': True, 'artifact_sha256': sha(artifact),
              'torch_version': torch.__version__, 'native_shape': [1080, 1920],
              'all_outputs_bitwise_equal': True, 'validation_count': len(rows),
              'original_model_definitions_imported': False, 'cases': rows,
              'scope': 'Fresh-process PyTorch reload only; no native runtime, temporal or game validation.'}
    write(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)


def export(args):
    from evaluate_progressive_student import paired_timing, validation_cases
    from fused_norm import FusedNorm
    from progressive_student import load_first
    from region_context_student import RegionFeatureRefinement
    from shared_feature_student import configure_shared

    if args.output.exists():
        raise ValueError('Use a new private output directory.')
    first, record = load_first(args.first)
    candidate_record = read(args.candidate / 'result.json')
    evaluation = read(args.evaluation)
    checkpoint = args.candidate / 'student-private.pt'
    if not (candidate_record['complete'] and candidate_record['first_stage_unchanged']
            and candidate_record['region_context']['mode'] == 'routed'
            and candidate_record['first_checkpoint_sha256'] == sha(args.first / 'student-private.pt')
            and evaluation['candidate_checkpoint_sha256'] == sha(checkpoint)):
        raise ValueError('Candidate provenance check failed.')
    model = RegionFeatureRefinement(first, 'routed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['grade_parameters'] != first.grade_parameters:
        raise ValueError('Grading configuration changed.')
    model.load_state_dict(state['state_dict'], strict=True)
    model.cuda().half().eval().to(memory_format=torch.channels_last)
    cases = validation_cases(args.base, args.lab, args.photos, args.images, record['controls'])
    args.output.mkdir(parents=True)
    input_row = cases[0]
    source = source_from_capture(input_row[1], input_row[2], record['controls'])
    with torch.inference_mode():
        # Original standard-operator path: no dependency on Python CUDA wrappers.
        program = torch.export.export(model, (source,), strict=True)
        operators = graph_operators(program)
        if not any('topk' in op for op in operators) or not any('scaled_dot_product_attention' in op for op in operators):
            raise ValueError('Data-dependent routing or attention is missing from the graph.')
        artifact = args.output / 'student-private.pt2'
        torch.export.save(program, artifact)
        exported = torch.export.load(artifact).module()
        rows = []
        for label, path, hashes, group in cases:
            sample = source_from_capture(path, hashes, record['controls'])
            eager, deployed = model(sample), exported(sample)
            if not torch.equal(eager.view(torch.int16), deployed.view(torch.int16)):
                raise ValueError('Export changed output: ' + label)
            if not torch.isfinite(deployed).all():
                raise ValueError('Nonfinite exported output: ' + label)
            rows.append({'label': label, 'group': group, 'capture': str(path.resolve()),
                         'capture_hashes': hashes, 'output_sha256': tensor_sha(eager), 'bitwise_equal': True})
            print(json.dumps({'label': label, 'export_bitwise_equal': True}), flush=True)
        # Reuse existing fusions unchanged to compare with the previously timed path.
        backend = FusedNorm()
        configure_shared(model, 'all-conditioning', backend)
        for row in rows:
            sample = source_from_capture(Path(row['capture']), row['capture_hashes'], record['controls'])
            if tensor_sha(model(sample)) != row['output_sha256']:
                raise ValueError('Export differs from the previously fused candidate: ' + row['label'])
        timing = paired_timing({'candidate_existing_fusions': model, 'export_standard_operators': exported}, source)
    result = {'complete': True, 'artifact_sha256': sha(artifact), 'artifact_bytes': artifact.stat().st_size,
              'candidate_checkpoint_sha256': sha(checkpoint), 'first_checkpoint_sha256': sha(args.first / 'student-private.pt'),
              'torch_version': torch.__version__, 'native_shape': [1080, 1920],
              'parameters': sum(value.numel() for value in model.parameters()),
              'strict_export': True, 'operators': operators, 'controls': record['controls'],
              'input': {'shape': [1, 3, 1080, 1920], 'dtype': 'float16', 'device': 'cuda', 'layout': 'channels_last'},
              'all_outputs_bitwise_equal': True, 'matches_existing_fused_candidate': True,
              'routing_preserved_as_tensor_operations': True, 'cases': rows, 'timing': timing,
              'scope': 'Fixed-shape inference export; same trained weights, no new kernels. Timing excludes application, D3D12 and frame transfer. Not a native runtime DLL or a quality acceptance.'}
    write(args.output / 'manifest-private.json', result)
    print(json.dumps({'complete': True, 'artifact_bytes': result['artifact_bytes'], 'timing': timing['summary']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    create = sub.add_parser('export')
    for name in ('base', 'lab', 'photos', 'images', 'first', 'candidate', 'evaluation', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Artifacts must stay outside the public repository.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.operation == 'export':
        export(args)
    else:
        verify(args.output)


if __name__ == '__main__':
    main()
