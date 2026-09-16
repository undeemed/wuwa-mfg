# SPDX-License-Identifier: Apache-2.0
"""Check auxiliary-gradient connectivity, split isolation and inference removal.

Uses private matched targets and one training input; never launches an app.
"""
import argparse
import json
from pathlib import Path

import torch

from compare_output_grade import read_capture
from feature_hint import FeatureHints
from output_grade import GradedStudent
from student_probe import HierarchicalStudent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory.')
    record = json.loads(args.training_result.read_text())
    controls, images, hashes = read_capture(args.capture)
    assert controls == record['controls'] and hashes in record['training_capture_hashes']
    torch.manual_seed(28411)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    network = HierarchicalStudent(16, 2).cuda()
    model = GradedStudent(network, record['architecture']['explicit_output_grading']).cuda()
    source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda()
    source = source.contiguous(memory_format=torch.channels_last)
    keys = set(model.state_dict())
    with torch.no_grad():
        before = model(source)
    hint = FeatureHints(args.targets, network, controls, record['training_capture_hashes'],
                        [record['validation_capture_hashes']] + [r['capture_hashes'] for r in record['extra_validation']])
    with torch.no_grad():
        assert torch.equal(before, model(source)), 'Attaching training hints changed RGB inference.'
    assert hint.loss('unknown-input') is None
    rejected = 0
    for key, split in hint.splits.items():
        if split == 'validation':
            try:
                hint.loss(key)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError('A validation feature entered the training loss.')
    assert rejected == hint.info['validation_target_count'] > 0
    optimizer = torch.optim.SGD(hint.projector.parameters(), lr=.001)
    model(source)
    loss = hint.loss(hashes['color'])
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    projector_gradient = float(hint.projector.weight.grad.abs().max())
    assert projector_gradient > 0
    optimizer.step()
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    model(source)
    hint.loss(hashes['color']).backward()
    backbone_gradient = float(network.stem.weight.grad.abs().max())
    assert backbone_gradient > 0, 'Auxiliary loss did not reach the student backbone.'
    assert all(not value.requires_grad and value.grad is None for value in hint.targets.values())
    with torch.no_grad():
        attached = model(source)
    args.output.mkdir(parents=True)
    hint.close(args.output)
    assert not network.decoder[0]._forward_hooks and not hint.current
    assert set(model.state_dict()) == keys, 'Training projector leaked into the inference checkpoint.'
    reloaded = GradedStudent(HierarchicalStudent(16, 2), record['architecture']['explicit_output_grading']).cuda()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(attached, model(source)) and torch.equal(attached, reloaded(source))
    report = {'complete': True, 'attach_and_remove_rgb_bit_equal': True,
              'inference_state_dict_unchanged': True, 'strict_reload_rgb_bit_equal': True,
              'validation_targets_rejected': rejected, 'unknown_input_has_no_auxiliary_loss': True,
              'first_projector_gradient_max': projector_gradient,
              'backbone_gradient_after_projector_step_max': backbone_gradient,
              'targets_are_constants': True, 'native_activations_used_at_inference': False,
              'target_manifest_sha256': hint.info['manifest_sha256'], 'capture_hashes': hashes}
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
