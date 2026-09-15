# SPDX-License-Identifier: Apache-2.0
"""Check backbone preservation, global influence, gradients and SDPA execution."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import torch
from torch.nn import functional as F

from attention_student import GlobalAttention
from student_probe import HierarchicalStudent


BASELINE = '75a64e09dc0471b8f11e838d93be4eb4c7deb1ad'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ['git', '-C', str(repo), 'show', f'{BASELINE}:research/neural-latency/student_probe.py'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    records = []
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'previous_student.py'
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('previous_student', path)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        for dtype in (torch.float32, torch.float16):
            torch.manual_seed(28411)
            old = previous.HierarchicalStudent(16, 2).cuda().to(dtype).eval()
            torch.manual_seed(28411)
            new = HierarchicalStudent(16, 2).cuda().to(dtype).eval()
            torch.manual_seed(28411)
            attention = HierarchicalStudent(16, 2, attention=True).cuda().to(dtype).eval()
            assert all(torch.equal(value, new.state_dict()[key]) and
                       torch.equal(value, attention.state_dict()[key])
                       for key, value in old.state_dict().items())
            # The original zero head would hide broken intermediate behavior.
            with torch.no_grad():
                old.head.weight.normal_(0, .1)
                old.head.bias.normal_(0, .01)
                for model in (new, attention):
                    model.head.load_state_dict(old.head.state_dict())
            for shape in ((1, 3, 64, 96), (2, 3, 68, 100), (1, 3, 1080, 1920)):
                image = torch.rand(shape, device='cuda', dtype=dtype).contiguous(memory_format=torch.channels_last)
                with torch.inference_mode():
                    expected, actual, zero_branch = old(image), new(image), attention(image)
                record = {'dtype': str(dtype), 'shape': list(shape),
                          'unchanged_backbone_exact': torch.equal(expected, actual),
                          'zero_attention_exact': torch.equal(expected, zero_branch)}
                records.append(record)
                assert record['unchanged_backbone_exact'] and record['zero_attention_exact']
        del old, new, attention, expected, actual, zero_branch

    torch.manual_seed(91427)
    layer = GlobalAttention(96).cuda()
    torch.nn.init.normal_(layer.project.weight, 0, .04)
    x = torch.randn(2, 96, 3, 5, device='cuda', requires_grad=True)
    actual = layer(x)
    tokens = x.flatten(2).transpose(1, 2)
    qkv = layer.qkv(layer.norm(tokens)).reshape(2, 15, 3, 3, 32)
    query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    manual = ((query @ key.transpose(-1, -2)) / (32 ** .5)).softmax(-1) @ value
    manual = manual.transpose(1, 2).reshape(2, 15, 96)
    expected = x + .1 * layer.project(manual).transpose(1, 2).reshape_as(x)
    numerical_error = float((actual - expected).abs().max())
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    changed = x.detach().clone()
    changed[:, :, -1, -1] = torch.randn(2, 96, device='cuda') * 2
    influence = float((layer(changed)[:, :, 0, 0] - actual[:, :, 0, 0]).abs().max())
    assert influence > 1e-6
    actual.square().mean().backward()
    gradients = {key: {'finite': bool(torch.isfinite(parameter.grad).all()),
                       'max_abs': float(parameter.grad.abs().max())}
                 for key, parameter in layer.named_parameters()}
    assert all(record['finite'] and record['max_abs'] > 0 for record in gradients.values())

    # Observe backend selection at the actual bottleneck, rather than assuming
    # that a paper's implementation or acceleration applies to this device.
    operators = {}
    for dtype in (torch.float32, torch.float16):
        candidate = GlobalAttention(96).cuda().to(dtype).eval()
        features = torch.randn(1, 96, 34, 60, device='cuda', dtype=dtype)
        with torch.inference_mode():
            candidate(features)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                candidate(features)
                torch.cuda.synchronize()
        operators[str(dtype)] = sorted(event.key for event in profiler.key_averages()
                                       if 'attention' in event.key.lower())
    report = {'schema': 1, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'backbone_reference_commit': BASELINE, 'backbone_cases': records,
              'manual_attention_max_abs': numerical_error,
              'distant_token_influence_max_abs': influence, 'gradients': gradients,
              'observed_sdpa_operators': operators,
              'scope': 'Implementation and training-mechanics checks; no native-quality or app-latency claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
