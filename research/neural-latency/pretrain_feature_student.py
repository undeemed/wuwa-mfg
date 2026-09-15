# SPDX-License-Identifier: Apache-2.0
"""Pretrain a private RGB student's backbone against native intermediate hints.

The zero-initialized RGB head is excluded from this loss. A later, separate RGB
training run loads these weights with a fresh optimizer and no feature loss.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from compare_output_grade import read_capture
from feature_hint import FeatureHints
from output_grade import GradedStudent
from student_probe import HierarchicalStudent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--baseline-result', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.exists() or any(p.resolve().is_relative_to(repo) for p in [args.targets, args.base, args.output]):
        raise ValueError('Use private data and a fresh private output directory.')
    if not 1 <= args.steps <= 3000:
        raise ValueError('Use a bounded pretraining run of up to 3000 steps.')
    baseline = json.loads(args.baseline_result.read_text())
    manifest = json.loads(args.targets.read_text())
    architecture = baseline['architecture']
    assert architecture['variant']=='hierarchical' and architecture['width']==16 and architecture['blocks']==2
    assert architecture['noise_channels']==0 and baseline['native_shape']==[1080, 1920]
    torch.manual_seed(28411)
    rng = np.random.default_rng(28411)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GradedStudent(HierarchicalStudent(16, 2), architecture['explicit_output_grading'])
    model = model.cuda().to(memory_format=torch.channels_last)
    training, validation, training_hashes, validation_hashes = [], [], [], []
    for row in manifest['cases']:
        name = row['name']
        trial = 'native-post-inputs-'+name if name in ['original', 'west'] else 'teacher-feature-'+name+'-original'
        assert Path(trial).name == trial
        controls, images, hashes = read_capture(args.base/'trials'/trial/'capture')
        assert controls==baseline['controls'] and hashes==row['capture_hashes']
        lookup = baseline['training_capture_hashes'] if row['split']=='train' else [baseline['validation_capture_hashes']]+[r['capture_hashes'] for r in baseline['extra_validation']]
        assert hashes in lookup
        source = torch.from_numpy(images['color']).permute(2, 0, 1).unsqueeze(0).cuda().contiguous(memory_format=torch.channels_last)
        if row['split']=='train':
            training.append((source, hashes)); training_hashes.append(hashes)
        else:
            assert row['split']=='validation'
            validation.append((source, hashes)); validation_hashes.append(hashes)
    assert len(training)==6 and len(validation)==3
    hint = FeatureHints(args.targets, model.network, baseline['controls'], training_hashes, validation_hashes)
    optimizer = torch.optim.AdamW([*model.parameters(), *hint.projector.parameters()], lr=.002, weight_decay=.0001)
    args.output.mkdir(parents=True)
    initial = hint.evaluate(model, training+validation)
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(args.steps):
        source, hashes = training[int(rng.integers(len(training)))]
        optimizer.zero_grad(set_to_none=True)
        lr = .00002+.5*(.002-.00002)*(1+math.cos(math.pi*step/max(1,args.steps-1)))
        for group in optimizer.param_groups:
            group['lr']=lr
        model(source)
        loss = hint.loss(hashes['color'])
        assert loss is not None
        loss.backward(); optimizer.step()
        if step%100==0 or step+1==args.steps:
            value = float(loss)
            assert math.isfinite(value)
            record = {'step': step+1, 'normalized_feature_mse': value, 'seconds': time.perf_counter()-started}
            history.append(record); print(json.dumps(record), flush=True)
        if time.perf_counter()-started > 600:
            raise RuntimeError('Pretraining exceeded its bounded runtime; no checkpoint published.')
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    assert model.network.head.weight.grad is None and model.network.head.bias.grad is None
    assert torch.count_nonzero(model.network.head.weight)==0 and torch.count_nonzero(model.network.head.bias)==0
    final = hint.evaluate(model, training+validation)
    hint.close(args.output)
    assert not model.network.decoder[0]._forward_hooks
    report = {'schema': 1, 'experiment': 'Feature-only backbone pretraining before separate RGB fitting',
              'target_achieved': False, 'quality_gate_passed': False,
              'architecture': architecture, 'controls': baseline['controls'], 'native_shape': [1080, 1920],
              'training_capture_hashes': training_hashes, 'validation_capture_hashes': validation_hashes,
              'completed_steps': args.steps, 'training_seconds': elapsed, 'seed': 28411,
              'optimization': {'initial_learning_rate': .002, 'final_learning_rate': .00002, 'cosine_decay': True},
              'feature_supervision': hint.info, 'initial_feature_metrics': initial, 'final_feature_metrics': final,
              'training': history, 'rgb_head_unchanged_zero': True, 'rgb_loss_used': False,
              'baseline_result_sha256': sha(args.baseline_result),
              'limitations': ['Only six native feature training inputs and three held-out feature inputs.',
                  'A separate RGB training stage must be evaluated before any quality conclusion.',
                  'Extra pretraining steps make the later model more expensive to train than the RGB-only baseline.',
                  'No application timing, temporal quality or game integration is established.']}
    torch.save({'architecture': architecture, 'state_dict': model.cpu().state_dict()}, args.output/'student-private.pt')
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'completed_steps': args.steps, 'training_seconds': elapsed, 'rgb_head_unchanged_zero': True}), flush=True)


if __name__ == '__main__':
    main()
