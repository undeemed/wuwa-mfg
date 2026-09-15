# SPDX-License-Identifier: Apache-2.0
"""Measure the affine rank limit of training-only native feature supervision.

Principal components are fitted on training targets only. Projection residuals
are optimistic oracle bounds: no RGB predictor or image-quality claim is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(array):
    x = array.reshape(-1, 32)
    total, gram = np.zeros(32, np.float64), np.zeros((32, 32), np.float64)
    for start in range(0, len(x), 8192):
        block = x[start:start+8192].astype(np.float64)
        total += block.sum(axis=0)
        gram += block.T @ block
    return len(x), total, gram


def projection_error(count, total, gram, basis, origin):
    residual = np.eye(32) - basis @ basis.T
    centered = gram/count - np.outer(total/count, origin) - np.outer(origin, total/count) + np.outer(origin, origin)
    return float(np.trace(residual @ centered @ residual.T)/32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.exists() or any(p.resolve().is_relative_to(repo) for p in [args.targets, args.student, args.output]):
        raise ValueError('Use private inputs and a fresh private output directory.')
    manifest = json.loads(args.targets.read_text())
    student = json.loads((args.student/'result.json').read_text())
    assert manifest['complete'] and manifest['normalization']['fit_split'] == 'train'
    hint = student['feature_supervision']
    assert hint['manifest_sha256'] == sha(args.targets)
    projector_file = args.student/'feature-projector-private.pt'
    checkpoint = torch.load(projector_file, map_location='cpu', weights_only=True)
    assert checkpoint['training_only'] and checkpoint['manifest_sha256'] == sha(args.targets)
    matrix = checkpoint['state_dict']['weight'].numpy().reshape(32, -1).astype(np.float64)
    bias = checkpoint['state_dict']['bias'].numpy().astype(np.float64)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10))
    learned_basis = left[:, :rank]
    stats = []
    position_sum = np.zeros((270, 480, 32), np.float64)
    position_count = 0
    for row in manifest['cases']:
        filename = row['target_file']
        assert Path(filename).name == filename
        target = args.targets.parent/filename
        assert sha(target) == row['target_sha256']
        array = np.load(target, allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (270, 480, 32) and np.isfinite(array).all()
        if row['split']=='train':
            position_sum += array
            position_count += 1
        stats.append((row, moments(array)))
    training = [s for row, s in stats if row['split'] == 'train']
    count = sum(s[0] for s in training)
    total = sum((s[1] for s in training), np.zeros(32))
    gram = sum((s[2] for s in training), np.zeros((32, 32)))
    assert count == manifest['normalization']['pixels_per_channel']
    mean = total/count
    covariance = gram/count - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    assert values.min() > -1e-8 and np.max(np.abs(mean)) < .0001
    position_template = position_sum/position_count
    args.output.mkdir(parents=True)
    np.savez(args.output/'private-pca.npz', mean=mean, basis=vectors, position_template=position_template)
    cases = []
    feature_metrics = {r['capture_hashes']['color']: r for r in hint['final_feature_metrics']}
    for row, (n, summed, products) in stats:
        observed = feature_metrics[row['capture_hashes']['color']]
        assert observed['split'] == row['split']
        scores = {str(k): projection_error(n, summed, products, vectors[:, :k], mean) for k in [8, 16, 24, 32]}
        learned = projection_error(n, summed, products, learned_basis, bias)
        array = np.load(args.targets.parent/row['target_file'], allow_pickle=False)
        position_error = float(np.mean(np.square(array-position_template)))
        per_image_constant = float((np.trace(products/n)-np.square(summed/n).sum())/32)
        assert min(scores.values()) >= -1e-8 and abs(scores['32']) < 1e-12
        assert learned <= observed['normalized_feature_mse'] + 1e-6
        cases.append({'label': row['name'], 'split': row['split'], 'capture_hashes': row['capture_hashes'],
                      'pca_projection_mse': scores, 'learned_projector_subspace_oracle_mse': learned,
                      'training_position_template_mse': position_error,
                      'per_image_channel_mean_oracle_mse': per_image_constant,
                      'actual_student_feature_mse': observed['normalized_feature_mse']})
    grouped = {s: {'pca_rank16_mse': float(np.mean([r['pca_projection_mse']['16'] for r in cases if r['split']==s])),
                  'learned_subspace_oracle_mse': float(np.mean([r['learned_projector_subspace_oracle_mse'] for r in cases if r['split']==s])),
                  'training_position_template_mse': float(np.mean([r['training_position_template_mse'] for r in cases if r['split']==s])),
                  'per_image_channel_mean_oracle_mse': float(np.mean([r['per_image_channel_mean_oracle_mse'] for r in cases if r['split']==s])),
                  'actual_student_feature_mse': float(np.mean([r['actual_student_feature_mse'] for r in cases if r['split']==s]))}
               for s in ['train', 'validation']}
    assert abs(grouped['train']['pca_rank16_mse'] - values[16:].sum()/32) < 1e-8
    report = {'schema': 1, 'complete': True, 'target_achieved': False, 'quality_gate_passed': False,
              'manifest_sha256': sha(args.targets), 'student_result_sha256': sha(args.student/'result.json'),
              'projector_sha256': sha(projector_file), 'fit_split': 'train', 'training_pixels': count,
              'learned_projector_rank': rank, 'channel_covariance_eigenvalues': values.tolist(),
              'cases': cases, 'group_means': grouped,
              'limitations': ['PCA uses native target activations and is an optimistic feature-space bound, not an RGB predictor.',
                  'A low feature reconstruction error does not prove low final RGB error or temporal fidelity.',
                  'The position template is fitted on training targets; per-image channel means are oracle native statistics, not available from an independent RGB predictor.',
                  'Six training and three validation feature images; not representative game data.',
                  'Projection vectors and all targets stay private; this makes no inference latency claim.']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(grouped, indent=2), flush=True)


if __name__ == '__main__':
    main()
