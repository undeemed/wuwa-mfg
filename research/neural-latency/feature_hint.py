# SPDX-License-Identifier: Apache-2.0
"""Training-only projection against matched, normalized native decoder targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FeatureHints:
    def __init__(self, manifest_path, network, controls, training_hashes, validation_hashes):
        path = Path(manifest_path).resolve()
        if path.is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError('Native feature targets must remain private.')
        manifest = json.loads(path.read_text())
        assert manifest['complete'] and manifest['controls'] == controls
        assert manifest['target_shape_hwc'] == [270, 480, 32]
        assert manifest['normalization']['fit_split'] == 'train'
        assert manifest['decoder_layout'] == 'two-plane-permuted'
        assert manifest['valid_native_decoder_crop_hwc'] == [540, 960, 32]
        train = {h['color']: h for h in training_hashes}
        validation = {h['color']: h for h in validation_hashes}
        assert not train.keys() & validation.keys()
        self.targets, self.splits, self.current = {}, {}, {}
        channel_sum, channel_sq, count = np.zeros(32, np.float64), np.zeros(32, np.float64), 0
        for row in manifest['cases']:
            digest = row['capture_hashes']['color']
            assert digest not in self.targets and row['split'] in ('train', 'validation')
            lookup = train if row['split'] == 'train' else validation
            assert lookup.get(digest) == row['capture_hashes'], 'Auxiliary targets must match both RGB input and output.'
            filename = row['target_file']
            assert Path(filename).name == filename
            data = (path.parent/filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row['target_sha256']
            target = np.load(path.parent/filename, allow_pickle=False)
            assert target.dtype == np.float32 and target.shape == (270, 480, 32) and np.isfinite(target).all()
            if row['split'] == 'train':
                flat = target.reshape(-1, 32).astype(np.float64)
                channel_sum += flat.sum(axis=0)
                channel_sq += np.square(flat).sum(axis=0)
                count += flat.shape[0]
            self.targets[digest] = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda()
            self.splits[digest] = row['split']
        assert count == manifest['normalization']['pixels_per_channel'] and count > 0
        expected_variance = np.square(np.array(manifest['normalization']['std'])/np.array(manifest['normalization']['scale']))
        assert np.max(np.abs(channel_sum/count)) < .0001
        assert np.max(np.abs(channel_sq/count-expected_variance)) < .0001
        self.projector = nn.Conv2d(network.head.in_channels, 32, 1).cuda()
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        self.hook = network.decoder[0].register_forward_hook(self._observe)
        self.info = {'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'training_target_count': sum(v=='train' for v in self.splits.values()),
                     'validation_target_count': sum(v=='validation' for v in self.splits.values()),
                     'training_rgb_count': len(train), 'validation_rgb_count': len(validation),
                     'projection_parameters': sum(p.numel() for p in self.projector.parameters()),
                     'target_shape_hwc': [270, 480, 32], 'projection_initialization': 'zero',
                     'normalization_fit_on_training_only': True,
                     'loss': 'Mean squared error of normalized decoder targets after a learned 1x1 projection.',
                     'training_only': True, 'native_activations_used_at_inference': False,
                     'target_capture_hashes': {split: [r['capture_hashes'] for r in manifest['cases'] if r['split']==split]
                                               for split in ('train', 'validation')}}

    def _observe(self, module, inputs, output):
        self.current['features'] = output

    def loss(self, color_hash):
        if color_hash not in self.targets:
            return None
        assert self.splits[color_hash] == 'train', 'Validation features cannot enter the training loss.'
        predicted = self.projector(self.current['features'])[:, :, :270, :480]
        return (predicted-self.targets[color_hash]).square().mean()

    @torch.no_grad()
    def evaluate(self, model, pairs):
        rows = []
        for source, hashes in pairs:
            key = hashes['color']
            if key not in self.targets:
                continue
            model(source.contiguous(memory_format=torch.channels_last))
            predicted = self.projector(self.current['features'])[:, :, :270, :480]
            delta = predicted-self.targets[key]
            rows.append({'capture_hashes': hashes, 'split': self.splits[key],
                         'normalized_feature_mse': float(delta.square().mean()),
                         'normalized_feature_mae': float(delta.abs().mean())})
        assert len(rows) == len(self.targets)
        return rows

    def close(self, output):
        self.hook.remove()
        self.current.clear()
        torch.save({'state_dict': self.projector.cpu().state_dict(), 'training_only': True,
                    'manifest_sha256': self.info['manifest_sha256']}, output/'feature-projector-private.pt')
