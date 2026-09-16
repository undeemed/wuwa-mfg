# SPDX-License-Identifier: Apache-2.0
"""Warm-start the existing routed student on broader native-teacher captures.

The inference architecture is unchanged. Frozen base features are recomputed;
CPU caching retains the exact FP16 source/target values without a GPU-size cache.
Development and final gameplay data are excluded from optimization.
"""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from compare_output_grade import read_capture
from progressive_student import load_first
from region_context_student import RegionFeatureRefinement, architecture
from student_training_pairs import rgb_loss
from train_student_collection import original_training_names


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_ram():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                   ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    value = Memory(); value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return value.avail_phys


def old_pairs(base, record):
    expected = {row['color']: row for row in record['training_capture_hashes']}
    collection = read(base/'teacher-illumination-manifest.json')
    assert collection['scene_restored'] and collection['dll_restored']
    scene_labels = set(original_training_names()) | {r['label'] for r in collection['views'] if r['split'] == 'train'}
    assert len(scene_labels) == 30 and all(Path(n).name == n for n in scene_labels)
    candidates = {}
    # Observer experiments contain byte-identical copies of old scene captures.
    # Prefer the original scene collection so aliases cannot change its domain.
    paths = sorted(base.rglob('frame-0.json'),
                   key=lambda p: (p.parent.parent.parent != base/'trials', str(p)))
    for metadata in paths:
        if metadata.parent.name != 'capture':
            continue
        # Read existing provenance first; only matching training frames load pixels.
        path = metadata.parent
        label = path.parent.name.removesuffix('-state')
        if path.parent.parent == base/'trials' and label not in scene_labels:
            continue
        if path.parent.name.endswith('-state') and (path.parent/'result.json').exists():
            hashes = read(path.parent/'result.json').get('capture_hashes', {})
        else:
            frame = read(metadata)
            resource = frame.get('resources', {}).get('color', {})
            name = resource.get('file', '')
            if Path(name).name != name or not name:
                continue
            hashes = {'color': sha(path/name)}
        if hashes.get('color') in expected and hashes['color'] not in candidates:
            if 'output' not in hashes:
                frame = read(metadata)
                name = frame['resources']['output']['file']
                assert Path(name).name == name
                hashes['output'] = sha(path/name)
            if hashes != expected[hashes['color']]:
                continue
            candidates[hashes['color']] = {'label': label, 'path': path,
                                         'group': 'old-scene' if path.parent.parent.name == 'trials' else 'old-photos',
                                         'capture_hashes': expected[hashes['color']]}
    if set(candidates) != set(expected):
        raise ValueError('Could not locate the exact existing 62 training pairs.')
    rows = [candidates[row['color']] for row in record['training_capture_hashes']]
    if Counter(row['group'] for row in rows) != Counter({'old-scene': 30, 'old-photos': 32}):
        raise ValueError('Unexpected old training-domain counts.')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'first', 'candidate', 'sources', 'teacher', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.exists() or args.output.resolve().is_relative_to(repo) or args.steps != 6000:
        raise ValueError('Use a fresh private output and the preregistered 6000 updates.')
    running = subprocess.run(['powershell', '-NoProfile', '-Command',
        '@(Get-Process -Name ngx_dlss_demo,Client-Win64-Shipping -ErrorAction SilentlyContinue).Count'],
        capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if running.stdout.strip() != '0':
        raise RuntimeError('Wait for capture collection to finish and the game to close.')
    sources, teacher, review = read(args.sources), read(args.teacher), read(args.review)
    assert sources['complete'] and teacher['complete'] and len(teacher['cases']) == 240
    assert teacher['prepared_manifest_sha256'] == sha(args.sources)
    assert review['complete'] and review['source_manifest_sha256'] == sha(args.sources)
    assert set(review['approved_ids']) == {r['id'] for r in sources['cases']} and not review['rejected_ids']
    assert Counter(r['split'] for r in sources['cases']) == Counter({'train': 200, 'development': 40})
    source_rows = {r['id']: r for r in sources['cases']}
    first, record = load_first(args.first)
    candidate_record = read(args.candidate/'result.json')
    assert candidate_record['first_checkpoint_sha256'] == sha(args.first/'student-private.pt')
    pairs = old_pairs(args.base, record)
    development = set()
    for row in teacher['cases']:
        source = source_rows[row['id']]
        assert row['split'] == source['split'] and row['source_sha256'] == source['source_sha256']
        assert row['controls'] == record['controls'] and row['four_fenced_frames_rehashed']
        assert row['texture_sha256'] == source['texture_sha256']
        if row['split'] == 'development':
            development.add(row['capture_hashes']['color'])
        else:
            pairs.append({'label': row['id'], 'path': args.base/(row['label']+'-state')/'capture',
                          'group': 'new-photos', 'capture_hashes': row['capture_hashes']})
    old_validation = {record['validation_capture_hashes']['color']} | {r['capture_hashes']['color'] for r in record['extra_validation']}
    training = {r['capture_hashes']['color'] for r in pairs}
    assert len(pairs) == len(training) == 262 and not training & (development | old_validation)
    assert len(development) == 40 and len(old_validation) == 16
    model = RegionFeatureRefinement(first, 'routed')
    frozen = {key: value.clone() for key, value in model.first.state_dict().items()}
    state = torch.load(args.candidate/'student-private.pt', map_location='cpu', weights_only=True)
    assert state['grade_parameters'] == first.grade_parameters
    model.load_state_dict(state['state_dict'], strict=True)
    assert all(torch.equal(value, frozen[key]) for key, value in model.first.state_dict().items())
    torch.manual_seed(190915); rng = np.random.default_rng(190915)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    model.cuda().float().to(memory_format=torch.channels_last)
    # Original teacher textures are FP16, so this host cache loses no information.
    required = len(pairs)*2*1080*1920*3*2
    if free_ram() < required + 4*1024**3:
        raise RuntimeError('Need room for the exact host image cache plus a 4 GiB reserve.')
    host = []
    for index, pair in enumerate(pairs):
        controls, arrays, hashes = read_capture(pair['path'])
        assert controls == record['controls'] and hashes == pair['capture_hashes']
        tensors = []
        for role in ('color', 'output'):
            half = arrays[role].astype(np.float16)
            assert np.array_equal(half.astype(np.float32), arrays[role])
            tensors.append(torch.from_numpy(half).permute(2, 0, 1).unsqueeze(0).contiguous(memory_format=torch.channels_last))
        host.append(tensors)
        if index % 25 == 0:
            print(json.dumps({'loaded_exact_pairs': index+1, 'total': len(pairs)}), flush=True)
    args.output.mkdir()
    optimizer = torch.optim.AdamW(model.refinement.parameters(), lr=.0002, weight_decay=.0001)
    groups = {name: [i for i, pair in enumerate(pairs) if pair['group'] == name]
              for name in ('old-scene', 'old-photos', 'new-photos')}
    if {name: len(indices) for name, indices in groups.items()} != {
            'old-scene': 30, 'old-photos': 32, 'new-photos': 200}:
        raise ValueError('Unexpected training sampling groups.')
    model.train(); history = []; counts = [0]*len(pairs); started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        lr = .000002 + .5*(.0002-.000002)*(1+math.cos(math.pi*step/(args.steps-1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        old_group = 'old-scene' if rng.random() < .5 else 'old-photos'
        old = int(rng.choice(groups[old_group]))
        new = int(rng.choice(groups['new-photos']))
        loss_total = 0.; pixel_total = 0.
        for index in (old, new):
            source, target = [t.cuda().float() for t in host[index]]
            output = model(source)
            loss, pixel = rgb_loss(output, target)
            (.5*loss).backward()
            loss_total += float(loss.detach())*.5; pixel_total += float(pixel.detach())*.5
            counts[index] += 1
        if not math.isfinite(loss_total):
            raise RuntimeError('Nonfinite loss; do not accept this fit.')
        optimizer.step()
        if step % 100 == 0 or step == args.steps-1:
            row = {'step': step+1, 'loss': loss_total, 'pixel_mae': pixel_total, 'seconds': time.perf_counter()-started}
            row['remaining_seconds_estimate'] = row['seconds']/(step+1)*(args.steps-step-1)
            history.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    assert all(torch.equal(value.cpu(), frozen[key]) for key, value in model.first.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.first.parameters())
    assert sum(counts[:62]) == sum(counts[62:]) == args.steps
    model.eval().cpu()
    torch.save({'state_dict': model.state_dict(), 'grade_parameters': model.first.grade_parameters}, args.output/'student-private.pt')
    result = {'complete': True, 'variant': 'region-context-broad-warm-start', 'target_achieved': False,
              'quality_gate_passed': False, 'native_shape': [1080, 1920], 'controls': record['controls'],
              'first_stage_unchanged': True, 'first_stage_has_gradients': False,
              'first_checkpoint_sha256': sha(args.first/'student-private.pt'),
              'first_result_sha256': sha(args.first/'result.json'),
              'starting_checkpoint_sha256': sha(args.candidate/'student-private.pt'),
              'starting_result_sha256': sha(args.candidate/'result.json'),
              'checkpoint_sha256': sha(args.output/'student-private.pt'),
              'source_manifest_sha256': sha(args.sources), 'teacher_manifest_sha256': sha(args.teacher),
              'visual_review_sha256': sha(args.review), 'training_frames': len(pairs), 'new_training_sources': 200,
              'development_count': 56, 'final_gameplay_test_count': 0,
              'no_training_development_overlap': True, 'parameters': sum(p.numel() for p in model.parameters()),
              'region_context': architecture('routed'),
              'completed_steps': args.steps, 'training_seconds': elapsed, 'history': history,
              'optimization': {'seed': 190915, 'optimizer': 'AdamW', 'initial_lr': .0002, 'final_lr': .000002,
                               'weight_decay': .0001, 'examples': 2*args.steps,
                               'expected_group_weights': [.25,.25,.5], 'groups': ['old-scene','old-photos','new-photos'],
                               'loss': 'L1 + 0.25 * horizontal and vertical gradient L1', 'training_precision': 'FP32, TF32 off'},
              'training_pairs': [{k:v for k,v in p.items() if k != 'path'} | {'sample_count': count} for p,count in zip(pairs,counts)],
              'scope': 'Same inference architecture and existing candidate warm start. Exact host FP16 input/target cache; FP32 model fitting. No final-test data, architecture change, new kernels or installation.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'complete': True, 'steps': args.steps, 'seconds': elapsed}), flush=True)


if __name__ == '__main__':
    main()
