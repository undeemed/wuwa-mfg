# SPDX-License-Identifier: Apache-2.0
"""Validated TRAIN-only sampling probabilities; no inference behavior changes."""
import numpy as np


def validate_coverage(record, hashes, checkpoint_sha):
    if not record['complete'] or record['first_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Coverage does not match this frozen first model.')
    if record['training_count'] != 62 or record['validation_count_excluded'] != 16:
        raise ValueError('Unexpected data split.')
    if record['teacher_targets_used_for_selection'] or record['validation_pixels_or_scores_used_for_selection']:
        raise ValueError('Sampling must be selected from TRAIN input descriptors only.')
    if record['clusters_per_domain'] != 4 or record['mixture_uniform_images'] != .5:
        raise ValueError('Unexpected preselected sampling rule.')
    if [row['capture_hashes'] for row in record['training']] != hashes:
        raise ValueError('Coverage order or training identities differ.')
    probabilities=[]
    for name, indices in [('scene',list(range(30))),('photos',list(range(30,62)))]:
        data=record['domains'][name]
        assignment=np.asarray(data['assignments'],dtype=np.int64)
        if data['indices']!=indices or assignment.shape!=(len(indices),) or np.any((assignment<0)|(assignment>=4)):
            raise ValueError('Unexpected domain or cluster assignments.')
        counts=np.bincount(assignment,minlength=4)
        if counts.tolist()!=data['cluster_counts'] or (counts==0).any():
            raise ValueError('Cluster membership counts differ.')
        expected=.5/len(indices)+.5/(4*counts[assignment])
        p=np.asarray(data['probabilities'],dtype=np.float64)
        if p.shape!=expected.shape or not np.array_equal(p,expected) or not np.isclose(p.sum(),1):
            raise ValueError('Sampling probabilities do not implement the recorded rule.')
        probabilities.append(p)
    return tuple(probabilities)


def draw_pair(rng, probabilities=None):
    if probabilities is None:
        return int(rng.integers(30)),30+int(rng.integers(32))
    return int(rng.choice(30,p=probabilities[0])),30+int(rng.choice(32,p=probabilities[1]))
