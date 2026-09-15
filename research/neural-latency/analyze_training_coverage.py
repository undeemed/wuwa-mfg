# SPDX-License-Identifier: Apache-2.0
"""Audit TRAIN-only feature clusters and propose a fixed diversity sampling mix.

No validation pixels, teacher targets or validation scores choose the clusters.
Feature vectors remain private; export assignments and numerical summaries only.
"""
import argparse,json,statistics
from pathlib import Path
import numpy as np
import torch
from collect_training_brightness import audited_brightness
from collect_demo_photo_training import audited_photos
from collect_diverse_images import audited_images
from progressive_student import load_first
from student_training_pairs import read,sha,training_pairs
from compare_output_grade import read_capture


def cluster_mix(vectors, *, seed=28411, groups=4):
    """Seeded farthest-first k-means; mix uniform items and uniform clusters."""
    x=np.asarray(vectors,dtype=np.float64)
    if x.ndim!=2 or len(x)<groups or not np.isfinite(x).all():raise ValueError('Expected finite TRAIN descriptors.')
    std=x.std(axis=0);active=std>1e-4
    if not active.any():raise ValueError('No variable feature dimensions.')
    x=(x[:,active]-x[:,active].mean(axis=0))/std[active]
    rng=np.random.default_rng(seed);indices=[int(rng.integers(len(x)))]
    while len(indices)<groups:
        distance=((x[:,None]-x[indices][None])**2).sum(axis=2).min(axis=1)
        distance[indices]=-1;chosen=int(distance.argmax())
        if distance[chosen]<=0:raise ValueError('Not enough distinct feature descriptors.')
        indices.append(chosen)
    centers=x[indices].copy();assignment=None
    for iteration in range(100):
        current=((x[:,None]-centers[None])**2).sum(axis=2).argmin(axis=1)
        counts=np.bincount(current,minlength=groups)
        if (counts==0).any():raise ValueError('Empty cluster; do not silently change the fixed sampling rule.')
        if assignment is not None and np.array_equal(current,assignment):break
        assignment=current;centers=np.stack([x[assignment==i].mean(axis=0) for i in range(groups)])
    probabilities=.5/len(x)+.5/(groups*counts[assignment])
    assert np.isclose(probabilities.sum(),1) and np.all(probabilities>0)
    return {'assignments':assignment.tolist(),'cluster_counts':counts.tolist(),'probabilities':probabilities.tolist(),
            'active_dimensions':int(active.sum()),'iterations':iteration+1,
            'uniform_cluster_mass':(counts/len(x)).tolist(),
            'mixed_cluster_mass':[float(probabilities[assignment==i].sum()) for i in range(groups)],
            'inertia_per_item':float(((x-centers[assignment])**2).sum(axis=1).mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','photos','images','brightness','baseline','first','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    first,record=load_first(args.first)
    pairs=training_pairs(args.base,args.photos,args.images,read(args.baseline/'result.json'))
    originals=audited_photos(args.photos,args.base,record['controls'])+audited_images(args.images,args.base,record['controls'],args.photos)
    identity_by_label={row['label']:row['source_sha256'] for row in originals if row['split']=='train'}
    bright=audited_brightness(args.brightness,args.base,record['controls'],args.photos,args.images)
    for row in bright:
        identity_by_label[row['label']]=row['source_sha256']
        pairs.append({'label':row['label'],'group':'bright-photos','path':args.base/(row['label']+'-state')/'capture','capture_hashes':row['capture_hashes']})
    assert len(pairs)==62 and [row['capture_hashes'] for row in pairs]==record['training_capture_hashes']
    validation={record['validation_capture_hashes']['color']}|{row['capture_hashes']['color'] for row in record['extra_validation']}
    assert len(validation)==16 and not validation&{row['capture_hashes']['color'] for row in pairs}
    first=first.cuda().float().eval().to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    rows=[];vectors=[]
    with torch.inference_mode():
        for pair in pairs:
            controls,arrays,hashes=read_capture(pair['path'])
            assert controls==record['controls'] and hashes==pair['capture_hashes']
            # Target pixels are not used for descriptors, clustering or sampling.
            image=arrays['color'];source=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float().contiguous(memory_format=torch.channels_last)
            features=[];first.network(source,feature_sink=features)
            deep=features[3]
            descriptor=torch.cat((deep.mean(dim=(0,2,3)),deep.std(dim=(0,2,3),unbiased=False))).cpu().numpy()
            assert descriptor.shape==(192,) and np.isfinite(descriptor).all();vectors.append(descriptor)
            rows.append({'label':pair['label'],'group':pair['group'],'capture_hashes':hashes,
                         'source_identity':identity_by_label.get(pair['label'],'sponza-scene'),
                         'input_mean':float(image.mean()),'input_std':float(image.std()),
                         'fraction_at_least_one':float(np.mean(image>=1)),
                         'gradient_magnitude':float(np.abs(np.diff(image,axis=0)).mean()+np.abs(np.diff(image,axis=1)).mean())})
    domains={}
    for name,indices in [('scene',list(range(30))),('photos',list(range(30,62)))]:
        data=np.stack([vectors[i] for i in indices]);cluster=cluster_mix(data)
        assert cluster==cluster_mix(data)
        domains[name]={'indices':indices,**cluster}
    # A deterministic fixture checks probability normalization and minority-cluster support.
    fixture=np.concatenate([np.tile([i*10.,i*i],(count,1)) for i,count in enumerate([12,4,3,1])])
    fixture_result=cluster_mix(fixture)
    assert sorted(fixture_result['cluster_counts'])==[1,3,4,12]
    for count,before,after in zip(fixture_result['cluster_counts'],fixture_result['uniform_cluster_mass'],fixture_result['mixed_cluster_mass']):
        assert abs(after-.25)<=abs(before-.25)+1e-12
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'first_checkpoint_sha256':sha(args.first/'student-private.pt'),
            'first_result_sha256':sha(args.first/'result.json'),'training_count':62,'validation_count_excluded':16,
            'distinct_source_identities':len({r['source_identity'] for r in rows}),
            'teacher_targets_used_for_selection':False,'validation_pixels_or_scores_used_for_selection':False,
            'descriptor':'192 scalar dimensions: mean and standard deviation of the frozen first encoder deepest 96 channels.',
            'sampling_rule':'Within each existing domain, mix 50% uniform-image and 50% uniform-cluster sampling. Keep scene/photo domain weights 50/50.',
            'seed':28411,'clusters_per_domain':4,'mixture_uniform_images':.5,
            'domains':domains,'training':rows,'deterministic_repeat_exact':True,'synthetic_probability_test_passed':True,
            'limitations':['Clusters describe this student feature space, not semantic categories or proven dataset deficiencies.',
                           'Source identities count one scene plus sixteen photos; scene poses are distinct views but not distinct scene sources.',
                           'No guarantee that balancing these clusters improves renderer fidelity. No training or inference speed claim.']}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'source_identities':report['distinct_source_identities'],'domains':{k:{key:v[key] for key in ['cluster_counts','uniform_cluster_mass','mixed_cluster_mass']} for k,v in domains.items()}},indent=2))


if __name__=='__main__':main()
