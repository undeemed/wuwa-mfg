# SPDX-License-Identifier: Apache-2.0
"""Validate the fixed mixture, legacy RNG path and malformed sampling reports."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from cluster_sampling import draw_pair,validate_coverage
from student_training_pairs import read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('coverage','first','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    record=read(args.coverage);first=read(args.first/'result.json');checkpoint=sha(args.first/'student-private.pt')
    probabilities=validate_coverage(record,first['training_capture_hashes'],checkpoint)
    original=np.random.default_rng(28411);current=np.random.default_rng(28411)
    for _ in range(4500):assert draw_pair(current)==(int(original.integers(30)),30+int(original.integers(32)))
    rng=np.random.default_rng(28411);counts=np.zeros(62,dtype=np.int64)
    for _ in range(4500):
        a,b=draw_pair(rng,probabilities);assert 0<=a<30<=b<62;counts[a]+=1;counts[b]+=1
    assert counts[:30].sum()==counts[30:].sum()==4500 and (counts>0).all()
    rng=np.random.default_rng(28411);repeat=np.zeros(62,dtype=np.int64)
    for _ in range(4500):
        for index in draw_pair(rng,probabilities):repeat[index]+=1
    assert np.array_equal(counts,repeat)
    rejected=[]
    changes=[('checkpoint',lambda r:r.update(first_checkpoint_sha256='0'*64)),
             ('validation-use',lambda r:r.update(validation_pixels_or_scores_used_for_selection=True)),
             ('targets-use',lambda r:r.update(teacher_targets_used_for_selection=True)),
             ('probability',lambda r:r['domains']['scene']['probabilities'].__setitem__(0,.9)),
             ('cluster-count',lambda r:r['domains']['photos']['cluster_counts'].__setitem__(0,1)),
             ('training-order',lambda r:r['training'].reverse())]
    for label,change in changes:
        altered=copy.deepcopy(record);change(altered)
        try:validate_coverage(altered,first['training_capture_hashes'],checkpoint)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'coverage_sha256':sha(args.coverage),
            'legacy_4500_pairs_exact':True,'mixed_repeat_exact':True,'every_training_image_sampled':True,
            'domain_counts':[int(counts[:30].sum()),int(counts[30:].sum())],'per_image_counts':counts.tolist(),'rejected_reports':rejected,
            'scope':'Sampling only; no model forward, training, validation pixel access or application launch.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
