# SPDX-License-Identifier: Apache-2.0
"""Audit native brightness pairs and the matched training sampling schedule."""
import argparse,json
from pathlib import Path
import numpy as np
from collect_demo_photo_training import read,sha,dump
from collect_training_brightness import audited_brightness,selected_sources
from compare_output_grade import read_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('base','photos','images','brightness','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    sources=selected_sources(a.photos,a.images)
    controls=read(a.base/(sources[0][0]['label']+'-state')/'result.json')['controls']
    bright=audited_brightness(a.brightness,a.base,controls,a.photos,a.images);rows=[]
    for (source,_),higher in zip(sources,bright):
        lowc,low,lowh=read_capture(a.base/(source['label']+'-state')/'capture')
        highc,high,highh=read_capture(a.base/(higher['label']+'-state')/'capture')
        assert lowc==highc==controls and lowh==source['capture_hashes'] and highh==higher['capture_hashes']
        row={'name':source['name'],'source_sha256':source['source_sha256'],'low_capture_hashes':lowh,'high_capture_hashes':highh,
            'capture_proof':higher['proof'],'statistics':{}}
        for role in ('color','output'):
            assert low[role].shape==high[role].shape==(1080,1920,3)
            stats=lambda x:{'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max()),'fraction_at_least_one':float(np.mean(x>=1))}
            row['statistics'][role]={'low':stats(low[role]),'high':stats(high[role]),'pair_mean_abs_difference':float(np.abs(low[role]-high[role]).mean())}
        rows.append(row)
    rng=np.random.default_rng(28411);scene_counts=[0]*30;photo_counts=[0]*32;schedule=[]
    for _ in range(4500):
        scene=int(rng.integers(30));photo=int(rng.integers(32));scene_counts[scene]+=1;photo_counts[photo]+=1;schedule.append([scene,photo])
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'training_identities':16,'new_training_frames':16,
        'emittances':[.1,1.],'native_shape':[1080,1920],'validation_identities_used':0,'controls':controls,
        'manifest_sha256':sha(a.brightness),'rows':rows,
        'paired_schedule':{'seed':28411,'updates':4500,'examples':9000,'domain_weights':[.5,.5],
            'scene_counts':scene_counts,'photo_slot_counts':photo_counts,'extra_photo_slot_draws':sum(photo_counts[16:]),
            'schedule_sha256':__import__('hashlib').sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest(),
            'control_mapping':'Photo slots 0..15 and 16..31 both use the same sixteen low-emission captures.',
            'candidate_mapping':'Photo slots 0..15 use low-emission captures; slots 16..31 use the same identities at emission 1.0.'},
        'limitations':['Native sample rendering determines the actual neural input; multiplying emission by ten does not imply a tenfold input change.',
            'Same source images and static first-reset captures; no new semantic identities or temporal examples.',
            'Validation remains outside fitting but previous scores motivated this test; no independent final quality acceptance.']}
    dump(a.output,report);print(json.dumps({'complete':True,'pairs':len(rows),'extra_photo_slot_draws':sum(photo_counts[16:])}))


if __name__=='__main__':main()
