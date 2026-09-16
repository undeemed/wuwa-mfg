# SPDX-License-Identifier: Apache-2.0
"""Capture brighter versions of the sixteen existing TRAIN photo identities.

Uses the checked hidden sample on an inactive private desktop. Original images,
scene assets and native tensors stay private; validation identities are excluded.
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
from collect_demo_photo_training import read, sha, dump, audited_photos
from collect_diverse_images import audited_images
from evaluate_photo_students import audit_capture
from make_demo_image_scene import write_scene


def selected_sources(photos, images):
    rows=[]
    for manifest in (photos, images):
        record=read(manifest);assert record['complete']
        for row in record['cases']:
            if row['split']=='train':
                assert sha(manifest.parent/(row['name']+'-original.jpg'))==row['source_sha256']
                assert sha(manifest.parent/(row['name']+'-1080.png'))==row['texture_sha256']
                rows.append((row,manifest.parent/(row['name']+'-1080.png')))
    assert len(rows)==16 and len({r['source_sha256'] for r,_ in rows})==16
    validation={r['source_sha256'] for m in (photos,images) for r in read(m)['cases'] if r['split']=='validation'}
    assert not validation & {r['source_sha256'] for r,_ in rows}
    return rows


def audited_brightness(manifest, base, controls, photos, images):
    record=read(manifest)
    assert record['complete'] and record['selection_sha256']==sha(manifest.parent/'selection.json')
    selection=read(manifest.parent/'selection.json')
    assert selection['photo_manifest_sha256']==sha(photos) and selection['image_manifest_sha256']==sha(images)
    sources=selected_sources(photos,images)
    assert len(record['cases'])==len(selection['cases'])==16
    seen=set();rows=[]
    for row,chosen,(source,_) in zip(record['cases'],selection['cases'],sources):
        assert all(row[k]==v for k,v in chosen.items())
        assert row['name']==source['name'] and row['source_label']==source['label']
        assert row['source_sha256']==source['source_sha256'] and row['texture_sha256']==source['texture_sha256']
        assert row['split']=='train' and row['emittance']==1. and row['label']=='teacher-bright-'+row['name']
        current,arrays,hashes,proof=audit_capture(base,row['label'])
        assert current==controls and hashes==row['capture_hashes']
        assert hashes['color']!=source['capture_hashes']['color'] and hashes['color'] not in seen
        seen.add(hashes['color']);del arrays
        state=base/(row['label']+'-state');scene=read(state/'result.json')['scene_manifest']
        assert scene['texture_sha256']==source['texture_sha256'] and scene['emittance']==1.
        assert sha(manifest.parent/(row['name']+'-scene')/'manifest.json')==row['scene_manifest_sha256']
        assert scene==read(manifest.parent/(row['name']+'-scene')/'manifest.json')
        for name,digest in scene['files_sha256'].items():
            assert Path(name).name==name and sha(state/'assets'/name)==digest
        rows.append({**row,'proof':proof})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('photos','images','base','demo-dir','capture-dll','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--capture-sha256',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if a.output.resolve().is_relative_to(repo):raise ValueError('Use a private output directory.')
    sources=selected_sources(a.photos,a.images)
    controls=read(a.base/(sources[0][0]['label']+'-state')/'result.json')['controls']
    audited_photos(a.photos,a.base,controls);audited_images(a.images,a.base,controls,a.photos)
    manifest=a.output/'manifest.json'
    if a.resume:
        record=read(manifest);selection=read(a.output/'selection.json')
        assert record['selection_sha256']==sha(a.output/'selection.json')
        assert selection['photo_manifest_sha256']==sha(a.photos) and selection['image_manifest_sha256']==sha(a.images)
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        selection={'schema':1,'photo_manifest_sha256':sha(a.photos),'image_manifest_sha256':sha(a.images),
            'scope':'Same sixteen training identities at emission 1.0, selected before capturing or fitting; no validation identities.', 'cases':[]}
        for source,texture in sources:
            scene=a.output/(source['name']+'-scene');write_scene(scene,texture,emittance=1.)
            selection['cases'].append({'name':source['name'],'source_label':source['label'],
                'source_sha256':source['source_sha256'],'texture_sha256':source['texture_sha256'],
                'split':'train','emittance':1.,'label':'teacher-bright-'+source['name'],
                'scene_manifest_sha256':sha(scene/'manifest.json')})
        dump(a.output/'selection.json',selection)
        record={'schema':1,'complete':False,'selection_sha256':sha(a.output/'selection.json'),'cases':[]};dump(manifest,record)
    assert len(selection['cases'])==16 and len(record['cases'])<=16
    for index,(chosen,(source,_)) in enumerate(zip(selection['cases'],sources)):
        assert chosen['name']==source['name'] and chosen['source_sha256']==source['source_sha256']
        scene=a.output/(source['name']+'-scene');assert sha(scene/'manifest.json')==chosen['scene_manifest_sha256']
        if index>=len(record['cases']):
            # Each capture retains four fenced frames, about 200 MB. Leave room for training outputs.
            assert shutil.disk_usage(a.base).free > (16-index)*230_000_000+2_000_000_000, 'Insufficient free disk for bounded collection.'
            command=[sys.executable,str(Path(__file__).parent/'collect_demo_image.py'),'--demo-dir',str(a.demo_dir),
                '--output-dir',str(a.base),'--scene-dir',str(scene),'--capture-dll',str(a.capture_dll),
                '--capture-sha256',a.capture_sha256,'--label',chosen['label']]
            with (a.output/(source['name']+'-collector.log')).open('x') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            current,arrays,hashes,_=audit_capture(a.base,chosen['label']);del arrays
            assert current==controls
            record['cases'].append({**chosen,'capture_hashes':hashes});dump(manifest,record)
        else:
            current,arrays,hashes,_=audit_capture(a.base,chosen['label']);del arrays
            assert current==controls and hashes==record['cases'][index]['capture_hashes']
        print(json.dumps({'completed':chosen['name'],'count':index+1,'total':16}),flush=True)
    record['complete']=True;dump(manifest,record)
    audited_brightness(manifest,a.base,controls,a.photos,a.images)
    print('PASS: sixteen brighter training captures audited; sample state restored.',flush=True)


if __name__=='__main__':main()
