# SPDX-License-Identifier: Apache-2.0
"""Revalidate the fixed 46-frame training set for training-only diagnostics."""
import hashlib
import json
from pathlib import Path

import torch

from collect_demo_photo_training import audited_photos
from collect_diverse_images import audited_images
from compare_output_grade import read_capture
from train_student_collection import original_training_names


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_pairs(base, photos, images, record):
    assert record['native_shape']==[1080,1920] and record['data_split']['training_view_count']==46
    expected={r['color']:r for r in record['training_capture_hashes']}
    validation={record['validation_capture_hashes']['color']}|{r['capture_hashes']['color'] for r in record['extra_validation']}
    assert len(expected)==46 and len(validation)==16 and not expected.keys()&validation
    collection=read(base/'teacher-illumination-manifest.json')
    assert collection['scene_restored'] and collection['dll_restored']
    labels=original_training_names()+[r['label'] for r in collection['views'] if r['split']=='train']
    assert len(labels)==30 and all(Path(n).name==n for n in labels)
    candidates=[(n,'scene',base/'trials'/n/'capture') for n in labels]
    rows=audited_photos(photos,base,record['controls'])+audited_images(images,base,record['controls'],photos)
    candidates += [(r['label'],'photos',base/(r['label']+'-state')/'capture') for r in rows if r['split']=='train']
    pairs=[]
    for label,group,path in candidates:
        controls,arrays,hashes=read_capture(path)
        assert controls==record['controls'] and hashes==expected[hashes['color']]
        pairs.append({'label':label,'group':group,'path':path,'capture_hashes':hashes})
        del arrays
    assert len(pairs)==46 and {p['capture_hashes']['color'] for p in pairs}==set(expected)
    return pairs


def load_pair(pair, controls):
    current,arrays,hashes=read_capture(pair['path'])
    assert current==controls and hashes==pair['capture_hashes']
    return [torch.from_numpy(arrays[k]).permute(2,0,1).unsqueeze(0).cuda().float().contiguous(memory_format=torch.channels_last)
            for k in ['color','output']]


def rgb_loss(predicted,target):
    pixel=(predicted-target).abs().mean()
    detail=((predicted[:,:,:,1:]-predicted[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1])).abs().mean()
    detail+=((predicted[:,:,1:,:]-predicted[:,:,:-1,:])-(target[:,:,1:,:]-target[:,:,:-1,:])).abs().mean()
    return pixel+.25*detail,pixel
