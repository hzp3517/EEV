import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

features_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'
train_h5f = h5py.File(os.path.join(features_dir, 'trill_distilled_train.h5'), 'r')
val_h5f = h5py.File(os.path.join(features_dir, 'trill_distilled_val.h5'), 'r')
test_h5f = h5py.File(os.path.join(features_dir, 'trill_distilled_test.h5'), 'r')
final_h5f = h5py.File(os.path.join(features_dir, 'trill_distilled_final.h5'), 'w')

set_name = 'train'
trill_set_group = final_h5f.create_group(set_name)
video_ids = train_h5f[set_name].keys()
for video_id in tqdm(video_ids, desc=set_name):
    trill_group = trill_set_group.create_group(video_id)
    trill_group['timestamp'] = train_h5f[set_name][video_id]['timestamp'][()]
    trill_group['feature'] = train_h5f[set_name][video_id]['feature'][()]

set_name = 'val'
trill_set_group = final_h5f.create_group(set_name)
video_ids = val_h5f[set_name].keys()
for video_id in tqdm(video_ids, desc=set_name):
    trill_group = trill_set_group.create_group(video_id)
    trill_group['timestamp'] = val_h5f[set_name][video_id]['timestamp'][()]
    trill_group['feature'] = val_h5f[set_name][video_id]['feature'][()]

set_name = 'test'
trill_set_group = final_h5f.create_group(set_name)
video_ids = test_h5f[set_name].keys()
for video_id in tqdm(video_ids, desc=set_name):
    trill_group = trill_set_group.create_group(video_id)
    trill_group['timestamp'] = test_h5f[set_name][video_id]['timestamp'][()]
    trill_group['feature'] = test_h5f[set_name][video_id]['feature'][()]