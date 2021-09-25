import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

features_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'
origin_h5f = h5py.File(os.path.join(osp.join(features_dir, 'backup_9.15'), 'inception.h5'), 'r')
add_h5f = h5py.File(os.path.join(features_dir, 'update_inception.h5'), 'r')
final_h5f = h5py.File(os.path.join(features_dir, 'inception.h5'), 'w')

for set_name in ['train', 'val']:
    inception_set_group = final_h5f.create_group(set_name)
    video_ids = origin_h5f[set_name].keys()
    for video_id in tqdm(video_ids, desc=set_name):
        inception_group = inception_set_group.create_group(video_id)
        inception_group['timestamp'] = origin_h5f[set_name][video_id]['timestamp'][()]
        inception_group['feature'] = origin_h5f[set_name][video_id]['feature'][()]

set_name = 'test'
inception_set_group = final_h5f.create_group(set_name)
video_ids = origin_h5f[set_name].keys()
for video_id in tqdm(video_ids, desc=set_name):
    inception_group = inception_set_group.create_group(video_id)
    inception_group['timestamp'] = origin_h5f[set_name][video_id]['timestamp'][()]
    inception_group['feature'] = origin_h5f[set_name][video_id]['feature'][()]
video_ids = add_h5f[set_name].keys()
for video_id in tqdm(video_ids, desc=set_name):
    inception_group = inception_set_group.create_group(video_id)
    inception_group['timestamp'] = add_h5f[set_name][video_id]['timestamp'][()]
    inception_group['feature'] = add_h5f[set_name][video_id]['feature'][()]



