import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

features_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'
origin_h5f = h5py.File(os.path.join(features_dir, 'vggish.h5'), 'r')
test_h5f = h5py.File(os.path.join(features_dir, 'vggish_test.h5'), 'r')
final_h5f = h5py.File(os.path.join(features_dir, 'vggish_final.h5'), 'w')


for set_name in ['train', 'val']:
    vggish_set_group = final_h5f.create_group(set_name)
    video_ids = origin_h5f[set_name].keys()
    for video_id in tqdm(video_ids, desc=set_name):
        vggish_group = vggish_set_group.create_group(video_id)
        vggish_group['timestamp'] = origin_h5f[set_name][video_id]['timestamp'][()]
        vggish_group['feature'] = origin_h5f[set_name][video_id]['feature'][()]

set_name = 'test'
vggish_set_group = final_h5f.create_group(set_name)
video_ids = test_h5f[set_name]
for video_id in tqdm(video_ids, desc=set_name):
    vggish_group = vggish_set_group.create_group(video_id)
    vggish_group['timestamp'] = test_h5f[set_name][video_id]['timestamp'][()]
    vggish_group['feature'] = test_h5f[set_name][video_id]['feature'][()]


