'''
得到：
/data8/hzp/evoked_emotion/EEV_process_data/target/invalid_test_target.h5：
结构：
/video_id （注意：所有无效的视频id）
/timestamp（一个视频中的所有时刻存成一个列表）
'''

import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

dataset_root = '/data8/datasets/eevchallenges/'
target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/'

partition = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')

set_name = 'test'
csv_path = os.path.join(dataset_root, set_name + '.csv')
df = pd.read_csv(csv_path)
all_video_id = np.array(df['Video ID'])
all_timestamp = np.array(df['Timestamp (milliseconds)'])

num_data = len(all_video_id)

target_path = os.path.join(target_dir, 'invalid_test_target.h5')
target_h5f = h5py.File(target_path, 'w')

cur_vid = all_video_id[0]
s_idx = 0
e_idx = 0

invalid_videos = [i.decode() for i in partition[set_name]['invalid']]

for e_idx in tqdm(range(num_data), desc=set_name):
    if all_video_id[e_idx] != cur_vid: #准备向h5文件中添加一个video的信息
        if cur_vid in invalid_videos: #只有当video是无效的时，才将该video的信息记录进h5文件
            video_group = target_h5f.create_group(cur_vid)
            video_group['timestamp'] = all_timestamp[s_idx: e_idx]

        cur_vid = all_video_id[e_idx]
        s_idx = e_idx

#最后一个video也不要落下
if cur_vid in invalid_videos: #只有当video是无效的时，才将该video的信息记录进h5文件
    video_group = target_h5f.create_group(cur_vid)
    video_group['timestamp'] = all_timestamp[s_idx:]

