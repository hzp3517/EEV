'''
得到三个集合作为标签的文件：
/data8/hzp/evoked_emotion/EEV_process_data/target/[set]_target.h5：
结构：
/video_id （注意：原文件中每个video需要先确认是否是有效的，只保存有效视频的数据）
/timestamp  /[15种类别]  /valid （均为列表，一个视频中的所有时刻存成一个列表。
                            注意原文件中15个类全0的标签需要在/valid 目录中专门记录当前时刻的标注是否有效，True表示有效，False表示无效）
注：test_target.h5中只有/timestamp
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
set_names = ['train', 'val', 'test']

partition = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')

for set_name in set_names:
    csv_path = os.path.join(dataset_root, set_name + '.csv')
    df = pd.read_csv(csv_path)
    if set_name == 'train' or set_name == 'val':
        all_video_id = np.array(df['YouTube ID'])
        all_label = np.array(df.values)[:, 2:] #15个类的标注值
        all_label = np.array(all_label, dtype=np.float32)
        all_valid = np.array([np.any(i) for i in all_label]) #记录当前时刻的标注是否有效，True表示有效，False表示无效
    else:
        all_video_id = np.array(df['Video ID'])
    all_timestamp = np.array(df['Timestamp (milliseconds)'])


    category_list = ['amusement', 'anger', 'awe', 'concentration', 'confusion', 'contempt', 'contentment', 'disappointment',
                        'doubt', 'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph']
    num_data = len(all_video_id)

    target_path = os.path.join(target_dir, set_name + '_target.h5')
    target_h5f = h5py.File(target_path, 'w')

    cur_vid = all_video_id[0]
    s_idx = 0
    e_idx = 0

    valid_videos = [i.decode() for i in partition[set_name]['valid']]

    for e_idx in tqdm(range(num_data), desc=set_name):
        if all_video_id[e_idx] != cur_vid: #准备向h5文件中添加一个video的信息
            if cur_vid in valid_videos: #只有当video是有效的时，才将该video的信息记录进h5文件
                video_group = target_h5f.create_group(cur_vid)
                video_group['timestamp'] = all_timestamp[s_idx: e_idx]
                if set_name == 'train' or set_name == 'val':
                    video_group['valid'] = all_valid[s_idx: e_idx]
                    for cate_idx in range(len(category_list)):
                        video_group[category_list[cate_idx]] = all_label[s_idx: e_idx, cate_idx].squeeze()

            cur_vid = all_video_id[e_idx]
            s_idx = e_idx

    #最后一个video也不要落下
    if cur_vid in valid_videos: #只有当video是有效的时，才将该video的信息记录进h5文件
        video_group = target_h5f.create_group(cur_vid)
        video_group['timestamp'] = all_timestamp[s_idx:]
        if set_name == 'train' or set_name == 'val':
            video_group['valid'] = all_valid[s_idx:]
            for cate_idx in range(len(category_list)):
                video_group[category_list[cate_idx]] = all_label[s_idx: e_idx, cate_idx].squeeze()
