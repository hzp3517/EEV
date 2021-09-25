'''
为target目录下的几个文件制作一个subset的版本专门用来调参（包括partition.h5以及[set]_target.h5）
'''
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv
import random

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/'
subset_dir = os.path.join(target_dir, 'subset_version')
mkdir(subset_dir)

set_list = ['train', 'val', 'test']
subset_num_dic = {'train': 202, 'val': 50, 'test': 88}

# #先统计一下各个set的video数量
# partition_h5f = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
# for set_name in set_list:
#     video_ids = partition_h5f[set_name]['valid'][()]
#     print(set_name, ':', len(video_ids))

partition_h5f = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
subset_partition = h5py.File(os.path.join(subset_dir, 'partition.h5'), 'w')
for set_name in set_list:
    target_h5f = h5py.File(os.path.join(target_dir, set_name + '_target.h5'), 'r')
    subset_target = h5py.File(os.path.join(subset_dir, set_name + '_target.h5'), 'w')
    video_ids = partition_h5f[set_name]['valid'][()]
    video_ids = [i.decode() for i in video_ids]
    video_ids = random.sample(video_ids, subset_num_dic[set_name])
    partition_set_group = subset_partition.create_group(set_name)
    partition_set_group['valid'] = video_ids
    for video in tqdm(video_ids, desc=set_name):
        target_group = subset_target.create_group(video)
        elements = target_h5f[video].keys()
        for element in elements:
            target_group[element] = target_h5f[video][element][()]
