'''
为target目录下的几个文件制作一个toy的版本专门用来debug（包括partition.h5以及[set]_target.h5）
'''
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/'
toy_dir = os.path.join(target_dir, 'toy_version')

set_list = ['train', 'val', 'test']
partition_h5f = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
toy_partition = h5py.File(os.path.join(toy_dir, 'partition.h5'), 'w')
for set_name in set_list:
    target_h5f = h5py.File(os.path.join(target_dir, set_name + '_target.h5'), 'r')
    toy_target = h5py.File(os.path.join(toy_dir, set_name + '_target.h5'), 'w')
    video_ids = partition_h5f[set_name]['valid'][()]
    video_ids = [i.decode() for i in video_ids[:10]]
    partition_set_group = toy_partition.create_group(set_name)
    partition_set_group['valid'] = video_ids
    for video in tqdm(video_ids, desc=set_name):
        target_group = toy_target.create_group(video)
        elements = target_h5f[video].keys()
        for element in elements:
            target_group[element] = target_h5f[video][element][()]

