'''
为features目录下的几个文件制作一个toy的版本专门用来debug（包括vggish.h5以及inception.h5）
其中数据与target中的toy文件一致
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

features_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'
toy_dir = os.path.join(features_dir, 'toy_version')
mkdir(toy_dir)
toy_target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/toy_version/'

set_list = ['train', 'val', 'test']

for feature in ['vggish', 'inception']:
    origin_feature = h5py.File(os.path.join(features_dir, feature + '.h5'), 'r')
    toy_feature = h5py.File(os.path.join(toy_dir, feature + '.h5'), 'w')
    for set_name in set_list:
        toy_target = h5py.File(os.path.join(toy_target_dir, set_name + '_target.h5'), 'r')
        video_list = list(toy_target.keys())
        toy_feature_set = toy_feature.create_group(set_name)
        for video in tqdm(video_list, desc=set_name):
            feature_group = toy_feature_set.create_group(video)
            elements = origin_feature[set_name][video].keys()
            for element in elements:
                feature_group[element] = origin_feature[set_name][video][element][()]