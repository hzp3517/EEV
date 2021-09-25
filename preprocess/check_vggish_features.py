'''
新抽的vggish特征里面好像少了一些视频的特征，需要check一下特征文件中的所有video名称是否与partition中所有valid的名称一致
'''

import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv
import math

set_list = ['train', 'val', 'test']
# set_list = ['test']
ft_path = '/data8/hzp/evoked_emotion/EEV_process_data/features/vggish.h5'
# ft_path = '/data8/hzp/evoked_emotion/EEV_process_data/features/vggish_test.h5'
partition_path = '/data8/hzp/evoked_emotion/EEV_process_data/target/partition.h5'
ft_h5f = h5py.File(ft_path, 'r')
par_h5f = h5py.File(partition_path, 'r')
for set_name in set_list:
    ft_vids = list(ft_h5f[set_name].keys())
    par_vids = [i.decode() for i in par_h5f[set_name]['valid']]
    # print(ft_vids)
    # print(par_vids)
    for vid in par_vids:
        if vid not in ft_vids:
            print(vid)
    print('---------------------')

