'''
根据/data8/datasets/eevchallenges/[set]_videos/中所有视频的名称，列出一个对应集合的所有有效视频和无效视频的列表，
    以h5格式存在/data8/hzp/evoked_emotion/EEV_process_data/target/中。
    格式：
    /partition.h5
    /train  /val  /test
    每个group下都有：/valid：所有有效视频名称列表  /invalid：所有无效视频名称列表
在这个过程中需要逐一检查每个名称是否已经在/data8/datasets/eevchallenges/[set]_video_names.txt中出现过。
'''

import os
import h5py 
import numpy as np

dataset_root = '/data8/datasets/eevchallenges/'
save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target'
set_names = ['train', 'val', 'test']

partition_path = os.path.join(save_dir, 'partition.h5')
partition_h5f = h5py.File(partition_path, 'w')

for set_name in set_names:
    video_dir = os.path.join(dataset_root, set_name + '_videos')
    entire_names_file_path = os.path.join(dataset_root, set_name + '_video_names.txt')

    valid_names = [i[:-4] for i in os.listdir(video_dir)]

    with open(entire_names_file_path, 'r') as f:
        entire_names = [i.strip() for i in f.readlines()]

    # #找出多余的视频：（[set]_video_names.txt中不存在该id的视频）
    # print('set: ', set_name)
    # for name in valid_names:
    #     if name not in entire_names:
    #         print(name)
    # print('\n')

    valid_videos = []
    invalid_videos = []

    for name in entire_names:
        if name in valid_names:
            valid_videos.append(name)
        else:
            invalid_videos.append(name)

    set_group = partition_h5f.create_group(set_name)
    set_group['valid'] = valid_videos
    set_group['invalid'] = invalid_videos


    


    
