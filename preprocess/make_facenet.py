import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

from tools.facenet import FacenetExtractor

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

def get_timestamp(target_dir): #一次性将所有集合的所有视频的timestamp信息读入
    '''
    {video_id: [timestamp_list]}
    '''
    all_videos = {}
    set_name_list = ['train', 'val', 'test']
    for set_name in set_name_list:
        target_path = osp.join(target_dir, set_name + '_target.h5')
        h5f = h5py.File(target_path, 'r')
        for video in h5f:
            all_videos[video] = np.array(h5f[video]['timestamp'])
    return all_videos



def get_facenet_ft(all_videos, audio_root, video_id, gpu_id):
    facenet = FacenetExtractor(gpu_id)
    frame_dir = osp.join(frame_root, video_id)
    frame_list = sorted(os.listdir(frame_dir))
    num_files = len(frame_list)
    timestamps = all_videos[video_id]
    num_ts = len(timestamps)

    if num_ts <= num_files:
        frame_list = frame_list[:num_ts]
    video_fts = []
    for frame in frame_list:
    # for frame in tqdm(frame_list):#
        frame_path = osp.join(frame_dir, frame)
        ft = facenet(frame_path) #返回的ft:(1024,)
        video_fts.append(ft)
    if num_ts > num_files: #如果最后缺帧，直接补0
        zero_ft = np.zeros((1024,))
        for i in range(num_ts - num_files):
            video_fts.append(zero_ft)
    
    #feature = np.array(video_fts)
    feature = np.array(video_fts, dtype=np.float32)
    return timestamps, feature


def make_facenet_feature(target_dir, audio_root, save_dir, gpu_id):
    partition = h5py.File(osp.join(target_dir, 'partition.h5'), 'r')
    facenet_h5f = h5py.File(osp.join(save_dir, 'facenet.h5'), 'w')
    all_videos = get_timestamp(target_dir)
    for set_name in ['train', 'val', 'test']:
        facenet_set_group = facenet_h5f.create_group(set_name)
        video_ids = partition[set_name]['valid']
        for _id in tqdm(video_ids, desc=set_name):
            _id = _id.decode()
            facenet_group = facenet_set_group.create_group(_id)
            timestamp, facenet_ft = get_facenet_ft(all_videos, audio_root, _id, gpu_id) #按照video_id号抽取特征
            facenet_group['timestamp'] = timestamp
            facenet_group['feature'] = facenet_ft



if __name__ == '__main__':
    frame_root = '/data8/hzp/evoked_emotion/EEV_process_data/frames'
    target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target'
    save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features'
    mkdir(save_dir)

    print('making facenet')
    make_facenet_feature(target_dir, frame_root, save_dir, gpu_id=7)