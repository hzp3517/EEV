import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

from tools.efficientnet import EfficientNetExtractor

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

def chunk(lst, chunk_size): # chunk的作用：配合循环使用，每n个元素截断一次形成一个列表
    idx = 0
    while chunk_size * idx < len(lst):
        yield lst[idx*chunk_size: (idx+1)*chunk_size]
        idx += 1

def get_timestamp(target_dir): # 一次性将所有集合的所有视频的timestamp信息读入
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


def get_efficientnet_ft(all_videos, frame_root, video_id, batch_size, gpu_id):
    efficientnet = EfficientNetExtractor(gpu_id=gpu_id)
    assert (video_id in all_videos.keys())
    frame_dir = osp.join(frame_root, video_id)
    frame_list = sorted(os.listdir(frame_dir))
    num_files = len(frame_list)
    timestamps = all_videos[video_id]
    num_ts = len(timestamps)

    if num_ts <= num_files:
        frame_list = frame_list[:num_ts]
    else:
        last_frame = frame_list[-1]
        for _ in range(num_ts - num_files):
            frame_list.append(last_frame)

    video_fts = []
    for frame_batch in chunk(frame_list, batch_size):
    # for frame_batch in tqdm(chunk(frame_list, batch_size)):
        frame_paths = [osp.join(frame_dir, i) for i in frame_batch]
        fts = efficientnet(frame_paths) # 返回的fts: (bs, 1280)
        if len(fts.shape) == 1 and fts.shape[0] == 1280: #恰好1个元素的情况，返回的fts的shape是(1280,)
            fts = np.expand_dims(fts, axis=0) # (1, 1280)
        video_fts.append(fts)
    video_fts = np.concatenate(video_fts, axis=0)
    video_fts = video_fts.astype(np.float32)
    
    return timestamps, video_fts



def make_efficientnet_feature(target_dir, frame_root, save_dir, batch_size, gpu_id):
    partition = h5py.File(osp.join(target_dir, 'partition.h5'), 'r')
    efficientnet_h5f = h5py.File(osp.join(save_dir, 'efficientnet.h5'), 'w')
    all_videos = get_timestamp(target_dir)
    for set_name in ['train', 'val', 'test']:
        efficientnet_set_group = efficientnet_h5f.create_group(set_name)
        video_ids = partition[set_name]['valid']
        for _id in tqdm(video_ids, desc=set_name):
            _id = _id.decode()
            efficientnet_group = efficientnet_set_group.create_group(_id)
            timestamp, efficientnet_ft = get_efficientnet_ft(all_videos, frame_root, _id, batch_size, gpu_id) #按照video_id号抽取特征
            efficientnet_group['timestamp'] = timestamp
            efficientnet_group['feature'] = efficientnet_ft





if __name__ == '__main__':
    frame_root = '/data8/hzp/evoked_emotion/EEV_process_data/frames'
    target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target'
    save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features'
    mkdir(save_dir)

    # all_videos = get_timestamp(target_dir)
    # timestamps, video_fts = get_efficientnet_ft(all_videos, frame_root, '-zC8-Jh2z9k', batch_size=64, gpu_id=6)
    # print(video_fts.shape)
    # print(video_fts)

    print('making efficientnet')
    make_efficientnet_feature(target_dir, frame_root, save_dir, batch_size=64, gpu_id=6)