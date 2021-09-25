import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

from tools.inception import InceptionExtractor

new_video_list = ['CBn2M-3Safw', 'npGwn9_2MsA', 'WTyLlQzzsvE', 'XCfRIUOyMNQ', 'Yesf4wi8DaM']

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


def get_inception_ft(all_videos, frame_root, video_id, gpu_id):
    inception = InceptionExtractor(gpu_id=gpu_id)
    assert (video_id in all_videos.keys())
    frame_dir = osp.join(frame_root, video_id)
    frame_list = sorted(os.listdir(frame_dir))
    num_files = len(frame_list)
    timestamps = all_videos[video_id]
    num_ts = len(timestamps)

    if num_ts <= num_files:
        frame_list = frame_list[:num_ts]
    video_fts = []
    for frame in frame_list:
        frame_path = osp.join(frame_dir, frame)
        ft = inception(frame_path) #返回的ft:(1, 2048)
        ft = np.squeeze(ft) #去掉冗余的维度
        video_fts.append(ft)
    if num_ts > num_files:
        frame_path = osp.join(frame_dir, frame_list[-1]) #重复最后一帧
        ft = inception(frame_path) #返回的ft:(1, 2048)
        ft = np.squeeze(ft) #去掉冗余的维度
        for i in range(num_ts - num_files):
            video_fts.append(ft)
    
    feature = np.array(video_fts, dtype=np.float32)
    return timestamps, feature



def make_inception_feature(target_dir, frame_root, save_dir, gpu_id):
    inception_h5f = h5py.File(osp.join(save_dir, 'update_inception.h5'), 'w')
    all_videos = get_timestamp(target_dir)
    inception_set_group = inception_h5f.create_group('test')
    for _id in tqdm(new_video_list):
        inception_group = inception_set_group.create_group(_id)
        timestamp, inception_ft = get_inception_ft(all_videos, frame_root, _id, gpu_id) #按照video_id号抽取特征
        inception_group['timestamp'] = timestamp
        inception_group['feature'] = inception_ft



def check(target_dir): #检查是否所有partition['valid']中出现的video都一定有其对应的标签
    all_videos = get_timestamp(target_dir)
    partition = h5py.File(osp.join(target_dir, 'partition.h5'), 'r')
    for set_name in ['train', 'val', 'test']:
        video_ids = partition[set_name]['valid']
        for video in tqdm(video_ids, desc=set_name):
            video = video.decode()
            if video not in all_videos.keys():
                print(video)
    print('-------finish checking-----------')


def check_image(frame_root): #检查是否所有frame下的目录中，所有图像的序号没有间断
    dir_list = os.listdir(frame_root)
    for dir in tqdm(dir_list):
        frame_list  = os.listdir(osp.join(frame_root, dir))
        frame_list = sorted(frame_list)
        num = len(frame_list)
        if num != int(frame_list[-1].split('.')[0]):
            print(dir)
    print('-------finish checking-----------')

def check_extracted(frame_root, target_dir): #检查是否所有有效的视频都抽出了图像
    extracted_video_list = os.listdir(frame_root)
    partition = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
    for set_name in ['train', 'val', 'test']:
        video_ids = partition[set_name]['valid']
        for video in video_ids:
            video = video.decode()
            if video not in extracted_video_list:
                print(video)





if __name__ == '__main__':
    frame_root = '/data8/hzp/evoked_emotion/EEV_process_data/frames'
    target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target'
    save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features'
    # mkdir(save_dir)

    # check(target_dir) #经检验，所有partition['valid']中出现的video都一定有其对应的标签
    # check_image(frame_root) #经检验，所有frame下的目录中，所有图像的序号没有间断

    # all_videos = get_timestamp(target_dir)
    # get_inception_ft(all_videos, frame_root, '-zC8-Jh2z9k')

    # check_extracted(frame_root, target_dir)

    print('making inception')
    make_inception_feature(target_dir, frame_root, save_dir, gpu_id=3)
