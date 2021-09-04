import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

from tools.vggish import VggishExtractor

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

def get_timestamp(target_dir): #一次性将所有集合的所有视频的timestamp信息读入
    '''
    video_id: [timestamp_list]
    '''
    all_videos = {}
    set_name_list = ['train', 'val', 'test']
    for set_name in set_name_list:
        target_path = osp.join(target_dir, set_name + '_target.h5')
        h5f = h5py.File(target_path, 'r')
        for video in h5f:
            all_videos[video] = np.array(h5f[video]['timestamp'])
    return all_videos



def get_vggish_ft(all_videos, audio_root, video_id, gpu_id):
    vggish = VggishExtractor(gpu_id=gpu_id)
    audio_path = osp.join(audio_root, video_id + '.wav')
    ft = vggish(audio_path)
    timestamps = all_videos[video_id]
    num_ts = len(timestamps)

    if num_ts <= len(ft):
        ft = ft[:num_ts]
    else:
        last_ft = ft[-1]
        pad_ft = []
        for i in range(num_ts - len(ft)):
            pad_ft.append(last_ft)
        pad_ft = np.array(pad_ft)
        ft = np.concatenate((ft, pad_ft), axis=0)

    ft = ft.astype(np.float32)
    return timestamps, ft



def make_vggish_feature(target_dir, audio_root, save_dir, gpu_id):
    partition = h5py.File(osp.join(target_dir, 'partition.h5'), 'r')
    vggish_h5f = h5py.File(osp.join(save_dir, 'vggish.h5'), 'w')
    all_videos = get_timestamp(target_dir)
    for set_name in ['train', 'val', 'test']:
        vggish_set_group = vggish_h5f.create_group(set_name)
        video_ids = partition[set_name]['valid']
        for _id in tqdm(video_ids, desc=set_name):
            _id = _id.decode()
            vggish_group = vggish_set_group.create_group(_id)
            timestamp, vggish_ft = get_vggish_ft(all_videos, audio_root, _id, gpu_id) #按照video_id号抽取特征
            vggish_group['timestamp'] = timestamp
            vggish_group['feature'] = vggish_ft


def check_extracted(audio_root, target_dir): #检查是否所有有效的视频都抽出了音频
    extracted_video_list = [i.split('.')[0] for i in os.listdir(audio_root)]
    partition = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
    for set_name in ['train', 'val', 'test']:
        video_ids = partition[set_name]['valid']
        for video in video_ids:
            video = video.decode()
            if video not in extracted_video_list:
                print(video)


if __name__ == '__main__':
    audio_root = '/data8/hzp/evoked_emotion/EEV_process_data/audios'
    target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target'
    save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features'
    mkdir(save_dir)

    # check_extracted(audio_root, target_dir)

    print('making vggish')
    make_vggish_feature(target_dir, audio_root, save_dir, gpu_id=2)