import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import tensorflow as tf

from tools.trill_distilled import TrillDistilledExtractor

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



# def get_trill_distilled_ft(all_videos, audio_root, video_id, batch_size, gpu_id):
def get_trill_distilled_ft(all_videos, audio_root, video_id, batch_size):
    # trill_distilled = TrillDistilledExtractor(seg_len=1/6, gpu_id=gpu_id) #标签timestamp是6Hz
    trill_distilled = TrillDistilledExtractor(seg_len=1/6) #标签timestamp是6Hz
    audio_path = osp.join(audio_root, video_id + '.wav')
    ft = trill_distilled(audio_path, batch_size=batch_size)
    timestamps = all_videos[video_id]
    num_ts = len(timestamps)
    if ft.shape[0] >= num_ts:
        ft = ft[:num_ts]
    else:
        pad_ft = []
        for _ in range(num_ts - ft.shape[0]):
            pad_ft.append(ft[-1])
        pad_ft = np.stack(pad_ft)
        ft = np.concatenate((ft, pad_ft), axis=0)
    ft = ft.astype(np.float32)
    return timestamps, ft



# def make_trill_distilled_feature(target_dir, audio_root, save_dir, batch_size, gpu_id):
def make_trill_distilled_feature(target_dir, audio_root, save_dir, batch_size):
    partition = h5py.File(osp.join(target_dir, 'partition.h5'), 'r')
    # trill_distilled_h5f = h5py.File(osp.join(save_dir, 'trill_distilled.h5'), 'w')
    trill_distilled_h5f = h5py.File(osp.join(save_dir, 'trill_distilled_train.h5'), 'w')


    all_videos = get_timestamp(target_dir)
    # for set_name in ['train', 'val', 'test']:
    for set_name in ['train']:
        trill_distilled_set_group = trill_distilled_h5f.create_group(set_name)
        video_ids = partition[set_name]['valid']


        # video_ids = video_ids[2500:] #


        for _id in tqdm(video_ids, desc=set_name):
            _id = _id.decode()
            trill_distilled_group = trill_distilled_set_group.create_group(_id)
            # timestamp, trill_distilled_ft = get_trill_distilled_ft(all_videos, audio_root, _id, batch_size, gpu_id) #按照video_id号抽取特征
            timestamp, trill_distilled_ft = get_trill_distilled_ft(all_videos, audio_root, _id, batch_size) #按照video_id号抽取特征
            trill_distilled_group['timestamp'] = timestamp
            trill_distilled_group['feature'] = trill_distilled_ft

            del timestamp
            del trill_distilled_ft


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

    gpu_id = 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) #默认情况下，TensorFlow会占用所有GPUs的所有GPU内存（取决于CUDA_VISIBLE_DEVICES这个系统变量）

    os.environ['CUDA_VISIBLE_DEVICES'] = ' ' #默认情况下，TensorFlow会占用所有GPUs的所有GPU内存（取决于CUDA_VISIBLE_DEVICES这个系统变量）

    physical_devices = tf.config.list_physical_devices('GPU') # 当前可用的gpu列表
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True) # 将GPU的显存使用策略设置为“仅在需要时申请显存空间”

    print('making trill_distilled')
    # make_trill_distilled_feature(target_dir, audio_root, save_dir, batch_size=32, gpu_id=1)
    make_trill_distilled_feature(target_dir, audio_root, save_dir, batch_size=16)