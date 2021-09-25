'''
从测试视频中以16000Hz的频率抽音频，并保存
保存语音文件名称：[video_id].wav
'''

import numpy as np
import json
import os
#import istarmap
from shutil import copyfile, copytree
from sys import exit
import sys
import csv
from tqdm import tqdm
import multiprocessing
import glob
import time

def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class AudioSplitor(object):
    ''' 把语音从视频中抽出来存在文件夹中
        eg: 输入视频/root/hahah/0.mp4, save_root='./test/audio'
            输出音频位置: ./test/audio/hahah/0.wav (注意第24行, 按需求可修改)
            保存的采样率是16000, 16bit, 如需修改请参考30行: _cmd = "ffmpeg -i ...."
    '''
    def __init__(self, save_root):
        self.audio_dir = save_root

    def __call__(self, video_path):
        basename = get_basename(video_path)
        save_path = os.path.join(self.audio_dir, basename + '.wav')
        if not os.path.exists(save_path):
            _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(video_path, save_path)
            os.system(_cmd)
        return save_path

if __name__ == '__main__':
    dataset_root = '/data8/datasets/eevchallenges/'
    # set_names = ['train', 'val', 'test']
    save_root = '/data8/hzp/evoked_emotion/EEV_process_data/audios'
    new_video_list = ['CBn2M-3Safw', 'npGwn9_2MsA', 'WTyLlQzzsvE', 'XCfRIUOyMNQ', 'Yesf4wi8DaM']
    new_video_list = [i + '.mp4' for i in new_video_list]
    video_dir = os.path.join(dataset_root, 'test_videos')
    video_list = [os.path.join(video_dir, i) for i in new_video_list]
    extract_audio = AudioSplitor(save_root=save_root)
    for video_path in tqdm(video_list):
        _ = extract_audio(video_path)

