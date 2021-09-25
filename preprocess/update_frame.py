'''
从测试视频中以6Hz的频率抽帧，并保存（注意保存的图像还没有resize）
保存帧图像文件名称：00000x.jpg
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
import cv2

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

class Video2Frame(object):
    ''' 
    把视频帧抽出来存在文件夹中
    '''
    def __init__(self, fps=6, save_root='/data8/hzp/EEV/preprocess/EEV/test_data'):
        self.fps = fps
        self.save_root = save_root
    
    def __call__(self, video_path):
        basename = get_basename(video_path)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, self.fps, save_dir) + '%6d.jpg' + " > /dev/null 2>&1" #-r设置帧率，-q:v 图像质量, 2为保存为高质量，-f输出格式
        os.system(cmd)
        #frames_count = len(glob.glob(os.path.join(save_dir, '*.jpg')))
        # self.print('Extract frames from {}, totally {} frames, save to {}'.format(video_path, frames_count, save_dir))
        return save_dir

if __name__ == '__main__':
    dataset_root = '/data8/datasets/eevchallenges/'
    # set_names = ['train', 'val', 'test']
    save_root = '/data8/hzp/evoked_emotion/EEV_process_data/frames'
    # mkdir(save_root)
    # for set_name in set_names:

    new_video_list = ['CBn2M-3Safw', 'npGwn9_2MsA', 'WTyLlQzzsvE', 'XCfRIUOyMNQ', 'Yesf4wi8DaM']
    new_video_list = [i + '.mp4' for i in new_video_list]
    video_dir = os.path.join(dataset_root, 'test_videos')
    video_list = [os.path.join(video_dir, i) for i in new_video_list]
    get_frames = Video2Frame(save_root=save_root)
    for video_path in tqdm(video_list):
        _frame_dir = get_frames(video_path)

