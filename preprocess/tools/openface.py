import os, glob
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append('/data12/MUSE2021/preprocess/')#

from tools.base_worker import BaseWorker

import math
from tools.denseface.config.dense_fer import model_cfg
from tools.denseface.model.dense_net import DenseNet
from tools.hook import MultiLayerFeatureExtractor

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


class OpenFaceTracker(BaseWorker):
    ''' 使用openface工具抽取人脸
        eg: 输入视频帧位置: "/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/data/raw/faces/1"（注意最后不要加斜杠）
            输出人脸图片位置：save_root = '/data12/MUSE2021/preprocess/tools/.tmp/openface_save/1'
            其中openface_save/1/1.csv文件是我们需要的，包含人脸关键点和AU等信息
    '''
    def __init__(self, save_root='/data12/MUSE2021/preprocess/tools/.tmp/openface_save/',
            openface_dir='/root/tools/OpenFace/build/bin', logger=None): #leo_hzp
        super().__init__(logger=logger)
        self.save_root = save_root
        self.openface_dir = openface_dir
    
    def __call__(self, frames_dir):
        basename = get_basename(frames_dir)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = '{}/FeatureExtraction -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                    self.openface_dir, frames_dir, save_dir
                )
        os.system(cmd)
        return save_dir


# class VideoFaceTracker(BaseWorker):
#     ''' 使用openface工具抽取人脸
#         eg: 输入视频帧位置: "./test/frame/hahah/0", 人脸图片位置save_root = './test/face'
#             输出人脸位置: "./test/face/hahah/0 (按需求可修改)
#             其中./test/face/hahah/0/0_aligned文件夹中包含人脸图片
#             ./test/face/hahah/0/0.csv中包含人脸关键点和AU等信息
#     '''
#     def __init__(self, save_root='test/track',
#             openface_dir='/root/tools/OpenFace/build/bin', logger=None):
#         #openface_dir='/root/tools/openface_tool/OpenFace/build/bin', logger=None):
#         super().__init__(logger=logger)
#         self.save_root = save_root
#         self.openface_dir = openface_dir
    
#     def __call__(self, frames_dir):
#         basename = get_basename(frames_dir)
#         basename = os.path.join(frames_dir.split('/')[-2], basename)
#         save_dir = os.path.join(self.save_root, basename)
#         mkdir(save_dir)
#         cmd = '{}/FaceLandmarkVidMulti -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
#                     self.openface_dir, frames_dir, save_dir
#                 )
#         os.system(cmd)
#         # self.print('Face Track in {}, result save to {}'.format(frames_dir, save_dir))
#         return save_dir

if __name__ == '__main__':
    face_dir = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/data/raw/renamed_faces/1'
    openface = OpenFaceTracker(save_root='/data12/MUSE2021/preprocess/tools/.tmp/openface_save_test/')
    save_dir = openface(face_dir)