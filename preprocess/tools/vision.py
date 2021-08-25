# import tensorflow as tf
# import collections
# from preprocess.tools.denseface.vision_network.models.dense_net import DenseNet

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

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

class Video2FrameTool(BaseWorker):
    def __init__(self, fps=10, logger=None):
        super().__init__(logger=logger)
        self.fps = fps
    
    def __call__(self, video_path, save_dir):
        if not(os.path.exists(save_dir)):
            mkdir(save_dir)
        cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, self.fps, save_dir) + '%4d.jpg' + " > /dev/null 2>&1"
        os.system(cmd)
        frames_count = len(glob.glob(os.path.join(save_dir, '*.jpg')))
        self.print('Extract frames from {}, totally {} frames, save to {}'.format(video_path, frames_count, save_dir))
        return save_dir

class DensefaceExtractor(BaseWorker):
    def __init__(self, mean=115.89650859267131, std=38.03799744727597, model_path=None, cfg=None, gpu_id=1):
        if cfg is None:
            cfg = model_cfg
        if model_path is None:
            model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.extractor = DenseNet(gpu_id, **cfg)
        self.extractor.to(self.device)
        state_dict = torch.load(model_path)
        self.extractor.load_state_dict(state_dict)
        self.extractor.eval()
        self.dim = 342
        self.mean = mean
        self.std = std
        
    def register_midlayer_hook(self, layer_names):
        self.ex_hook = MultiLayerFeatureExtractor(self.extractor, layer_names)
    
    def get_mid_layer_output(self):
        if getattr(self, 'ex_hook') is None:
            raise RuntimeError('Call register_midlayer_hook before calling get_mid_layer_output')
        return self.ex_hook.extract()
    
    def print_network(self):
        self.print(self.extractor)
    
    def __call__(self, img):
        print(img)
        if not isinstance(img, (np.ndarray, str)):
            raise ValueError('Input img parameter must be either str of img path or img np.ndarrays')
        if isinstance(img, np.ndarray):
            if img.shape == (64, 64):
                raise ValueError('Input img ndarray must have shape (64, 64), gray scale img')
        if isinstance(img, str):
            img_path = img
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if not isinstance(img, np.ndarray):
                    raise IOError(f'Warning: Error in {img_path}')
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (64, 64))
        
            else:
                feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
                return feat, np.ones([1, 8]) / 8
            
        # preprocess 
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, -1) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1

        # forward
        img = torch.from_numpy(img).to(self.device)
        self.extractor.set_input({"images": img})
        self.extractor.forward()
        ft, soft_label = self.extractor.out_ft, self.extractor.pred
        # a = soft_label.detach().cpu().numpy()[0].tolist()
        # print([f'{x:.4f}' for x in a])
        # print(['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con'])
        # input()
        return ft.detach().cpu().numpy(), soft_label.detach().cpu().numpy()


if __name__ == '__main__':
    face_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/data/raw/faces/1/232500.jpg'
    denseface = DensefaceExtractor()
    ft, label = denseface(face_path) #label不需要
    print(ft.shape)
    #print(label)
    print(ft)