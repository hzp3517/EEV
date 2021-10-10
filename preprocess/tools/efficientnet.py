'''
得到EfficientNet-B0特征
'''

import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms

sys.path.append('/data8/hzp/evoked_emotion/EEV/preprocess')
from tools.base_worker import BaseWorker

class EfficientNetExtractor(BaseWorker):
    def __init__(self, gpu_id=0):
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = EfficientNet.from_pretrained('efficientnet-b0') #b0~b7均可使用，但对输入图像尺寸的要求不同
        self.model.to(self.device) #将网络放到指定的设备上
        self.model.eval()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling.to(self.device) #将网络放到指定的设备上
        self.avg_pooling.eval()
        self.input_size = 224 # b0对应的输入图像尺寸
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def preprocess_official(self, img_path):
        '''
        https://github.com/lukemelas/EfficientNet-PyTorch 官方代码中的图像预处理方法
        只做了归一化，未做padding。
        '''
        tfms = transforms.Compose([transforms.Resize([self.input_size, self.input_size]), transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),]) #RGB
        img = tfms(Image.open(img_path)).unsqueeze(0)
        return img

    def preprocess_Huynh(self, img_path):
        '''
        1st 代码中的图像预处理方法（原代码中无normalize）
        做了padding以保留非padding部分的长宽比。
        '''
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if h < w:
            pad_h = w - h
            pad_w = 0
        else:
            pad_w = h - w
            pad_h = 0

        img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                    mode='constant')
        # np.pad(array, pad_width, mode, **kwargs)：
        #   pad_width：每个轴边缘需要填充的数值数目。参数输入方式为((a1, b1), ..., (an, bn))，其中(a1, b1)表示第1轴两边分别填充a1和b1个数值。
        #   mode：填充的方式。'constant'表示连续填充相同的值，默认每个轴都填0。

        assert (img.shape[0] == img.shape[1] and img.shape[2] == 3)

        img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #(224, 224, 3)
        # img = img[:, :, ::-1].copy() # 把颜色顺序从BGR转为RGB

        img = img / 255.0 #将像素值的范围缩到[0, 1]

        # # normalize。如果把pad的0像素和图像本身的0像素视作一样的，这样写即可，否则需要对pad的部分赋予mean。
        img = (img - self.mean) / self.std # 原代码中没有做normalize

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        
        return img #(1, 3, 224, 224)



    def __call__(self, img_list):
        # if not isinstance(img, (np.ndarray, str)):
        #     raise ValueError('Input img parameter must be either str of img path or img np.ndarrays')
        # if isinstance(img, np.ndarray):
        #     if img.shape == (self.input_size, self.input_size, 3):
        #         raise ValueError('Input img ndarray must have shape ({}, {}, 3)'.format(str(self.input_size), str(self.input_size)))

        # 图像预处理
        imgs = []
        for img in img_list:
            # img = self.preprocess_official(img)
            img = self.preprocess_Huynh(img)
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0).float().to(self.device) # (bs, 3, 224, 224)

        features = self.model.extract_features(imgs) # shape: (bs, 1280, 7, 7)
        # pytorch的efficientnet实现中，foward()函数中将得到的features做avg pooling，然后过dropout和fc得到要返回的logits
        features = self.avg_pooling(features) # shape: (bs, 1280, 1, 1)
        features = features.squeeze() # shape: (bs, 1280)

        return features.detach().cpu().numpy()





if __name__ == '__main__':
    img_path_1 = '/data8/hzp/evoked_emotion/EEV_process_data/frames/-Dzh3EhJbBg/000001.jpg'
    img_path_2 = '/data8/hzp/evoked_emotion/EEV_process_data/frames/-Dzh3EhJbBg/000002.jpg'
    img_path_list = [img_path_1, img_path_2] #一个batch的图像列表
    efficientnet = EfficientNetExtractor(gpu_id=1)
    fts = efficientnet(img_path_list)
    print(fts.shape) # shape: (bs, 1280)
    print(fts)