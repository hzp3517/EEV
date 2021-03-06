'''
得到inception-v3特征
'''

import cv2
import numpy as np
import os
import sys
import glob
import torch
from PIL import Image
from torchvision import transforms

sys.path.append('/data8/hzp/evoked_emotion/EEV/preprocess')
from tools.inception_v3.inception_net import inception_v3
from tools.base_worker import BaseWorker


class InceptionExtractor(BaseWorker):
    def __init__(self, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229], gpu_id=0):
        '''
        注意：mean和std都需要自己在自己的数据集上重新算一遍，而且这里是三通道的图像，这里的顺序是BGR。
        pytorch教程中inception-v3模型使用样例中给出的值：mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]（已经按照BGR的顺序调整过）
        所有video中每个video各抽10帧求得的值：mean=[0.33373028, 0.33624287, 0.37479046], std=[0.29977513, 0.30170013, 0.31386868]
        目前的值为数据集中所有帧的均值和方差：mean=[0.34218652, 0.34897283, 0.39456914], std=[0.29780704, 0.30029216, 0.31210843]
        '''
        self.extractor = inception_v3(pretrained=True) #加载inception_v3模型
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.extractor.to(self.device) #将网络放到指定的设备上
        self.extractor.eval()
        self.dim = 2048
        self.mean = mean
        self.std = std

    def __call__(self, img_list):
        imgs = []
        for img in img_list:
            if os.path.exists(img):
                img = cv2.imread(img)
                if not isinstance(img, np.ndarray):
                    raise IOError(f'Warning: Error in {img_path}')
                img = cv2.resize(img, (299, 299))
            else:
                img = np.zeros((299, 299, 3)) # 缺图就返回一张全0的图

            #preprocess
            img = img / 255.0 #将像素值的范围缩到[0, 1]
            img = (img - self.mean) / self.std # normalize。img的后缘维度和mean和std相同，可以采用广播机制直接计算
            img = img[:, :, ::-1].copy() # 把颜色顺序从BGR转为RGB
            img = torch.from_numpy(img)
            img = img.unsqueeze(0) #增加batch_size维度
            img = img.permute(0, 3, 1, 2) #(bs, h, w, c) -> (bs, c, h, w)
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0).float().to(self.device) # (bs, c, h, w)

        soft_labels, fts = self.extractor(imgs)
        return fts.detach().cpu().numpy()


if __name__ == '__main__':
    img_path_1 = '/data8/hzp/evoked_emotion/EEV_process_data/frames/-Dzh3EhJbBg/000001.jpg'
    img_path_2 = '/data8/hzp/evoked_emotion/EEV_process_data/frames/-Dzh3EhJbBg/000002.jpg'
    img_path_list = [img_path_1, img_path_2] #一个batch的图像列表
    inception = InceptionExtractor()
    fts = inception(img_path_list)
    print(fts.shape)
    print(fts)
    





