'''
先用MTCNN检测人脸，获得人脸tensor，再送入FaceNet里面得到最终的embedding
'''

import torch
import cv2
import numpy as np
import os
import sys
import glob
# from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

sys.path.append('/data8/hzp/evoked_emotion/EEV/preprocess')
from tools.base_worker import BaseWorker
from tools.facenet_pytorch.models.mtcnn import MTCNN
from tools.facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

# def chunk(lst, chunk_size): #chunk的作用：如果一个目录下的人脸图片较多，超过chunk_size张，就每chunk_size张截断一次
#     idx = 0
#     while chunk_size * idx < len(lst):
#         yield lst[idx*chunk_size: (idx+1)*chunk_size]
#         idx += 1


class FacenetExtractor(BaseWorker):
    def __init__(self, gpu_id=0):
        self.device = torch.device("cuda:{}".format(gpu_id))

        #文章里说用两个最大的人脸，所以我这里就设为True了，但是这样可能导致选出的脸并非置信度最高的；
        #keep_all=True表示返回所有检测到的脸，这样我就可以选取前两个了
        self.mtcnn = MTCNN(image_size=160, select_largest=True, keep_all=True, device=self.device).eval()

        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

    def __call__(self, img):
        frame = cv2.imread(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # Detect face
        face = self.mtcnn(frame) #返回值shape: torch.Size([num_faces, 3, 160, 160])

        if face == None: #无人脸的情况
            embedding = torch.zeros(1024).to(self.device)
        elif len(face)==1:
            face = face.to(self.device)
            embedding = self.resnet(face)
            embedding = torch.cat((embedding[0], torch.zeros(512).to(self.device)))
        else: #2个或以上
            face = face.to(self.device)
            face = face[:2] #只取前两个脸
            embedding = self.resnet(face)
            embedding = torch.cat((embedding[0], embedding[1]))

        return embedding.detach().cpu().numpy()


if __name__ == '__main__':
    #img_path = '/data8/hzp/evoked_emotion/EEV_process_data/frames/-Dzh3EhJbBg/000001.jpg'
    img_path = '/data8/hzp/evoked_emotion/EEV_process_data/frames/--BPh-G6HAE/000001.jpg'# 没脸
    facenet = FacenetExtractor(gpu_id=2)
    ft = facenet(img_path)
    print(ft.shape)
    print(ft)




