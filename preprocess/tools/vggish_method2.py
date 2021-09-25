'''
得到vggish特征
'''

import numpy as np
import os
import sys
import glob
import torch
import urllib

sys.path.append('/data8/hzp/evoked_emotion/EEV/preprocess')
from tools.base_worker import BaseWorker
from tools.torchvggish.vggish import Postprocessor, VGGish

class VggishExtractor(BaseWorker):
    def __init__(self, vggish_path='/data8/hzp/models/torchvggish/vggish-10086976.pth', 
                    pca_path='/data8/hzp/models/torchvggish/vggish_pca_params-970ea276.pth', gpu_id=1):
        self.device = torch.device("cuda:{}".format(gpu_id))
        #self.extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.extractor = VGGish(vggish_path, pca_path, gpu_id)#

        self.extractor.to(self.device) #将网络放到指定的设备上
        self.extractor.eval()


    def __call__(self, audio_path):
        if os.path.exists(audio_path):
            ft = self.extractor(audio_path)
            return ft.detach().cpu().numpy()
        else:
            return None



if __name__ == '__main__':
    # url, audio_path= ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
    # try: urllib.URLopener().retrieve(url, audio_path)
    # except: urllib.request.urlretrieve(url, audio_path)

    # audio_path = '/data8/hzp/evoked_emotion/EEV_process_data/audios/-NpS1zlc9IE.wav'  #这个样例长度为 21min 38s
    # audio_path = '/data8/hzp/evoked_emotion/EEV_process_data/audios/wzDOl5PQE1k.wav'  #这个样例长度为 27min 51.30s，应该是数据集里最长的，卡空的时候可以处理
    # audio_path = '/data8/hzp/evoked_emotion/EEV_process_data/audios/-01d8S_0AHs.wav' #这个样例长度为2min 28s。即共148s，抽出的特征长为154
    audio_path = '/data8/hzp/evoked_emotion/EEV_process_data/audios/BDnhz1xAJgA.wav' #这个样例长度为6min 15s。即共375s，抽出的特征长为390

    vggish = VggishExtractor(gpu_id=3)
    ft = vggish(audio_path)
    print(ft.shape)
    print(ft)

