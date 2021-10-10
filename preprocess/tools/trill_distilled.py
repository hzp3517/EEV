'''
参考1st的代码，直接用tensorflow_hub中的模型得到基于MobileNet-V2的TRILL-distilled特征。
需要配置tensorflow-gpu=2.2.0 (pip), cudnn=7.6.5 (conda), cudatoolkit=10.1 (conda)
'''
import gc
import os
import sys
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import soundfile as sf

from tensorflow.keras import layers
import tensorflow_hub as hub
# import librosa
import numpy as np

sys.path.append('/data8/hzp/evoked_emotion/EEV/preprocess')
from tools.base_worker import BaseWorker

class TrillDistilledExtractor(BaseWorker):
    '''
    模型链接：https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3
    '''
    def __init__(self, seg_len=1/6, gpu_id=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) #默认情况下，TensorFlow会占用所有GPUs的所有GPU内存（取决于CUDA_VISIBLE_DEVICES这个系统变量）
        physical_devices = tf.config.list_physical_devices('GPU') # 当前可用的gpu列表
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True) #使用哪一块gpu

        self.seg_len = seg_len
        self.DEFAULT_SR = 16000
        self.model = tf.keras.Sequential([hub.KerasLayer('/data8/hzp/models/trill-distilled',
                                                 arguments={'sample_rate': tf.constant(self.DEFAULT_SR, tf.int32)},
                                                 trainable=False, output_key='embedding',
                                                 output_shape=[None, 2048])]) #语音的MobileNet

    def read_wav(self, wav_path):
        wav_data, sr = sf.read(wav_path, dtype='int16')
        assert sr == 16000
        time_length = len(wav_data) / sr #len(wav_data):2377352, sr:16000. time_length=148.5845
        timestamps = [self.seg_len * n for n in range(int(time_length / self.seg_len))] #列表中的元素即为所有timestamp时刻（每个1/6s时间段的起始时刻）：[0, 1/6, 2/6, ..., 148+(2/6)]
        num_ts = len(timestamps)
        if num_ts > 0 and wav_data.shape[0] % num_ts > 0: #平均每段timestamp对应几帧语音帧，对最后一段timestamp不足的语音帧直接在右侧补0
            num_pad = num_ts - (wav_data.shape[0] % num_ts)
            wav_data = np.pad(wav_data, ((0, num_pad)), mode='constant')
        wav_data = wav_data.reshape(num_ts, -1)
        return wav_data

    def __call__(self, wav_path, batch_size=2048):
        wav_data = self.read_wav(wav_path)
        features = self.model.predict(wav_data, batch_size=batch_size)
        features = np.squeeze(features)
        return features


if __name__ == '__main__':
    trill_distilled_extract = TrillDistilledExtractor(gpu_id=0)
    audio_path = "/data8/hzp/evoked_emotion/EEV_process_data/audios/-01d8S_0AHs.wav" #这个样例长度为2min 28s，即共148s
    fts = trill_distilled_extract(audio_path, batch_size=2048)
    print(fts)
    print(fts.shape)
    print(type(fts))


