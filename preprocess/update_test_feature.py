'''
对于将之前test集合相比官方有效的视频缺少的5部视频，抽取出相应的feature并且添加到各feature对应的h5文件中
'''
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv

features_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'

new_video_list = ['CBn2M-3Safw', 'npGwn9_2MsA', 'WTyLlQzzsvE', 'XCfRIUOyMNQ', 'Yesf4wi8DaM']

