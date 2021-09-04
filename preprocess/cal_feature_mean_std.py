'''
在训练集上计算特征的均值和方差，用于语音特征的归一化
'''
import h5py
import numpy as np
import torch
from tqdm import tqdm
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def load_train_video_ids(root):
    partition_h5f = h5py.File(os.path.join(root, 'target', 'partition.h5'), 'r')
    video_ids = partition_h5f['train']['valid']
    video_ids = list(map(lambda x: str(x.decode()), video_ids))
    return video_ids
    
def cal_mean_std(root, video_ids, feature):
    input_file = os.path.join(root, 'features', '{}.h5'.format(feature))
    mkdir(os.path.join(root, 'mean_std_on_train'))
    output_file = os.path.join(root, 'mean_std_on_train', '{}.h5'.format(feature)) #记录mean和std信息的h5文件路径
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    feature_data = [in_data['train'][_id]['feature'][()] for _id in video_ids] #读取h5文件，读入所有训练集上的特征
    feature_data = np.concatenate(feature_data, axis=0) #这里是将不同个音频的数据拼起来，拼完之后第0维代表包括了所有音频数据的每一帧。也就是说这时有2个维度：[所有音频合起来的seq_len, ft_dim]
    mean_f = np.mean(feature_data, axis=0)
    std_f = np.std(feature_data, axis=0)
    std_f[std_f == 0.0] = 1.0
    group = h5f.create_group('train') #创建一个名为'train'的组
    group['mean'] = mean_f
    group['std'] = std_f
    print(mean_f.shape)
    return mean_f, std_f





if __name__ == '__main__':
    root = '/data8/hzp/evoked_emotion/EEV_process_data/'
    # target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/'
    # feature_dir = '/data8/hzp/evoked_emotion/EEV_process_data/features/'
    video_ids = load_train_video_ids(root)
    features = ['vggish']
    for ft in features:
        print('process feature:', ft)
        mean_f, std_f = cal_mean_std(root, video_ids, ft) #返回值为np.array类型