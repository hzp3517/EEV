import h5py
import numpy as np
import torch
from tqdm import tqdm
import os
import glob
import cv2
from random import shuffle
import math

# def cal_mean_std_origin(image_root, save_dir): #注意：颜色通道顺序为BGR以适应opencv
#     img_size = 299
#     all_images = []
#     video_ids = os.listdir(image_root)

#     video_ids = video_ids[:10]#

#     for video_id in tqdm(video_ids):
#         image_dir = os.path.join(image_root, video_id)
#         image_lst = glob.glob(os.path.join(image_dir, '*.jpg')) #image_dir为某个视频中的所有人脸图像目录

#         for image in image_lst:
#             img = cv2.imread(image)
#             if img is None:
#                 continue
#             img = cv2.resize(img, (img_size, img_size))
#             img = img.reshape(-1, 3)
#             img = img / 256.0 #将像素值的范围缩到[0, 1]
#             all_images.append(img)

#     all_images = np.concatenate(all_images, axis=0)
#     print(all_images.shape)

#     mean = all_images.mean(axis=0)
#     std = all_images.std(axis=0)

#     print('MEAN:', mean)
#     print('STD:', std)

#     # # 记录下运算的结果：
#     # record_file = os.path.join(save_dir, 'image_mean_std.txt')
#     # with open(record_file, 'w') as f:
#     #     f.write('mean:' + str(mean) + '\n')
#     #     f.write('std:' + str(std) + '\n')


def cal_mean_std(image_root, save_dir): #注意：颜色通道顺序为BGR以适应opencv
    '''
    方差计算公式：s² = (x1² + x2²+ ... + xn²)/n - [(x1 + x2 + ... + xn)/n]²，即：(平方平均)²-(算数平均)²，也即：平方和的均值-(算数平均)²
    标准差为 s
    '''
    img_size = 299
    mean_list = [] #每个视频中所有像素的算术平均
    pow_mean_list = [] #每个视频中所有像素的平方平均
    img_num_list = []
    video_ids = os.listdir(image_root)

    for video_id in tqdm(video_ids):
        video_img_list = []
        cnt_img = 0
        image_dir = os.path.join(image_root, video_id)
        image_lst = glob.glob(os.path.join(image_dir, '*.jpg')) #image_dir为某个视频中的所有人脸图像目录

        for image in image_lst:
            img = cv2.imread(image)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img.reshape(-1, 3)
            img = img / 256.0 #将像素值的范围缩到[0, 1]
            video_img_list.append(img)
            cnt_img += 1
        video_images = np.concatenate(video_img_list, axis=0)
        pow_video_images = pow(video_images, 2)
        video_mean = video_images.mean(axis=0)
        pow_video_mean = pow_video_images.mean(axis=0)

        mean_list.append(video_mean)
        pow_mean_list.append(pow_video_mean)
        img_num_list.append(cnt_img)
    
    mean_array = np.stack(mean_list)
    pow_mean_array = np.stack(pow_mean_list)
    img_num_array = np.stack(img_num_list)

    all_img_num = img_num_array.sum()

    img_num_array = np.expand_dims(img_num_array, axis=1) #扩充维度
    img_num_array = np.repeat(img_num_array, 3, axis=1) #在对应维度重复

    weighted_mean_array = mean_array * img_num_array # *为对应位置相乘
    weighted_pow_mean_array = pow_mean_array * img_num_array
    # global_mean_array = weighted_mean_array / all_img_num
    # global_pow_mean_array = weighted_pow_mean_array / all_img_num

    mean = weighted_mean_array.sum(axis=0) / all_img_num
    pow_mean = weighted_pow_mean_array.sum(axis=0) / all_img_num
    var = pow_mean - pow(mean, 2)
    std = np.array([math.sqrt(i) for i in var])

    print('MEAN:', mean)
    print('STD:', std)

    # 记录下运算的结果：
    record_file = os.path.join(save_dir, 'image_mean_std.txt')
    with open(record_file, 'w') as f:
        f.write('mean:' + str(mean) + '\n')
        f.write('std:' + str(std) + '\n')


if __name__ == '__main__':
    image_root = '/data8/hzp/evoked_emotion/EEV_process_data/frames/'
    save_dir = '/data8/hzp/evoked_emotion/EEV_process_data/' #计算出的mean和std的结果文件保存目录
    cal_mean_std(image_root, save_dir)
    # cal_mean_std_origin(image_root, save_dir)
