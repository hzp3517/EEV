import h5py
import numpy as np
import torch
from tqdm import tqdm
import os
import glob
import cv2
from random import shuffle

# def cal_mean_std_origin(image_root, save_dir): #注意：颜色通道顺序为BGR以适应opencv
#     img_size = 299
#     all_images = []
#     video_ids = os.listdir(image_root)

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

#     mean = all_images.mean()
#     std = all_images.std()

#     print('MEAN:', mean)
#     print('STD:', std)

#     # 记录下运算的结果：
#     record_file = os.path.join(save_dir, 'image_mean_std.txt')
#     with open(record_file, 'w') as f:
#         f.write('mean:' + str(mean) + '\n')
#         f.write('std:' + str(std) + '\n')


def cal_mean_std_random_videos(image_root, save_dir, sample_num=350): #随机采样一小部分视频计算mean和std #注意：颜色通道顺序为BGR以适应opencv
    img_size = 299
    all_images = []
    video_ids = os.listdir(image_root)
    shuffle(video_ids)
    video_ids = video_ids[:sample_num]

    for video_id in tqdm(video_ids):
        image_dir = os.path.join(image_root, video_id)
        image_lst = glob.glob(os.path.join(image_dir, '*.jpg')) #image_dir为某个视频中的所有人脸图像目录

        for image in image_lst:
            img = cv2.imread(image)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img.reshape(-1, 3)
            img = img / 256.0 #将像素值的范围缩到[0, 1]
            all_images.append(img)

    all_images = np.concatenate(all_images, axis=0)
    print(all_images.shape)

    mean = all_images.mean(axis=0)
    std = all_images.std(axis=0)

    print('MEAN:', mean)
    print('STD:', std)

    # 记录下运算的结果：
    record_file = os.path.join(save_dir, 'image_mean_std.txt')
    with open(record_file, 'w') as f:
        f.write('mean:' + str(mean) + '\n')
        f.write('std:' + str(std) + '\n')



# def cal_mean_std_recursion(x_cur, E_pre, F_pre, n): # 均值和方差递推公式的实现
#     '''
#     递推公式：
#     E(n) = E(n-1) + [x(n) - E(n-1)] / n;
#     F(n) = F(n-1) + [x(n) - E(n-1)][x(n) - E(n)];
#     其中E(0) = x(0), F(0) = 0.
#     返回E(n), F(n).
#     '''
#     E_cur = E_pre + (x_cur - E_pre) / n
#     F_cur = F_pre + (x_cur - E_pre)*(x_cur - E_cur)
#     n_post = n + 1
#     return E_cur, F_cur, n_post


# def cal_mean_std(image_root, save_dir): #注意：颜色通道顺序为BGR以适应opencv
#     '''
#     这样会特别慢，算一个图都要20分钟
#     '''

#     img_size = 299
#     video_ids = os.listdir(image_root)

#     # video_ids = video_ids[:10]#

#     #首先要得到x0，这个就要得到第一个视频中的第一张图的第一个像素（三个通道各自的值组成的array）
#     first_video_image_lst = glob.glob(os.path.join(image_root, video_ids[0], '*.jpg'))
#     first_video_images = []
#     for image in first_video_image_lst:
#         img = cv2.imread(image)
#         if img is None:
#             continue
#         img = cv2.resize(img, (img_size, img_size))
#         img = img.reshape(-1, 3)
#         img = img / 256.0 #将像素值的范围缩到[0, 1]
#         first_video_images.append(img)
#     first_video_images = np.concatenate(first_video_images, axis=0)
#     x_pre = first_video_images[0] #x(0)
#     E_pre = x_pre.astype(np.float32) #E(0)=x(0)
#     F_pre = np.zeros([3], dtype=np.float32) #F(0)=0
#     n = 1

#     for x_cur in first_video_images[1:]: #去掉第一个像素
#         E_cur, F_cur, n = cal_mean_std_recursion(x_cur, E_pre, F_pre, n)
#         E_pre = E_cur
#         F_pre = F_cur

#     # 再处理第一部之后所有部video
#     video_ids = video_ids[1:]
#     for video_id in tqdm(video_ids):
#         image_dir = os.path.join(image_root, video_id)
#         image_lst = glob.glob(os.path.join(image_dir, '*.jpg')) #image_dir为某个视频中的所有人脸图像目录
#         video_images = []

#         for image in image_lst:
#             img = cv2.imread(image)
#             if img is None:
#                 continue
#             img = cv2.resize(img, (img_size, img_size))
#             img = img.reshape(-1, 3)
#             img = img / 256.0 #将像素值的范围缩到[0, 1]
#             video_images.append(img)
#         video_images = np.concatenate(video_images, axis=0)

#         for x_cur in video_images:
#             E_cur, F_cur, n = cal_mean_std_recursion(x_cur, E_pre, F_pre, n)
#             E_pre = E_cur
#             F_pre = F_cur


#     print(E_pre, F_pre)


def cal_mean_std_random_frames(image_root, save_dir, sample_num=10): #在每个视频中随机采样sample_num帧 #注意：颜色通道顺序为BGR以适应opencv
    img_size = 299
    all_images = []
    video_ids = os.listdir(image_root)

    for video_id in tqdm(video_ids):
        image_dir = os.path.join(image_root, video_id)
        image_lst = glob.glob(os.path.join(image_dir, '*.jpg')) #image_dir为某个视频中的所有人脸图像目录

        #随机从当前视频中选取sample_num帧
        shuffle(image_lst)
        image_lst = image_lst[:sample_num]

        for image in image_lst:
            img = cv2.imread(image)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img.reshape(-1, 3)
            img = img / 256.0 #将像素值的范围缩到[0, 1]
            all_images.append(img)

    all_images = np.concatenate(all_images, axis=0)
    print(all_images.shape)

    mean = all_images.mean(axis=0)
    std = all_images.std(axis=0)

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
    cal_mean_std_random_frames(image_root, save_dir, 10)