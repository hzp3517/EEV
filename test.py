import os
import time
import numpy as np
import json
import pandas as pd
import h5py
import csv
from tqdm import tqdm
from opts.test_opts import TestOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from models.utils.config import OptConfig
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func
from utils.tools import calc_total_dim, make_folder, get_each_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def eval(model, val_iter, best_window=None):
    model.eval()
    total_pred = []
    total_label = []
    total_length = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        pred = remove_padding(model.output.detach().cpu().numpy(), lengths)
        # label = remove_padding(data[opt.target].numpy(), lengths)
        label = remove_padding(data['label'].numpy(), lengths)
        valid = remove_padding(data['valid'].numpy(), lengths)
        # total_pred += pred
        # total_label += label

        for i in range(len(pred)):
            _pred = np.delete(pred[i], np.argwhere(valid[i]==0).squeeze(), axis=0) # 去掉全0标注的时刻对应的预测结果
            _label = np.delete(label[i], np.argwhere(valid[i]==0).squeeze(), axis=0) # 去掉全0标注的时刻对应的标签
            total_pred.append(_pred)
            total_label.append(_label)

    # # calculate metrics
    # if smooth:
    #     #这里可能需要对total_pred进行处理后再送入smooth_fuc，出来也要进行处理
    #     total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=None)

    total_pred = scratch_data(total_pred) #(total_len, 15)
    total_label = scratch_data(total_label) #(total_len, 15)

    total_pred = total_pred.transpose() #(15, total_len)
    total_label = total_label.transpose() #(15, total_len)

    mse_list, rmse_list, pcc_list, ccc_list = [], [], [], []

    for cate_pred, cate_label in zip(total_pred, total_label):
        mse, rmse, pcc, ccc = evaluate_regression(cate_pred, cate_label)
        mse_list.append(mse)
        rmse_list.append(rmse)
        pcc_list.append(pcc)
        ccc_list.append(ccc)

    mean_mse = np.mean(mse_list)
    mean_rmse = np.mean(rmse_list)
    mean_pcc = np.mean(pcc)
    mean_ccc = np.mean(ccc)

    model.train()

    return mean_mse, mean_rmse, mean_pcc, mean_ccc, total_pred, total_label

def test(model, test_iter, best_window=None):
    model.eval()
    total_pred = {}
    total_timestamp = {}
    for i, data in enumerate(test_iter):         # inner loop within one epoch
        model.set_input(data, load_label=False)  # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        vids = data['vid']
        pred = remove_padding(model.output.detach().cpu().numpy(), lengths)      
        timestamp = remove_padding(data['timestamp'].numpy(), lengths)
        for i, vid in enumerate(vids):
            # total_pred[vid] = pred[i] #应该是(len, 15)
            total_pred[vid] = pred[i].transpose() #应该是(15, len)
            total_timestamp[vid] = timestamp[i]
    
    # if smooth:
    #     # total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=logger)
    #     #这里可能需要对total_pred进行处理后再送入smooth_fuc，出来也要进行处理
    #     total_pred, best_window = smooth_func(total_pred, label=None, best_window=best_window, logger=None)
    
    return total_pred, total_timestamp

def load_config(opt_path):
    trn_opt_data = json.load(open(opt_path))
    trn_opt = OptConfig()
    trn_opt.load(trn_opt_data)
    load_dim(trn_opt)
    trn_opt.gpu_ids = opt.gpu_ids
    # trn_opt.dataroot = 'dataset/wild'
    trn_opt.serial_batches = True # 顺序读入
    # if not hasattr(trn_opt, 'normalize'):       # previous model has no attribute normalize
    #     setattr(trn_opt, 'normalize', False)
    if not hasattr(trn_opt, 'loss_type'):
        setattr(trn_opt, 'loss_type', 'mse')
    return trn_opt

def load_model_from_checkpoint(opt_config, cpkt_dir):
    model = create_model(opt_config)
    model.load_networks_folder(cpkt_dir)
    model.eval()
    model.cuda()
    model.isTrain = False
    return model

# def load_template(vid, target):
#     root = opt.template_dir
#     root = os.path.join(root, target)
#     df = pd.read_csv(os.path.join(root, str(vid)+'.csv'))
#     return df

def make_result_dict(category_list, pred, timestamp, invalid_target_path):
    # 对于valid的视频，记录其预测结果
    pred_keys = list(pred.keys())
    timestamp_keys = list(timestamp.keys())
    assert pred_keys == timestamp_keys
    result_dict = {}
    for vid in pred_keys:
        result_dict[vid] = {}
        result_dict[vid]['timestamp'] = timestamp[vid]
        for i, cate in enumerate(category_list):
            result_dict[vid][cate] = pred[vid][i]

    # 对于invalid的视频，在所有timestamp处补0
    h5f = h5py.File(invalid_target_path, 'r')
    for vid in h5f.keys():
        result_dict[vid] = {}
        result_dict[vid]['timestamp'] = h5f[vid]['timestamp']
        for i, cate in enumerate(category_list):
            result_dict[vid][cate] = np.zeros(len(h5f[vid]['timestamp']))

    return result_dict


def make_csv(category_list, result_dict, save_dir):
    origin_test_csv = '/data8/datasets/eevchallenges/test.csv'
    df = pd.read_csv(origin_test_csv)
    all_video_id = np.array(df['Video ID'])
    all_timestamp = np.array(df['Timestamp (milliseconds)'])

    video_order_list = [] # 按照test.csv文件中的顺序排列的各视频名称列表
    all_info_dict = {}
    all_info_dict['Video ID'] = all_video_id
    all_info_dict['Timestamp (milliseconds)'] = all_timestamp
    for cate in category_list:
        # all_label_dict[cate] = [cate] #列表中先加入每个列的表头（即类别名称）
        all_info_dict[cate] = []

    num_data = len(all_video_id)

    cur_vid = all_video_id[0]
    s_idx = 0
    e_idx = 0

    for e_idx in tqdm(range(num_data)):
    # for e_idx in range(num_data):
        if all_video_id[e_idx] != cur_vid: 
            video_order_list.append(cur_vid)
        cur_vid = all_video_id[e_idx]
        s_idx = e_idx
    video_order_list.append(cur_vid) #最后一个video也不要落下

    for cate in category_list:
        for vid in video_order_list:
            all_info_dict[cate].append(result_dict[vid][cate])
        all_info_dict[cate] = np.concatenate(all_info_dict[cate])

    assert num_data == len(all_info_dict['anger']) # 确认要输出的数据行数和原文件中的行数是完全一致的

    # 写入csv文件
    save_path = os.path.join(save_dir, 'result.csv')
    df = pd.DataFrame(all_info_dict)
    df.to_csv(save_path, index=False, sep=',') # index表示是否显示行名


def load_dim(trn_opt):
    if trn_opt.feature_set != 'None':
        # input_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.feature_set.split(','))))
        input_dim = get_each_dim(list(map(lambda x: x.strip(), trn_opt.feature_set.split(',')))) #得到每个特征对应的向量维度
        setattr(trn_opt, "input_dim", input_dim)                # set input_dim attribute to opt


def check_timestamp(timestamp1, timestamp2):
    keys1 = sorted(list(timestamp1))
    keys2 = sorted(list(timestamp2))
    assert keys1 == keys2, '{}\n{}'.format(keys1, keys2)
    for key in keys1:
        assert (timestamp1[key] == timestamp2[key]).all()


if __name__ == '__main__':
    smooth = False
    # default_window_size = 10
    opt = TestOptions().parse()                         # get test options
    checkpoints = opt.test_checkpoints.strip().split(';')
    test_timestamps = None
    # val_label = None


    for checkpoint in checkpoints:
        if len(checkpoint) == 0:
            continue
        checkpoint = checkpoint.replace(' ', '')
        print('In model from {}: '.format(checkpoint))
        opt_path = os.path.join(opt.checkpoints_dir, checkpoint, 'train_opt.conf')
        trn_opt = load_config(opt_path)

        # val_dataset, test_dataset = create_dataset_with_args(trn_opt, set_name=['val', 'test'])  # create a dataset given opt.dataset_mode and other options
        test_dataset_tuple = create_dataset_with_args(trn_opt, set_name=['test'])  # create a dataset given opt.dataset_mode and other options
        test_dataset = test_dataset_tuple[0] #上面一行返回的是一个元组，要把里面的dataloader元素取出来

        checkpoint_dir = os.path.join(opt.checkpoints_dir, checkpoint)
        model = load_model_from_checkpoint(trn_opt, checkpoint_dir)
        
        # # validation
        # mse, rmse, pcc, ccc, preds, labels = eval(model, val_dataset) # 注意，这里的preds和labels的shape均为(15, total_len)
        # if not isinstance(val_label, np.ndarray):
        #     val_label = labels
        # else:
        #     assert(val_label == labels).all()
        # print('Val result mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (mse, rmse, pcc, ccc)) #各统计量在15个类上的平均值

        # generate test_data
        print('Testing ... \n')
        prediction, timestamp = test(model, test_dataset) # 注意，这里的prediction是个字典，prediction[vid]的shape均为(15, len)
        
        category_list = test_dataset.dataset.category_list # EEV数据集的类别列表
        data_root = test_dataset.dataset.root #数据feature和target文件的存放位置根目录
        invalid_target_path = os.path.join(data_root, 'target', 'invalid_test_target.h5')

        if not test_timestamps:
            test_timestamps = timestamp
        else:
            check_timestamp(test_timestamps, timestamp)

        csv_save_folder = os.path.join(opt.submit_dir, opt.name, checkpoint)
        make_folder(csv_save_folder)

        result_dict = make_result_dict(category_list, prediction, timestamp, invalid_target_path) #构建一个存储所有视频的预测结果的字典（包括invalid视频，将所有对应timestamp位置补0）

        print('making csv file ...')
        make_csv(category_list, result_dict, csv_save_folder)



 
        
