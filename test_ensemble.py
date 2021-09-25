import os
import time
import numpy as np
import json
import pandas as pd
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

def test(model, val_iter, best_window=None):
    model.eval()
    total_pred = {}
    total_timestamp = {}
    for i, data in enumerate(val_iter):         # inner loop within one epoch
        model.set_input(data, load_label=False)  # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        vids = data['vid']
        pred = remove_padding(model.output.detach().cpu().numpy(), lengths)      
        timestamp = remove_padding(data['timestamp'].numpy(), lengths)
        for i, vid in enumerate(vids):
            total_pred[vid] = pred[i] #应该是(len, 15)
            total_timestamp[vid] = timestamp[i]
    
    # if smooth:
    #     # total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=logger)
    #     #这里可能需要对total_pred进行处理后再送入smooth_fuc，出来也要进行处理
    #     total_pred, best_window = smooth_func(total_pred, label=None, best_window=best_window, logger=None)
    
    return total_pred, total_timestamp

def load_window_size(config_path, default_window_size):
    if os.path.exists(config_path):
        data = f.open(config_path).read()
        return int(data)
    else:
        return default_window_size

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

def ensemble_all_preds(all_preds):
    ans = {}
    vids = list(all_preds[0].keys())
    for vid in vids:
        all_pred_vid = [pred[vid] for pred in all_preds]
        all_pred_vid = np.asarray(all_pred_vid)
        all_pred_vid = np.mean(all_pred_vid, axis=0)
        ans[vid] = all_pred_vid
    return ans

def load_template(vid, target):
    root = opt.template_dir
    root = os.path.join(root, target)
    df = pd.read_csv(os.path.join(root, str(vid)+'.csv'))
    return df

def make_csv(pred, timestamp, save_dir):
    pred_keys = sorted(list(pred.keys()), key=lambda x: int(x))
    timestamp_keys = sorted(list(timestamp.keys()), key=lambda x: int(x))
    assert pred_keys == timestamp_keys
    for vid in pred_keys:



        save_path = os.path.join(save_dir, str(vid)+'.csv')
        df = load_template(vid, target)
        assert (df['timestamp'] == timestamp[vid]).all()
        df['value'] = pred[vid]
        df.to_csv(save_path, index=None)

def load_dim(trn_opt):
    if trn_opt.feature_set != 'None':
        # input_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.feature_set.split(','))))
        input_dim = get_each_dim(list(map(lambda x: x.strip(), opt.feature_set.split(',')))) #得到每个特征对应的向量维度
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
    all_preds = []
    tst_timestamps = None
    total_preds = []
    val_label = None
    for checkpoint in checkpoints:
        if len(checkpoint) == 0:
            continue
        checkpoint = checkpoint.replace(' ', '')
        print('In model from {}: '.format(checkpoint))
        opt_path = os.path.join(opt.checkpoints_dir, checkpoint, 'train_opt.conf')
        trn_opt = load_config(opt_path)
        # window_path = os.path.join(opt.checkpoints_dir, checkpoint, 'best_eval_window')
        # best_window = load_window_size(window_path, default_window_size)
        # assert trn_opt.target == opt.target             # check target of model and tst target
        val_dataset, tst_dataset = create_dataset_with_args(trn_opt, set_name=['val', 'tst'])  # create a dataset given opt.dataset_mode and other options
        checkpoint_dir = os.path.join(opt.checkpoints_dir, checkpoint)
        model = load_model_from_checkpoint(trn_opt, checkpoint_dir)
        # eval val set
        # mse, rmse, pcc, ccc, preds, labels = eval(model, val_dataset, best_window)
        mse, rmse, pcc, ccc, preds, labels = eval(model, val_dataset) # 注意，这里的preds和labels的shape均为(15, total_len)
        total_preds.append(preds)
        if not isinstance(val_label, np.ndarray):
            val_label = labels
        else:
            assert(val_label == labels).all()
        print('Val result mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (mse, rmse, pcc, ccc)) #各统计量在15个类上的平均值

        # generate tst_data
        print('Testing ... \n')
        # prediction, timestamp = test(model, tst_dataset, best_window)
        prediction, timestamp = test(model, tst_dataset) # 注意，这里的prediction是个字典，prediction[vid]的shape均为(len, 15)
        if not tst_timestamps:
            tst_timestamps = timestamp
        else:
            check_timestamp(tst_timestamps, timestamp)

        all_preds.append(prediction)
        if opt.write_sub_results: # 即使在ensemble时，也将每个模型的预测结果分别记录下来
            csv_folder = os.path.join(opt.submit_dir, opt.name, checkpoint)
            make_folder(csv_folder)
            make_csv(prediction, timestamp, csv_folder) #这里需要修改一下，把15个类都列好



 
        
    # 以下部分暂未修改
    # make ensemble prediction
    total_preds = np.asarray(total_preds)
    total_preds = np.mean(total_preds, axis=0)
    mse, rmse, pcc, ccc = evaluate_regression(val_label, total_preds)
    print('Ensemble Val result mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (mse, rmse, pcc, ccc))

    ensemble_pred = ensemble_all_preds(all_preds)
    ensemble_dir = os.path.join(opt.submit_dir, opt.name, opt.target, 'ensemble')
    make_folder(ensemble_dir)
    make_csv(prediction, tst_timestamps, ensemble_dir, opt.target)
    