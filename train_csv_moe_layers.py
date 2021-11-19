'''
采用自动化脚本跑实验的时候，自动将结果写入csv文件
'''

import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func
# from utils.tools import calc_total_dim
from utils.tools import get_each_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import fcntl
import csv

def test(model, test_iter):
    pass

def eval(model, val_iter):
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

        # print(len(pred), len(label), len(valid)) #batch_size
        # print(pred[0].shape) #(len, 15)
        # print(pred[1].shape) #(len, 15)
        # print(pred[2].shape) #(len, 15)
        # print(pred[3].shape) #(len, 15)
        # print('----------') #(len, 15)
        # print(label[0].shape) #(len, 15)
        # print(valid[0].shape) #(len,)

        for i in range(len(pred)):
            _pred = np.delete(pred[i], np.argwhere(valid[i]==0).squeeze(), axis=0) # 去掉全0标注的时刻对应的预测结果
            _label = np.delete(label[i], np.argwhere(valid[i]==0).squeeze(), axis=0) # 去掉全0标注的时刻对应的标签

            # print(_pred.shape)#

            total_pred.append(_pred)
            total_label.append(_label)


    # # calculate metrics
    # best_window = None
    # if smooth:
    #     #这里可能需要对total_pred进行处理后再送入smooth_fuc，出来也要进行处理
    #     total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=logger)

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

    return mean_mse, mean_rmse, mean_pcc, mean_ccc, best_window

def clean_chekpoints(checkpoints_dir, expr_name, store_epoch):
    root = os.path.join(checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

#--------------只输出pcc结果--------------
# def auto_write_csv(csv_result_dir, opt, best_eval_epoch, best_eval_pcc):
#     name = opt.name
#     if opt.suffix:
#         suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#         suffix = suffix.replace(',', '-')
#     name = name.replace(suffix, '')

#     csv_path = os.path.join(csv_result_dir, name + '.csv')

#     feature = opt.feature_set
#     lr = opt.lr
#     run_idx = opt.run_idx

#     lines = []
#     row_pos = None

#     f = open(csv_path, "r+")
#     fcntl.flock(f.fileno(), fcntl.LOCK_EX) #加锁
#     reader = csv.reader(f)
#     feature = feature.replace(',', '+')

#     line = next(reader) #表头
#     column_pos = None
#     c_id = 0
#     for c in line:
#         if str(lr) + '_run' + str(run_idx) == c:
#             column_pos = c_id
#             break
#         c_id += 1
#     lines.append(line)

#     row_id = 1
#     for line in reader:
#         lines.append(line)
#         if feature == line[0]:
#             row_pos = row_id
#         row_id += 1
        
#     f.seek(0) #写之前先要把文件指针归零
#     writer = csv.writer(f)
#     if row_pos and column_pos:
#         lines[row_pos][column_pos] = round(best_eval_pcc, 6)
#     writer.writerows(lines)
#     fcntl.flock(f.fileno(), fcntl.LOCK_UN) #解锁
#     f.close()


#--------------同时输出best_epoch和pcc-------------
def auto_write_csv(csv_result_dir, opt, best_eval_epoch, best_eval_pcc):
    name = opt.name
    if opt.suffix:
        suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        suffix = suffix.replace(',', '-')
    name = name.replace(suffix, '')

    csv_path = os.path.join(csv_result_dir, name + '.csv')

    feature = opt.feature_set
    expert_num = opt.expert_num
    run_idx = opt.run_idx

    lines = []
    row_pos = None

    f = open(csv_path, "r+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX) #加锁
    reader = csv.reader(f)
    feature = feature.replace(',', '+')

    line = next(reader) #表头
    epoch_column_pos = None
    pcc_column_pos = None

    c_id = 0
    for c in line:
        if str(expert_num) + '_run' + str(run_idx) + '_epoch' == c:
            epoch_column_pos = c_id
        if str(expert_num) + '_run' + str(run_idx) + '_pcc' == c:
            pcc_column_pos = c_id
        c_id += 1
    lines.append(line)

    row_id = 1
    for line in reader:
        lines.append(line)
        if feature == line[0]:
            row_pos = row_id
        row_id += 1
        
    f.seek(0) #写之前先要把文件指针归零
    writer = csv.writer(f)
    if row_pos and epoch_column_pos and pcc_column_pos:
        lines[row_pos][pcc_column_pos] = round(best_eval_pcc, 6)
        lines[row_pos][epoch_column_pos] = best_eval_epoch
    writer.writerows(lines)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN) #解锁
    f.close()




if __name__ == '__main__':
    smooth = False
    best_window = None
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name)   # get logger path
    suffix = opt.name                                   # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    
    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)                         # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
                                                        # calculate input dims
    if opt.feature_set != 'None':
        # input_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.feature_set.split(','))))
        input_dim = get_each_dim(list(map(lambda x: x.strip(), opt.feature_set.split(',')))) #得到每个特征对应的向量维度
        setattr(opt, "input_dim", input_dim)                # set input_dim attribute to opt
    
    model = create_model(opt, logger=logger)    # create a model given opt.model and other options
    model.setup(opt)                            # regular setup: load and print networks; create schedulers
    total_iters = 0                             # the total number of training iterations
    best_eval_pcc = 0                           # record the best eval pcc
    best_eval_epoch = -1                        # record the best eval epoch
    best_eval_window = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        iter_data_statis = 0.0          # record total data reading time
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()   # timer for computation per iteration
            iter_data_statis += iter_start_time-iter_data_time
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)           # unpack data from dataset and apply preprocessing
            model.run()                     # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec, Data loading: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, iter_data_statis))
        model.update_learning_rate()                      # update learning rates at the end of every epoch.

        # eval trn set
        mse, rmse, pcc, ccc, window = eval(model, dataset)
        logger.info('Trn result of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))
        
        # eval val set
        mse, rmse, pcc, ccc, window = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))
        if pcc > best_eval_pcc:
            best_eval_epoch = epoch
            best_eval_pcc = pcc
            best_eval_window = window
    
    # print best eval result
    logger.info('Best eval epoch %d found with pcc %f' % (best_eval_epoch, best_eval_pcc))
    logger.info(opt.name)
    # record best window
    if smooth:
        f = open(os.path.join(opt.checkpoints_dir, opt.name, 'best_eval_window'), 'w')
        f.write(str(best_eval_window))
        f.close()
    
    # write to result dir
    clean_chekpoints(opt.checkpoints_dir, opt.name, best_eval_epoch)
    autorun_result_dir = 'autorun/results'
    f = open(os.path.join(autorun_result_dir, opt.name + '.txt'), 'w')
    f.write('Best eval epoch %d found with pcc %.4f' % (best_eval_epoch, best_eval_pcc))
    f.close()

    #write to csv result
    csv_result_dir = os.path.join('autorun', 'csv_results', 'moe_layers')
    auto_write_csv(csv_result_dir, opt, best_eval_epoch, best_eval_pcc)
