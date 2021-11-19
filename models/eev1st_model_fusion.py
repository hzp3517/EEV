import torch
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .base_model import BaseModel
# from .networks.tcn import TCNModel
# from .networks.regressor import FcRegressor
# from .networks.loss import PCCLoss

import sys
sys.path.append('/data8/hzp/evoked_emotion/EEV/')#
from models.base_model import BaseModel#
from models.networks.tcn import TCNModel#
from models.networks.regressor import FcRegressor#
from models.networks.loss import PCCLoss#

class EEV1stModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('--max_seq_len', type=int, default=60, help='max sequence length of gru')
        parser.add_argument('--dropout_rate', default=0.2, type=float, help='drop out rate of TCN')
        parser.add_argument('--tcn_channels', default='(512),(512,512)', 
                            help='number of tcn channels. each feature with a bracket. split by comma.')
                            # 每个括号中是同一个特征堆叠的多个tcn的输出维度
        parser.add_argument('--hidden_size', default='128', type=str, help='size of regressor hidden layer, split by comma')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'pcc'], help='use MSEloss or PCCloss')
        # parser.add_argument('--optimizer', type=str, default='sgd', choices=['Adam', 'sgd'], help="Optimizer")
        # parser.add_argument('--emotion', type=int, default=0, help="Emotion index to be learned (0-14) or all (-1)")
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the EEV1stModel class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = [opt.loss_type]
        self.model_names = ['_tcn', '_reg']

        self.pretrained_model = []

        # self.max_seq_len = opt.max_seq_len
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.tcn_channels = []
        pattern = re.compile(r'[(](.*?)[)]', re.S) #最小匹配，匹配小括号中的内容
        for i in re.findall(pattern, opt.tcn_channels):
            self.tcn_channels.append(list(map(lambda x: int(x.strip()), i.split(','))))
        self.hidden_size_list = list(map(lambda x: int(x.strip()), opt.hidden_size.split(',')))
        assert len(self.feature_set) == len(self.tcn_channels)

        self.net_tcn = TCNModel()




    def set_input(self, input, load_label=True):
        pass

    def run(self):
        pass









if __name__ == '__main__':
    import sys
    sys.path.append('/data8/hzp/evoked_emotion/EEV/utils')#
    # from tools import calc_total_dim
    from tools import get_each_dim
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'efficientnet,trill_distilled'
        norm_features = 'trill_distilled'
        # max_seq_len = 60
        # gru_layers = 2
        hidden_size = '128'
        # input_dim = calc_total_dim(list(map(lambda x: x.strip(), feature_set.split(',')))) #计算出拼接后向量的维度
        input_dim = get_each_dim(list(map(lambda x: x.strip(), feature_set.split(',')))) #得到每个特征对应的向量维度
        lr = 1e-3
        beta1 = 0.5
        batch_size = 8
        epoch_count = 1
        niter=20
        niter_decay=30
        gpu_ids = 0
        isTrain = True
        checkpoints_dir = ''
        name = ''
        cuda_benchmark = ''
        dropout_rate = 0.3
        loss_type = 'mse'
        dataset_mode = 'eev'
        serial_batches = True
        num_threads = 0
        max_dataset_size = float("inf")
        tcn_channels = '(512),(512,512)'

    opt = test()
    net_a = EEV1stModel(opt)








    # dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

    # total_iters = 0                             # the total number of training iterations
    # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    #     epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    #     for i, data in enumerate(dataset):  # inner loop within one epoch
    #         total_iters += 1                # opt.batch_size
    #         epoch_iter += opt.batch_size
    #         net_a.set_input(data)           # unpack data from dataset and apply preprocessing
    #         net_a.run()