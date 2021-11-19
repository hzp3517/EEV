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

'''
只输入单个特征，多特征的融合通过ensemble实现
'''

class EEV1stModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('--max_seq_len', type=int, default=60, help='max sequence length of gru')
        parser.add_argument('--dropout_rate', default=0.2, type=float, help='drop out rate of TCN')
        parser.add_argument('--tcn_channels', default='512', help='number of TCN channels, split by the comma')
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
        self.tcn_channels = list(map(lambda x: int(x.strip()), opt.tcn_channels.split(',')))
        self.hidden_size_list = list(map(lambda x: int(x.strip()), opt.hidden_size.split(',')))

        assert len(opt.input_dim) == 1 # 只实现单个特征的预测，多个特征的融合直接通过ensemble实现
        self.input_dim = opt.input_dim[0]

        self.net_tcn = TCNModel(self.input_dim, self.tcn_channels, kernel_size=3, dropout=opt.dropout_rate)
        self.net_reg = FcRegressor(self.tcn_channels[-1], self.hidden_size_list, output_dim=15, 
                                   dropout=0, dropout_input=False)

        #settings
        if self.isTrain:
            if opt.loss_type == 'mse':
                self.criterion_reg = torch.nn.MSELoss(reduction='sum') #这里可能需要改成mean试一下
            elif opt.loss_type == 'pcc':
                self.criterion_reg = PCCLoss()
            else:
                self.criterion_reg = torch.nn.L1Loss(reduction='sum')
            
            parameters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            # self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.SGD(parameters, lr=opt.lr) #1st代码用的SGD

            self.optimizers.append(self.optimizer)

    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        # self.feature_list = [i.to(self.device) for i in input['feature_list']]
        self.feature = input['feature_list'][0].to(self.device) # 此版代码中只会接收一个特征 # (bs, seq_len, embd_dim)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if load_label:
            self.target = input['label'].to(self.device)
            self.valid = input['valid'].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature.size(0)
        batch_max_length = torch.max(self.length).item()

        # print(self.feature.shape) # (bs, seq_len, embd_dim)

        prediction = self.forward_step(self.feature)
        self.output = prediction

        # backward
        if self.isTrain:
            self.optimizer.zero_grad()
            self.backward_step(prediction, self.target, self.mask, self.valid)
            self.optimizer.step()

    def forward_step(self, ft):
        ft = ft.permute(0, 2, 1) # transfrom (bs, seq_len, embd_dim) to (bs, embd_dim, seq_len)
        tcn_out = self.net_tcn(ft)
        tcn_out = tcn_out.permute(0, 2, 1) # transfrom (bs, embd_dim, seq_len) to (bs, seq_len, embd_dim)
        out, _ = self.net_reg(tcn_out)
        return out

    def backward_step(self, pred, target, mask, valid):
        """Calculate the loss for back propagation"""
        valid = valid.unsqueeze(dim=2).repeat(1, 1, 15)
        pred = pred * mask * valid # /len(valid) 可能还要除以特征长度
        target = target * mask * valid
        batch_size = target.size(0)
        loss_name = self.loss_names[0]
        exec('self.loss_{} = self.criterion_reg(pred, target) / batch_size'.format(loss_name))
        exec('self.loss_{}.backward(retain_graph=False)'.format(loss_name))

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)


if __name__ == '__main__':
    import sys
    sys.path.append('/data8/hzp/evoked_emotion/EEV/utils')#
    # from tools import calc_total_dim
    from tools import get_each_dim
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'trill_distilled'
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
        tcn_channels = '512,512'
        # tcn_channels = '512'

    opt = test()
    net_a = EEV1stModel(opt)


    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

    total_iters = 0                             # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            net_a.set_input(data)           # unpack data from dataset and apply preprocessing
            net_a.run()