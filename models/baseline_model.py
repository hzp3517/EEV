import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .base_model import BaseModel
# from .networks.grus import GrusModel
# from .networks.context_gating import ContextGatingModel
# from .networks.moe import MixtureOfExpertsModel
# from .networks.loss import PCCLoss

import sys
sys.path.append('/data8/hzp/evoked_emotion/EEV/')#
from models.base_model import BaseModel#
from models.networks.grus import GrusModel#
from models.networks.context_gating import ContextGatingModel#
from models.networks.moe import MixtureOfExpertsModel#
from models.networks.loss import PCCLoss#

class BaselineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=60, help='max sequence length of gru')
        parser.add_argument('--gru_layers', default=2, type=int, help='gru layers')
        parser.add_argument('--expert_num', type=int, default=10, help='number of experts used in MoE model')
        parser.add_argument('--hidden_size', default='512,128', type=str, help='size of gru hidden layer, split by comma')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of grus')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'pcc'], help='use MSEloss or PCCloss')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the BaselineModel class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = [opt.loss_type]
        self.model_names = ['_grus', '_cg_1', '_moe', '_cg_2'] #grus对应的就是多个gru以及最后的concat过程


        self.gru_layers = opt.gru_layers
        self.pretrained_model = []

        self.max_seq_len = opt.max_seq_len
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.hidden_size_list = list(map(lambda x: int(x.strip()), opt.hidden_size.split(',')))
        assert len(self.feature_set) == len(self.hidden_size_list)
        self.num_gru = len(self.feature_set) #gru的数量，即特征的数量
        
        # net gru
        self.net_grus = GrusModel(self.num_gru, opt.input_dim, self.hidden_size_list, self.gru_layers, opt.dropout_rate)

        fusion_ft_size = 0
        for i in self.hidden_size_list:
            fusion_ft_size += i

        # net context gating 1
        self.net_cg_1 = ContextGatingModel(fusion_ft_size)

        # net MoE
        self.net_moe = MixtureOfExpertsModel(fusion_ft_size, output_size=15, expert_num=opt.expert_num) #15个表情类别

        # net context gating 2
        self.net_cg_2 = ContextGatingModel(15)

        #settings
        if self.isTrain:
            if opt.loss_type == 'mse':
                self.criterion_reg = torch.nn.MSELoss(reduction='sum') #这里可能需要改成mean试一下
            elif opt.loss_type == 'pcc':
                self.criterion_reg = PCCLoss()
            else:
                self.criterion_reg = torch.nn.L1Loss(reduction='sum')
            
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        # self.feature = input['feature'].to(self.device)
        self.feature_list = [i.to(self.device) for i in input['feature_list']]
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if load_label:
            self.target = input['label'].to(self.device)
            self.valid = input['valid'].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.feature_list[0].size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        # previous_h = torch.zeros(self.gru_layers, batch_size, self.hidden_size_list[i])
        # previous_h_list = [torch.zeros(self.gru_layers, batch_size, i) for i in self.hidden_size_list]
        previous_h_list = [torch.zeros(self.gru_layers, batch_size, i).to(self.device) for i in self.hidden_size_list]
        # previous_h_list = previous_h_list.to(self.device)#

        for step in range(split_seg_num):
            feature_step_list = [i[:, step*self.max_seq_len: (step+1)*self.max_seq_len] for i in self.feature_list]
            prediction, previous_h_list = self.forward_step(feature_step_list, previous_h_list)

            for previous_h in previous_h_list:
                previous_h = previous_h.detach_()

            self.output.append(prediction)
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                valid = self.valid[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask, valid)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)

    def forward_step(self, ft_list, state_list):
        assert self.num_gru == len(ft_list) == len(state_list)
        cat_r_out, hidden_list = self.net_grus(ft_list, state_list) 

        # print(cat_r_out.shape)

        cg1_out = self.net_cg_1(cat_r_out)
        moe_out = self.net_moe(cg1_out)
        out = self.net_cg_2(moe_out)

        # print(out.shape)#

        return out, hidden_list


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


# if __name__ == '__main__':
#     import sys
#     sys.path.append('/data8/hzp/evoked_emotion/EEV/utils')#
#     # from tools import calc_total_dim
#     from tools import get_each_dim
#     from data import create_dataset, create_dataset_with_args

#     class test:
#         feature_set = 'inception,vggish'
#         norm_features = 'vggish'
#         max_seq_len = 60
#         gru_layers = 2
#         hidden_size = '512,128'
#         # input_dim = calc_total_dim(list(map(lambda x: x.strip(), feature_set.split(',')))) #计算出拼接后向量的维度
#         input_dim = get_each_dim(list(map(lambda x: x.strip(), feature_set.split(',')))) #得到每个特征对应的向量维度
#         lr = 1e-4
#         beta1 = 0.5
#         batch_size = 8
#         epoch_count = 1
#         niter=20
#         niter_decay=30
#         gpu_ids = 0
#         isTrain = True
#         checkpoints_dir = ''
#         name = ''
#         cuda_benchmark = ''
#         dropout_rate = 0.3
#         loss_type = 'mse'
#         dataset_mode = 'eev'
#         serial_batches = True
#         num_threads = 0
#         max_dataset_size = float("inf")
#         expert_num = 10

#     opt = test()
#     net_a = BaselineModel(opt)

#     dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

#     total_iters = 0                             # the total number of training iterations
#     for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
#         epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
#         for i, data in enumerate(dataset):  # inner loop within one epoch
#             total_iters += 1                # opt.batch_size
#             epoch_iter += opt.batch_size
#             net_a.set_input(data)           # unpack data from dataset and apply preprocessing
#             net_a.run()
        


        
