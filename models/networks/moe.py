import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax

class MixtureOfExpertsModel(nn.Module):
    '''
    Mixture of Experts (MoE) Model
    '''
    def __init__(self, input_size, output_size=15, expert_num=10):
        super(MixtureOfExpertsModel, self).__init__()
        self.expert_num = expert_num
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(expert_num)])
        self.gating_linears = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(expert_num)])
        self.softmax = nn.Softmax(dim=-1) #输入数据shape: (bs, output_size, expert_num)，在expert_num维度上做softmax

    def forward(self, x):
        '''
        x.shape = (bs, input_size)
        '''
        miu, xi = [], []
        for i in range(self.expert_num):
            miu_i = self.experts[i](x)
            xi_i = self.gating_linears[i](x)
            miu.append(miu_i) # shape: (bs, output_size)
            xi.append(xi_i) 
        miu = torch.stack(miu, dim=-1) #shape: (bs, output_size, expert_num)
        xi = torch.stack(xi, dim=-1)
        g = self.softmax(xi)
        out = torch.sum(g * miu, dim=-1)
        return out #shape: (bs, output_size)

if __name__ == '__main__':
    model = MixtureOfExpertsModel(128)
    input = torch.rand((8, 128))
    out = model(input)
    print(model)
    print(out.shape)
