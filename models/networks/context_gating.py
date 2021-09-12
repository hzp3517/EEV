import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextGatingModel(nn.Module):
    '''
    context gating model
    '''
    def __init__(self, input_size):
        super(ContextGatingModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size) #不改变size
        self.bn = nn.BatchNorm1d(input_size) #做batchnorm1d的时候是把中间的维度视作卷积通道了
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_t = self.linear(x)
        x_t = self.bn(x_t.transpose(1, 2)).transpose(1, 2)
        x_t = self.sigmoid(x_t)
        output = x_t * x #逐元素相乘
        return output


if __name__ == '__main__':
    model = ContextGatingModel(512)
    input = torch.rand((8, 60, 512)) # (bs:8, seq_len:60, ft_dim:512)
    out = model(input)
    print(model)
    print(out.shape)

