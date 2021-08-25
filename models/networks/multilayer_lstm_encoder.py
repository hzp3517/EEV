import typing_extensions
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F


class MultilayerLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_n_layers):
        '''
        input_size: LSTM的输入维度
        hidden_size: LSTM的hidden维度
        rnn_n_layers: LSTM的层数
        '''
        super(MultilayerLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_n_layers = rnn_n_layers
        assert self.rnn_n_layers >= 1
        self.rnn_list = nn.ModuleList()
        for i in range(self.rnn_n_layers):
            self.rnn_list.append(nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=1))

    def forward(self, x):
        '''
        x: 传入的tensor
        '''
        post_h_list = [] #要返回的h列表
        post_c_list = [] #要返回的c列表
        input = x
        for i in range(self.rnn_n_layers):
            hidden, (h, c) = self.rnn_list[i](input) #lstm
            hidden = hidden + input
            input = hidden
            post_h_list.append(h)
            post_c_list.append(c)
        return hidden, (post_h_list, post_c_list)

class MultilayerBiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_n_layers):
        '''
        input_size: LSTM的输入维度
        hidden_size: LSTM的hidden维度
        rnn_n_layers: LSTM的层数
        '''
        super(MultilayerBiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_n_layers = rnn_n_layers
        assert self.rnn_n_layers >= 1
        self.rnn_list = nn.ModuleList()
        self.cnn_list = nn.ModuleList()
        for _ in range(self.rnn_n_layers - 1):
            self.rnn_list.append(nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True, num_layers=1))
            self.cnn_list.append(nn.Conv1d(2*hidden_size, hidden_size, 3, stride=1, padding=1)) #in_channels, out_channels, kernel_size, stride, padding
        self.rnn_list.append(nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True, num_layers=1))

    def expand_dim(self, input):
        '''
        input.shape: tensor(bs, len, dim)
        output shape:tensor(bs, len, 2 * dim)
        '''
        input = input.permute(2, 0, 1)
        res_list = []
        for sub_input in input:
            res_list.append(sub_input)
            res_list.append(sub_input)
        output = torch.stack(res_list, dim=2)
        return output

    def forward(self, x):
        '''
        先进行残差运算，再经过cnn
        '''
        post_h_list = [] #要返回的h列表
        post_c_list = [] #要返回的c列表
        input = x
        res = self.expand_dim(input) #fc输出在正反向上重复
        for i in range(self.rnn_n_layers - 1):
            hidden, (h, c) = self.rnn_list[i](input) #lstm
            res = res + hidden
            per_res = res.permute(0, 2, 1) #[bs, len, channel] -> [bs, channel, len]
            input = self.cnn_list[i](per_res)
            input = input.permute(0, 2, 1) #[bs, channel, len] -> [bs, len, channel]
            post_h_list.append(h)
            post_c_list.append(c)
        hidden, (h, c) = self.rnn_list[-1](input) #lstm
        res = res + hidden
        post_h_list.append(h)
        post_c_list.append(c)
        return hidden, (post_h_list, post_c_list)

if __name__ == '__main__':
    # model = MultilayerLSTMEncoder(128, 128, 3)
    # input = torch.rand(8, 30, 128)
    # r_out, post_states = model(input)
    # print(r_out.shape)

    model = MultilayerBiLSTMEncoder(128, 128, 3)
    input = torch.rand(8, 30, 128)
    r_out, post_states = model(input)
    print(r_out.shape)



