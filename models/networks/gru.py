import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class GruModel(nn.Module):
    ''' Gru model
    '''
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(GruModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        # GRU后还要再加一个dropout。因为nn.GRU中加入的dropout不包含最后一层: "introduces a Dropout layer on the outputs of each GRU layer 'except the last layer'"，
        #   但文章中说For the GRU, we apply dropout of 0.3 in "each layer".
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, state):
        '''
        Parameters:
        ------------------------
        x: input feature seqences
        states: (h_0, c_0)
        '''
        r_out, hidden = self.rnn(x, state)
        r_out = self.dropout(r_out)
        return r_out, hidden

if __name__ == '__main__':
    model = GruModel(300, 128, num_layers=2, dropout=0.3)
    input = torch.rand((8, 60, 300)) #(bs, seq_len, input_size)
    state = torch.zeros((2, 8, 128)) #(num_layers, bs, hidden_size)
    out, state = model(input, state)
    print(model)
    print(out.shape, state.shape)
