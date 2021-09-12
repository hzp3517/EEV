import torch
import torch.nn as nn
import torch.nn.functional as F

class dp(nn.Module):
    ''' dp model
    '''
    def __init__(self, dropout=0.3):
        super(dp, self).__init__()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        return self.dropout(x)

if __name__ == 'main':
    model = dp()
    input = torch.rand((8, 60, 300)) #(bs, seq_len, input_size)
    out = model(input)
    print(out.shape)
    print(out)

