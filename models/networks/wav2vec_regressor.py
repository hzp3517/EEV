import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class Wav2VecRegressor(Wav2Vec2PreTrainedModel):
    '''
    model = Wav2VecRegressor.from_pretrained('xxx_model')
    '''
    def __init__(self, config): 
        super().__init__(config)
        self.wav2vec = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.reg_layer = nn.Linear(config.hidden_size, 1) #一层线性回归层
        self.init_weights() #初始化self.reg_layer中的参数

    def forward(self, A_feats, mask_ft, start_timestamp, end_timestamp):
        # 1. Feed the input to Wav2vec model to obtain contextualized representations
        outputs = self.wav2vec(
            A_feats,
            attention_mask=mask_ft,
            output_hidden_states=True)
        hidden_state = outputs.last_hidden_state
        # print(hidden_state.shape) #tensor(bs, num_window, hidden_size=1024)，其中num_window=音频长度(s) / 0.02 - 1

        # 2. tensor(batch_size, num_window, hidden_size) -> tensor(batch_size, len_timestamps, hidden_size)
        aligned_hidden_states = []
        mask_ts = []
        batch_size = len(hidden_state)
        for batch_idx in range(batch_size):
            e_idx = None
            for i in range(len(mask_ft[batch_idx])):
                if mask_ft[batch_idx][i] == 0:
                    e_idx = i
                    break
            if mask_ft[batch_idx][-1] == 1:
                e_idx = len(mask_ft[batch_idx])
            hs = hidden_state[batch_idx][0: e_idx] #tensor(valid_num_window, hidden_size)
            valid_num_window = len(hs)

            num_ts = int((end_timestamp[batch_idx] - start_timestamp[batch_idx]) // 500 + 1)
            aligned_hs_lists = []
            num_win_per_ts = 25 # 500 / step(20) = 25
            for i in range(num_ts):
                if (i + 1) * num_win_per_ts > valid_num_window:
                    if i * num_win_per_ts >= valid_num_window:
                        hs_ts = aligned_hs_lists[-1] #如果整段500ms区间中均无有效的特征，则直接以上一个500ms的平均特征填充
                    else:
                        hs_ts = hs[i * num_win_per_ts: valid_num_window]
                        hs_ts = hs_ts.mean(dim=0)
                else:
                    hs_ts = hs[i * num_win_per_ts: (i + 1) * num_win_per_ts]
                    hs_ts = hs_ts.mean(dim=0)
                aligned_hs_lists.append(hs_ts)
            aligned_hs = torch.stack(aligned_hs_lists)
            aligned_hidden_states.append(aligned_hs)
            mask_ts.append(torch.ones(len(aligned_hs)).long())
        aligned_hidden_states = pad_sequence(aligned_hidden_states, padding_value=torch.tensor(0), batch_first=True)
        mask_ts = pad_sequence(mask_ts, padding_value=torch.tensor(0), batch_first=True).long()

        # 3. 回归层
        # tensor(len_timestamps, hidden_size) -> tensor(len_timestamps, 1)
        pred = self.reg_layer(aligned_hidden_states)
        pred = pred.squeeze(dim=2)
        if pred.device != torch.device('cpu'):
            mask_ts = mask_ts.cuda(pred.device) #将mask_ts与pred放到同一块卡上，便于之后直接计算
        return pred, mask_ts
        


        





if __name__ == '__main__':
    wav2vec_path = '/data8/hzp/models/wav2vec-xlsr-german/'
    wav2vec = Wav2VecRegressor.from_pretrained(wav2vec_path)

    A_feat = torch.tensor(np.load('/data12/MUSE2021/data/A_feat.npy'))
    mask_ft = torch.tensor(np.load('/data12/MUSE2021/data/mask_ft.npy'))

    start_ts = torch.tensor([  500.,  5000., 10500.])
    end_ts = torch.tensor([ 4500., 10000., 15500.])

    arousal = torch.tensor([[-0.0015,  0.0182,  0.0397,  0.0649,  0.0905,  0.1163,  0.1434,  0.1716,
          0.2023,  0.0000,  0.0000],
        [ 0.2355,  0.2691,  0.3005,  0.3311,  0.3606,  0.3823,  0.3993,  0.4153,
          0.4295,  0.4394,  0.4476],
        [ 0.4547,  0.4613,  0.4671,  0.4702,  0.4702,  0.4693,  0.4685,  0.4669,
          0.4637,  0.4601,  0.4567]])

    valence = torch.tensor([[0.0539, 0.0619, 0.0716, 0.0819, 0.0926, 0.1067, 0.1225, 0.1383, 0.1532,
         0.0000, 0.0000],
        [0.1663, 0.1778, 0.1871, 0.1973, 0.2059, 0.2144, 0.2221, 0.2339, 0.2461,
         0.2570, 0.2668],
        [0.2736, 0.2806, 0.2877, 0.2957, 0.3066, 0.3206, 0.3369, 0.3525, 0.3686,
         0.3853, 0.4004]])

    anno12_EDA = torch.tensor([[0.0370, 0.0523, 0.0703, 0.0911, 0.1147, 0.1411, 0.1702, 0.2019, 0.2380,
         0.0000, 0.0000],
        [0.2748, 0.3113, 0.3477, 0.3823, 0.4158, 0.4478, 0.4778, 0.5055, 0.5308,
         0.5535, 0.5734],
        [0.5905, 0.6049, 0.6164, 0.6256, 0.6325, 0.6371, 0.6398, 0.6408, 0.6407,
         0.6395, 0.6377]])

    pred, mask_ts = wav2vec(A_feat, mask_ft, start_ts, end_ts)

    print(pred)
    print(mask_ts)