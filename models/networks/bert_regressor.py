import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel
# from .classifier import FcClassifier
from models.networks.classifier import FcClassifier


class BertRegressor(BertPreTrainedModel):
    '''
    model = BertRegressor.from_pretrained('xxx_model')
    '''
    #def __init__(self, config, _reg_layers, _reg_dropout=0.3): 
    def __init__(self, config): 
        '''
        config: BertConfig对象
        reg_layers: list(map(lambda x: int(x), opt.regress_layers.split(',')))，
            其中opt.regress_layers示例：'256,128'
        reg_dropout: 回归层的dropout率
        '''
        super().__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.reg_layer = nn.Linear(config.hidden_size, 1) #一层线性回归层
        #self.reg_net = FcClassifier(config.hidden_size, _reg_layers, 1, dropout=_reg_dropout)
        self.init_weights() #初始化self.reg_layer中的参数

    def forward(self, input_ids, attention_mask, cur_mask, word_start_times, word_end_times, timestamps):
        # 1. Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
        hidden_state = outputs.last_hidden_state
        #print(hidden_state.shape) #torch.Size([bs, len, hidden_size])

        # 2. tensor(batch_size, len_words, hidden_size) -> tensor(batch_size, len_timestamps, hidden_size)
        aligned_hidden_states = []
        mask_ts = []
        batch_size = len(hidden_state)
        for batch_idx in range(batch_size):
            sen_s_idx = None
            sen_e_idx = None
            for i in range(len(cur_mask[batch_idx])):
                if cur_mask[batch_idx][i] == 1:
                    sen_s_idx = i
                    break
            for i in range(sen_s_idx, len(cur_mask[batch_idx])):
                if cur_mask[batch_idx][i] == 0:
                    sen_e_idx = i
                    break
            if cur_mask[batch_idx][-1] == 1:
                #sen_e_idx = len(cur_mask[batch_idx]) + 1
                sen_e_idx = len(cur_mask[batch_idx])
            hs = hidden_state[batch_idx][sen_s_idx: sen_e_idx] #tensor(len_words, hidden_size)

            valid_timestamp = []
            for i in timestamps[batch_idx]:
                if i != 0:
                    valid_timestamp.append(i)

            aligned_hs_lists = []
            for _ in valid_timestamp:
                aligned_hs_lists.append([])
            sentence_start = valid_timestamp[0]
            for i in range(len(hs)): #遍历所有subword对应的hidden_state
                start = word_start_times[batch_idx][i]
                end = word_end_times[batch_idx][i]
                for ts in range(start, end + 500, 500):
                    ts_idx = (ts - sentence_start) // 500
                    aligned_hs_lists[ts_idx].append(hs[i]) #将该subword的hidden_state添加到其对应的时间戳列表中
            
            aligned_hs = []
            for hs_list in aligned_hs_lists:
                if len(hs_list) == 0:
                    aligned_hs.append(torch.zeros(self.hidden_size))
                else: 
                    aligned_hs.append(torch.mean(torch.stack(hs_list), dim=0))
            aligned_hs = torch.stack(aligned_hs) #应为tensor(len_timestamps, hidden_size)
            aligned_hidden_states.append(aligned_hs)
            mask_ts.append(torch.ones(len(aligned_hs)))
        aligned_hidden_states = pad_sequence(aligned_hidden_states, padding_value=torch.tensor(0), batch_first=True)
        mask_ts = pad_sequence(mask_ts, padding_value=torch.tensor(0), batch_first=True).long()

        # 3. 回归层
        # tensor(len_timestamps, hidden_size) -> tensor(len_timestamps, 1)
        # pred, _ = self.reg_layer(aligned_hidden_states)
        pred = self.reg_layer(aligned_hidden_states)
        pred = pred.squeeze(dim=2)
        if pred.device != torch.device('cpu'):
            mask_ts = mask_ts.cuda(pred.device) #将mask_ts与pred放到同一块卡上，便于之后直接计算
        return pred, mask_ts






if __name__ == '__main__':
    bert_path = '/data8/hzp/models/bert-base-german-uncased'
    # _reg_layers = [256, 128]
    # _reg_dropout = 0.3
    bert = BertRegressor.from_pretrained(bert_path)
    # bert = BertRegressor.from_pretrained(bert_path, _reg_layers=_reg_layers, _reg_dropout=_reg_dropout)
    #print(bert)

    input_ids = torch.tensor([[  102,     0,   103,     0,   103,   260,   103,   691,   103,   657,
           103,   532,   103, 28854,   103,   865,   103,   260,   103,  1896,
           181,   103,  9965,   103,  2042,   103,     0,   224,   832,   260,
           503,  1379,  2548,  1713,   260,   691,   281,   420,   142,  4360,
         16631,   456,   709,  2431,   709,   103,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [  102,     0,     0,   260,   691,   657,   532, 28854,   865,   260,
          1896,   181,  9965,  2042,   103,     0,   103,   224,   103,   832,
           103,   260,   103,   503,   103,  1379,  2548,   103,  1713,   103,
           260,   103,   691,   103,   281,   103,   420,   103,   142,   103,
          4360,   103, 16631,   456,   103,   709,   103,  2431,   103,   709,
           103,     0,  5639,   249, 18383,   197, 18084,  2129,   691,   951,
           750, 24106,  1661,   256,   197,  7778,  2129,   143,   103,     0,
             0,     0,     0,     0,     0,     0,     0],
        [  102,     0,   224,   832,   260,   503,  1379,  2548,  1713,   260,
           691,   281,   420,   142,  4360, 16631,   456,   709,  2431,   709,
           103,     0,   103,  5639,   103,   249, 18383,   103,   197,   103,
         18084,   103,  2129,   103,   691,   103,   951,   103,   750, 24106,
          1661,   256,   103,   197,   103,  7778,   103,  2129,   103,   143,
           103,     0, 10927, 30945,   924,  1896,   181,   788,  6925, 10315,
          3291,  5299,   466, 10360, 30940,   260, 10927, 30945,   924,   277,
          9586,  2042,  6255, 20296,   106,   271,   103]])

    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]])

    cur_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]])

    word_start_times = torch.tensor([[  500,   500,  1000,  1000,  1500,  1500,  1500,  1500,  1500,  1500,
          1500,  1500,  2000,  2000,  2500,  2500,  2500,  2500,  3000,  3000,
          3000,  4000,  4000,  4500,  4500,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0],
        [ 5000,  5000,  5500,  5500,  6000,  6000,  6000,  6000,  6000,  6000,
          6500,  6500,  6500,  7000,  7000,  7500,  7500,  8000,  8000,  8500,
          8500,  8500,  8500,  8500,  8500,  9000,  9000,  9000,  9000,  9000,
          9500,  9500,  9500,  9500, 10000, 10000],
        [10500, 10500, 11000, 11000, 11500, 11500, 11500, 12500, 12500, 12500,
         12500, 13000, 13000, 14000, 14000, 14000, 14000, 14500, 14500, 14500,
         14500, 14500, 15000, 15000, 15000, 15000, 15000, 15000, 15500, 15500,
             0,     0,     0,     0,     0,     0]])

    word_end_times = torch.tensor([[  500,   500,  1000,  1000,  1500,  1500,  1500,  1500,  1500,  1500,
          2000,  2000,  2500,  2500,  2500,  2500,  3000,  3000,  4000,  4000,
          4000,  4500,  4500,  4500,  4500,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0],
        [ 5000,  5000,  6000,  6000,  6000,  6000,  6000,  6000,  6500,  6500,
          7000,  7000,  7000,  7500,  7500,  8000,  8000,  8500,  8500,  8500,
          8500,  8500,  8500,  9000,  9000,  9000,  9000,  9500,  9500,  9500,
          9500,  9500, 10000, 10000, 10000, 10000],
        [10500, 10500, 11500, 11500, 12500, 12500, 12500, 12500, 12500, 13000,
         13000, 14000, 14000, 14000, 14000, 14500, 14500, 15000, 15000, 15000,
         15000, 15000, 15000, 15000, 15000, 15000, 15500, 15500, 15500, 15500,
             0,     0,     0,     0,     0,     0]])

    timestamps = torch.tensor([[  500,  1000,  1500,  2000,  2500,  3000,  3500,  4000,  4500,     0,
             0],
        [ 5000,  5500,  6000,  6500,  7000,  7500,  8000,  8500,  9000,  9500,
         10000],
        [10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000,
         15500]])

    bert(input_ids, attention_mask, cur_mask, word_start_times, word_end_times, timestamps)