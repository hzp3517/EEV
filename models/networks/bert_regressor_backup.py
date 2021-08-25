import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel


# class BertRegressor(BertPreTrainedModel):


class BertClassifier(BertPreTrainedModel):
    '''
    model = BertClassifier.from_pretrained('xxx_model')
    '''
    def __init__(self, config, num_classes, embd_method): # 
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        if self.embd_method not in ['cls', 'mean', 'max']:
            raise NotImplementedError('Only [cls, mean, max] embd_method is supported, \
                but got', config.embd_method)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels) #这就是一个分类层
        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        # Feed the input to Bert model to obtain contextualized representations
        '''
        ['i']
        '''
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        cls_token = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # using different embed method
        if self.embd_method == 'cls':
            cls_reps = cls_token
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits, hidden_states


if __name__ == '__main__':
    bert_path = '/data8/hzp/models/bert-base-german-uncased/'
    num_classes = 4
    embd_method = 'cls'
    bert = BertClassifier.from_pretrained(bert_path, num_classes=num_classes, embd_method=embd_method)
    #print(bert)



