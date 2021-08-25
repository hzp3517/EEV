import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
# from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import (
    RobertaModel, 
    RobertaForSequenceClassification, 
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel
)
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
# from models.networks.tools import init_weights


class Wav2VecClassifier(Wav2Vec2PreTrainedModel):
    def __init__(self, config, num_classes, embd_method): # 
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        if self.embd_method not in ['last', 'mean', 'max']:
            raise NotImplementedError('Only [last, mean, max] embd_method is supported, \
                but got', embd_method)

        self.model = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.model(
            input_ids,
            # attention_mask=attention_mask,
            # output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        # using different embed method
        if self.embd_method == 'last':
            cls_reps = last_hidden[:, -1]
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits, last_hidden

def bert_classifier(num_classes, bert_name):
    model = BertForSequenceClassification.from_pretrained(
        bert_name, num_labels=num_classes,
        output_attentions=False, output_hidden_states=True
    )
    return model

def roberta_classifier(num_classes, bert_name):
    model = RobertaForSequenceClassification.from_pretrained(
        bert_name, num_labels=num_classes,
        output_attentions=False, output_hidden_states=True
    )
    return model