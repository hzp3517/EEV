import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
import h5py
import os
import numpy as np
from tqdm import tqdm
import glob
import csv
import json

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

class BertExtractor(object):
    def __init__(self, cuda=False, cuda_num=None):
        # self.tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')
        # self.model = BertModel.from_pretrained('dbmdz/bert-base-german-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('/data8/hzp/models/bert-base-german-uncased')
        self.model = BertModel.from_pretrained('/data8/hzp/models/bert-base-german-uncased')
        
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False
        
    def tokenize(self, word_lst):
        word_lst = ['[CLS]'] + word_lst + ['[SEP]']
        word_idx = []
        ids = []
        for idx, word in enumerate(word_lst):
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            token_ids = self.tokenizer.convert_tokens_to_ids(ws)
            ids.extend(token_ids)
            if word not in ['[CLS]', '[SEP]']:
                word_idx += [idx-1] * len(token_ids)
        return ids, word_idx
    
    def get_embd(self, token_ids):
        # token_ids = torch.tensor(token_ids)
        # print('TOKENIZER:', [self.tokenizer._convert_id_to_token(_id) for _id in token_ids])
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        if self.cuda:
            token_ids = token_ids.to(self.cuda_num)
            
        with torch.no_grad():
            outputs = self.model(token_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output
        #sequence_output：encoder端最后一层编码层的特征向量
        #pooled_output：[CLS]这个token对应的向量，把它作为整个句子的特征向量


if __name__ == '__main__':
    extractor = BertExtractor(cuda=True, cuda_num=7)

    import pandas as pd
    #csv_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/transcription_segments/47/47_12.csv'
    csv_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/transcription_segments/50/50_12.csv'
    df = pd.read_csv(csv_path)
    wrd_lst = np.array(df['word']).tolist()

    sentence = ' '.join(wrd_lst) #德文语句
    #print(sentence)

    sequence_feat, pooled_feat = extractor.extract(sentence)
    sequence_feat = sequence_feat.cpu()
    #print(sequence_feat)
    sequence_feat = sequence_feat.squeeze()
    #print(sequence_feat)
    print(sentence)

    # # -----一句话一起送--------
    # print('---------------------------------')
    tokenizer = BertTokenizer.from_pretrained('/data8/hzp/models/bert-base-german-uncased')
    # tokenizer = AutoTokenizer.from_pretrained('/data8/hzp/models/bert-base-german-uncased')
    # subword_sentence = tokenizer.tokenize(sentence)
    # # subword_sentence = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    # print(subword_sentence)

    #------逐个词送------------
    word_list = sentence.split(' ')
    print('---------------------------------')
    for word in word_list:
        #subword_word = tokenizer.tokenize(word)
        subword_word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        print(word, subword_word)

    # #-------[PAD]---------------
    # print('--------------------------')
    # subword_word = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]'))
    # print(word, subword_word)


    print(len(sentence.split(' ')))
    print(sequence_feat.shape)