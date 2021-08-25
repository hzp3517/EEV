import os
import pickle as pkl
import numpy as np
import re
from nltk import word_tokenize
from tqdm import tqdm

def replace_umlauts(text):
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res

def get_glove_model(glove_src, save_path='German_glove_dict.pkl'):
    if os.path.exists(save_path):
        ans = pkl.load(open(save_path, 'rb'))
        return ans
    
    ans = {}
    for line in tqdm(open(glove_src, encoding='utf8').readlines(), 
                desc='Building Glove'):
        line = line.strip()
        elements = line.split(' ')
        word = elements[0]
        embd = list(map(lambda x: float(x), elements[1:]))
        ans[word] = np.array(embd)
    
    print('Glove model init at {}'.format(save_path))
    pkl.dump(ans, open(save_path, 'wb'))
    return ans

class GloveDictionary(object):
    def __init__(self, glove_src='/data8/hzp/tools/glove/german_glove/glove.news.2013.300d.txt'): #德文词典
        self.model = get_glove_model(glove_src)

    def get_glove_wrd(self, word): # 送入一个词
        if isinstance(self.model.get(word), np.ndarray):
            return self.model[word]
        else:
            return np.zeros(300)
        
    def __call__(self, word):  # 送入原德语单词
        word = replace_umlauts(word) #将德语特殊字符替换为普通的英文字母

        print(word, end='\t')#

        return self.get_glove_wrd(word)



if __name__ == '__main__':
    #wrd_lst = ['gewissenhaftigkeit']
    #word = 'beworben'

    import pandas as pd
    csv_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/transcription_segments/10/10_1.csv'
    df = pd.read_csv(csv_path)
    wrd_lst = np.array(df['word']).tolist()
    #wrd_lst = ['Happiness', 'is', 'a', 'way', 'station', 'between', 'too', 'much', 'and', 'too', 'little']
    gd = GloveDictionary()
    for word in wrd_lst:
        ft = gd(word)
        if ft.any():
            print('not 0')
        else:
            print('all 0')
        #print(ft)