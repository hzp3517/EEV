import os
import pickle as pkl
import numpy as np
import re
from nltk import word_tokenize
from tqdm import tqdm
import gensim

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

def get_word2vec_model(word2vec_src, save_path='German_word2vec_dict.pkl'):
    if os.path.exists(save_path):
        ans = pkl.load(open(save_path, 'rb'))
        return ans

    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_src, binary=True)
    vocab = model.vocab
    vocab_list = vocab.keys()

    word2vecs = {}
    for word in vocab_list:
        if vocab.get(word) is not None:
            word2vecs[word] = model.get_vector(word) #vector: ndarray

    print('word2vec model init at {}'.format(save_path))
    pkl.dump(word2vecs, open(save_path, 'wb'))
    return word2vecs

class Word2vecDictionary(object):
    def __init__(self, word2vec_src='/data2/zjm/tools/word2vec_models/gensim.german.model'): #德文词典模型
        self.model = get_word2vec_model(word2vec_src)

    def get_word2vec_wrd(self, word): # 送入一个词
        if isinstance(self.model.get(word), np.ndarray):
            return self.model[word]
        elif isinstance(self.model.get(str.capitalize(word)), np.ndarray): #将单词转换为首字母大写的形式，再去词典中匹配
            return self.model[str.capitalize(word)]
        else:
            return np.zeros(300)
        
    def __call__(self, word):  # 送入原德语单词
        word = replace_umlauts(word) #将德语特殊字符替换为普通的英文字母
        return self.get_word2vec_wrd(word)


if __name__ == '__main__':
    import pandas as pd
    csv_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/transcription_segments/47/47_10.csv'
    df = pd.read_csv(csv_path)
    wrd_lst = np.array(df['word']).tolist()
    wrd_lst = ['ich', 'habe', 'mich', 'hier', 'beworben', 'damit', 'ich', 'natürlich', 'euer', 'team']
    gd = Word2vecDictionary()
    for word in wrd_lst:
        ft = gd(word)
        #print(ft)
        print(ft.shape)