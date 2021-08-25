import os
import pickle as pkl
import numpy as np
import re
from nltk import word_tokenize
from tqdm import tqdm

# def wash(text):
#     punc = '~`!#$%^&*()\[\]_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
#     ans = ''
#     for _char in text:
#         if _char in punc:
#             ans += ' ' + _char + ' '
#         else:
#             ans += _char
#     return ans.lower().replace('  ', ' ').rstrip(' ')

# def wash(text):
#     punc = '~`!#$%^&*()\[\]_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
#     text = re.sub(r"[%s]+" %punc, "",text)
#     return text

def get_glove_model(glove_src, save_path='glove_dict.pkl'):
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
    def __init__(self, glove_src='/data2/zjm/tools/word2vec_models/glove.840B.300d.txt'):
        self.model = get_glove_model(glove_src)

    def get_glove_wrd(self, word): # 送入一个词
        if isinstance(self.model.get(word), np.ndarray):
            return self.model[word]
        else:
            return np.zeros(300)
        
    def __call__(self, sentence):  # 送入一个句子
        wrd_lst = word_tokenize(sentence)
        ft = np.array([self.get_glove_wrd(wrd) for wrd in wrd_lst])
        return ft


if __name__ == '__main__':
    sentence = 'It\'s a cat!'
    gd = GloveDictionary()
    ft = gd(sentence)
    print(ft.shape)