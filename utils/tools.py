import h5py
import os, glob

def get_dim(feat_name):
    dim_dict = {
        # 'landmarks_3d': 204,
        # 'fasttext': 300, 
        # 'pdm': 40, 
        # 'xception': 2048, 
        # 'landmarks_2d': 136, 
        # 'vggface': 512, 
        # 'pose': 6, 
        # 'egemaps': 88, 
        # 'bert': 768,
        # 'gocar': 350, 
        # 'gaze': 120, 
        # 'openpose': 54, 
        # 'au': 35, 
        # 'deepspectrum': 4096,
        # 'bert_base_cover': 768,
        # 'bert_medium_cover': 512,
        # 'bert_mini_cover': 256,
        # 'albert_base_cover': 768,
        # 'vggish': 128,
        # 'denseface': 342,
        # 'glove': 300,
        # 'senet50': 256,
        # 'noisy': 1536,
        # 'effnet_finetune': 256,
        # 'effnet_finetune_e7': 256,
        # 'effnet_finetune_aug': 256,
        # 'vgg16': 512,
        # 'lld': 130,
        # 'wav2vec': 768,
        # 'compare_mean': 130,
        # 'compare_downsample': 130,
        # 'word2vec': 300,
        # 'FAU': 35,
        # 'head_pose': 3,
        # 'eye_gaze': 120,
        # 'gaze_pattern': 18,
        # 'timesformer': 768,
        # 'openpose': 28,
        # 'optical_flow2': 1024,
        # 'BPM': 1,
        # 'ECG': 1,
        # 'resp': 1,
        # 'wav2vec_german': 1024
        'vggish': 128,
        'inception': 2048,
        # 'facenet': 1024,
        'efficientnet': 1280,
        'trill_distilled': 2048
    }
    if dim_dict.get(feat_name) is not None:
        return dim_dict[feat_name]
    else:
        return dim_dict[feat_name.split('_')[0]]

def calc_total_dim(feature_set):
    return sum(map(lambda x: get_dim(x), feature_set))

def get_each_dim(feature_set):
    return list(map(lambda x: get_dim(x), feature_set))

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# def get_all_dims():
#     root = '/data7/lrc/MuSe2020/MuSe2020_features/wild/feature'
#     h5s = glob.glob(os.path.join(root, '*.h5'))
#     ans = {}
#     for h5 in h5s:
#         name = h5.split('/')[-1].split('.')[0]
#         h5f = h5py.File(h5, 'r')
#         size = h5f['trn']['100']['feature'][()].shape[1]
#         ans[name] = size
    
#     print(ans)

# get_all_dims()