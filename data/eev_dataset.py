import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
from tqdm import tqdm

import sys
sys.path.append('/data8/hzp/evoked_emotion/EEV/')#
from data.base_dataset import BaseDataset#
# from .base_dataset import BaseDataset


class EEVDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.add_argument('--normalize', type=bool, default=False, help='whether to normalize (on train set) or not.')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize (on train set), split by comma, eg: "egemaps,vggface"')
        return parser

    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test]
        '''
        super().__init__(opt)
        self.root = '/data8/hzp/evoked_emotion/EEV_process_data/'
        self.category_list = ['amusement', 'anger', 'awe', 'concentration', 'confusion', 'contempt', 'contentment', 
                        'disappointment', 'doubt', 'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph']
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        # self.normalize = opt.normalize
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.set_name = set_name
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"EEV dataset {set_name} created with total length: {len(self)}")

    def normalize_on_train(self, feature_name, features):
        '''
        features的shape：[len, ft_dim]
        mean_f与std_f的shape：[ft_dim,]，已经经过了去0处理
        '''
        mean_std_file = h5py.File(os.path.join(self.root, 'mean_std_on_train', feature_name + '.h5'), 'r')
        mean_train = np.array(mean_std_file['train']['mean'])
        std_train = np.array(mean_std_file['train']['std'])
        features = (features - mean_train) / std_train
        return features

    def load_label(self):
        # partition_h5f = h5py.File(os.path.join(self.root, 'target', 'partition.h5'), 'r')
        partition_h5f = h5py.File(os.path.join(self.root, 'target', 'toy_version', 'partition.h5'), 'r')
        self.video_ids = sorted(partition_h5f[self.set_name]['valid'])
        self.video_ids = list(map(lambda x: x.decode(), self.video_ids))
        # label_h5f = h5py.File(os.path.join(self.root, 'target', '{}_target.h5'.format(self.set_name)), 'r')
        label_h5f = h5py.File(os.path.join(self.root, 'target', 'toy_version', '{}_target.h5'.format(self.set_name)), 'r')
        self.target = {}
        # for _id in self.video_ids:
        print('load label:')#
        for _id in tqdm(self.video_ids, desc=self.set_name):#
            self.target[_id] = {}
            if self.set_name != 'test':
                self.target[_id]['timestamp'] = torch.from_numpy(label_h5f[_id]['timestamp'][()]).long()
                self.target[_id]['valid'] = torch.as_tensor(label_h5f[_id]['valid'][()]).long()
                self.target[_id]['length'] = torch.as_tensor(len(self.target[_id]['timestamp'])).long()#
                for cate in self.category_list:
                    self.target[_id][cate] = torch.from_numpy(label_h5f[_id][cate][()]).float()
            else:
                self.target[_id]['timestamp'] = torch.from_numpy(label_h5f[_id]['timestamp'][()]).long()
                self.target[_id]['length'] = torch.as_tensor(len(self.target[_id]['timestamp'])).long()#

    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            h5f = h5py.File(os.path.join(self.root, 'features', '{}.h5'.format(feature_name)), 'r')
            feature_data = {}
            # for _id in self.video_ids:
            print('load feature:')#
            for _id in tqdm(self.video_ids, desc=self.set_name):
                feature_data[_id] = h5f[self.set_name][_id]['feature'][()] #shape:(len, ft_dim)
                assert len(h5f[self.set_name][_id]['timestamp'][()]) == len(self.target[_id]['timestamp']), '\
                    Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                #normalize on train:
                if feature_name in self.norm_features:
                    feature_data[_id] = self.normalize_on_train(feature_name, feature_data[_id])
            self.feature_data[feature_name] = feature_data

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        feature_list = []
        feature_dim = []
        for feature_name in self.feature_set:
            data = torch.from_numpy(self.feature_data[feature_name][video_id]).float()
            feature_list.append(data)
            feature_dim.append(self.feature_data[feature_name][video_id].shape[1])
        feature_dim = torch.from_numpy(np.array(feature_dim)).long()

        target_data = self.target[video_id]
        return {**{"feature_list": feature_list, "feature_dims": feature_dim, "vid": video_id},
                **target_data, **{"feature_names": self.feature_set}}

    def __len__(self):
        return len(self.video_ids)

    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        feature_num = len(batch[0]['feature_list'])
        feature = []
        for i in range(feature_num):
            feature_name = self.feature_set[i]
            pad_ft = pad_sequence([sample['feature_list'][i] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            pad_ft = pad_ft.float()
            feature.append(pad_ft)
        feature = torch.cat(feature, dim=2) #pad_ft:(bs, seq_len, ft_dim)，将各特征拼接起来

        timestamp = pad_sequence([sample['timestamp'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        length = torch.tensor([sample['length'] for sample in batch])
        vid = [sample['vid'] for sample in batch]

        if self.set_name != 'test':
            valid = pad_sequence([sample['valid'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            label_dict = {}
            for cate in self.category_list:
                label_dict[cate] = pad_sequence([sample[cate] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)

        # feature_dims = batch[0]['feature_dims']
        feature_names = batch[0]['feature_names']
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0

        if self.set_name != 'test':
            return_dict = {}
            return_dict['feature'] = feature.float()
            return_dict['timestamp'] = timestamp.long()
            return_dict['mask'] = mask.float()
            return_dict['length'] = length
            return_dict['feature_names'] = feature_names
            return_dict['vid'] = vid
            for cate in self.category_list:
                return_dict[cate] = label_dict[cate]
            return return_dict
        else:
            return {
                'feature': feature.float(), 
                'timestamp': timestamp.long(),
                'mask': mask.float(),
                'length': length,
                'feature_names': feature_names,
                'vid': vid
            }


if __name__ == '__main__':
    class test:
        feature_set = 'vggish'
        dataroot = '/data8/hzp/evoked_emotion/EEV_process_data/'
        max_seq_len = 60
        norm_features = 'vggish'
    
    opt = test()
    a = EEVDataset(opt, 'train')
    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    print(batch_data['feature'].shape)

    print(batch_data[a.category_list[5]].shape)

    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['feature_names'])
    print(batch_data['vid'])
    # print(data['feature'].shape)
    # print(data['feature_lens'])
    # print(data['feature_names'])
    # print(data['length'])
