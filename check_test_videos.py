'''
对比一下官方给出的test集合无效video ids和本地无效的video ids
'''

import os
import h5py
import numpy as np

target_dir = '/data8/hzp/evoked_emotion/EEV_process_data/target/'
official_invalid_path = 'official_invalid_test_videos.txt'

official_invalid_test_ids = []
with open(official_invalid_path, 'r') as f:
    official_invalid_test_ids = f.readlines()
official_invalid_test_ids = [i.strip() for i in official_invalid_test_ids]
# print(official_invalid_test_ids)

partition_h5f = h5py.File(os.path.join(target_dir, 'partition.h5'), 'r')
local_invalid_test_ids = partition_h5f['test']['invalid'][()]
local_invalid_test_ids = [i.decode() for i in local_invalid_test_ids]
# print(local_test_ids)


#-----官方无效列表里没有本地却无效的-------
print('--------------local invalid but official valid---------------')
for id in local_invalid_test_ids:
    if id not in official_invalid_test_ids:
        print(id)

#-----官方无效列表里有但本地有效的-------
print('--------------local valid though official invalid---------------')
for id in official_invalid_test_ids:
    if id not in local_invalid_test_ids:
        print(id)



