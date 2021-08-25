import os
import numpy as np
import pandas as pd
import scipy.signal as spsig


class ComParEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=50, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            #opensmile_tool_dir = '/root/opensmile-2.3.0/'
            opensmile_tool_dir = '/data8/hzp/emo_bert/tools/opensmile-2.3.0'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = df.iloc[:, 2:]
        if self.downsample > 0:
            if len(wav_data) > self.downsample:
                wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:
                wav_data = None
                print(f'Error in {wav}, no feature extracted')

        return np.array(wav_data)


if __name__ == '__main__':
    audio_path = '/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/audio_segments/1/1.wav'
    compare = ComParEExtractor(downsample=-1)
    ft = compare(audio_path) #如果没能成功提取出特征，则返回None
    print(type(ft))
    print(ft.shape)

    a = np.zeros((1, 130))

    b = np.concatenate((ft, a), axis=0)
    print(b.shape)