import os
import glob
import numpy as np
from framework_pytorch.core import BasicDataset
from framework_pytorch.utils.utils import load_file, dict_concat

class CMRDataset(BasicDataset):
    def __init__(self,
                data_1,
                data_2,
                data_3,
                data_suffix,
                preprocesses=None,
                augmentation=None,
                shuffle=False,
                seg_suffix=None):
        super().__init__(data_1, data_suffix, preprocesses, augmentation)
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3
        self.shuffle = shuffle
        self.seg_suffix = seg_suffix

    def __getitem__(self, idx):
        if self.shuffle and idx == 0:
            np.random.shuffle(self.data_1)
            if self.data_2 is not None:  
                np.random.shuffle(self.data_2)
                np.random.shuffle(self.data_3)

        x1_name = self.data_1[idx]
        data_dict = self.load_dict(x1_name)
        
        if self.data_2 is not None and self.data_3 is not None:
            x2_name = self.data_2[idx%len(self.data_2)]
            x3_name = self.data_3[idx%len(self.data_3)]
            data_dict_2 = self.load_dict(x2_name)
            data_dict_3 = self.load_dict(x3_name)

            data_dict = dict_concat(data_dict, data_dict_2)
            data_dict = dict_concat(data_dict, data_dict_3)



        return data_dict

    def load_dict(self, fname):
        
        data_dict = {}
        data_dict.update({self._org_suffix: load_file(fname)})
        if self._other_suffix[0] is not None:
            lab = int(os.path.basename(os.path.dirname(os.path.dirname(fname))).split('_')[-1])
            data_dict.update({self._other_suffix[0]: lab})
        # if self.seg_suffix is not None and self.data_2 is None and self.data_3 is None:
        if self.seg_suffix is not None:
            s_name = fname.replace(self._org_suffix, self.seg_suffix)
            data_dict.update({self.seg_suffix: load_file(s_name)})
        
        data_dict = self.augmentation(data_dict)
        data_dict = self.pre_process(data_dict)

        return data_dict


    # def __getitem__(self, idx):
    #     if self.shuffle and idx == 0:
    #         np.random.shuffle(self.data_1)
    #         np.random.shuffle(self.data_2)
    #         np.random.shuffle(self.data_3)

        

    #     x1_name = self.data_1[idx]
    #     x2_name = self.data_2[idx%len(self.data_2)]
    #     x3_name = self.data_3[idx%len(self.data_3)]

    #     data_dict_1 = self.load_dict(x1_name)
    #     data_dict_2 = self.load_dict(x2_name)
    #     data_dict_3 = self.load_dict(x3_name)

    #     data_dict = dict_concat(data_dict_1, data_dict_2)
    #     data_dict = dict_concat(data_dict, data_dict_3)

    #     return data_dict

    # def load_dict(self, fpath):
    #     data_list = glob.glob(fpath + '/*' + self._org_suffix)
    #     lab = int(os.path.basename(os.path.dirname(fpath)).split('_')[-1])
    #     data_dict = {}
    #     for fname in data_list:
    #         sub_data_dict = {}
    #         sub_data_dict.update({self._org_suffix: load_file(fname)})
    #         sub_data_dict.update({self._other_suffix[0]: lab})
    #         sub_data_dict = self.augmentation(sub_data_dict)
    #         sub_data_dict = self.pre_process(sub_data_dict)

    #         data_dict = dict_concat(data_dict, sub_data_dict)
    #     return data_dict
