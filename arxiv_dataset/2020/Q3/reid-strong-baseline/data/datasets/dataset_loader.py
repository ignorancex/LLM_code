# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import io
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from data.transforms.customized_data_augmentation import part_substitution
from RAP_script.rap_data_loading import load_rap_dataset
from RAP_script import rap_data_loading


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, rap_data_=None, transform=None, is_train=True, swap_roi_rou=False):
        self.dataset = dataset
        self.transform = transform
        self.rap_data_ = rap_data_
        self.is_train = is_train
        self.swap_roi_rou=swap_roi_rou
        self.image_obj = part_substitution(probability=0.3, rap_data__=rap_data_,constraint_funcs=None, other_attrs=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, img_labels = self.dataset[index]
        #img_path, pid, camid = self.dataset[index]
        img = read_image(img_path) # img is read as PIL image
        if self.transform is not None:
            if self.is_train:
                if self.swap_roi_rou: # Augment the data with exchanging the region of interest with region of uninterest?
                    while True:
                        imge = self.image_obj(current_image_path=img_path)
                        if imge is not None:
                            break

                    if not isinstance(imge, Image.Image): # convert the image type to PIL if it is not already
                        img = Image.fromarray(imge, 'RGB')
                    else:
                        img = imge

                img = self.transform(img) # in this line **train** transformation (other augmentations) is applied

            else: # validation of test phase
                if self.swap_roi_rou:  # Augment the data with exchanging the region of interest with region of uninterest?
                    while True:
                        imge = self.image_obj(current_image_path=img_path)
                        if imge is not None:
                            break

                    if not isinstance(imge, Image.Image):  # convert the image type to PIL if it is not already
                        img = Image.fromarray(imge, 'RGB')
                    else:
                        img = imge

                img = self.transform(img) # in this line **val** transformation is applied

        return img, pid, camid, img_path
