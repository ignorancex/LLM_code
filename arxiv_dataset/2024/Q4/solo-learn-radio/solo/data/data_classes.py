import os
import os.path
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clip
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from solo.data.image_loaders import load_fits_image, load_npy_image, load_grey_image, load_rgb_image

class MiraBest(Dataset):
    def __init__(self, root="../mirabest", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.squeeze(np.load(image_path))
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class RGZ(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../RGZ-D1-smorph-dataset", transform=None, data_dict=None,  datalist="info_wo_nan.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class FRG(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../frg-FirstRadioGalaxies", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.array(Image.open(image_path)).astype(np.float32)
            #img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class VLASS(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../vlass", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.load(image_path)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)
    
class ROBIN(Dataset):
    def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        """
            balancing_strategy could be:
                as_is                   = datalist as loaded
                balanced_downsampled    = each class downsampled to min if min_samples_class is None or to fixed value
                balanced_fixed          = each class down/over-sampled to fixed value
        """
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled
        

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return self.remove_nan(img), self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)
    
    def pad_to_square(self, image_npy):
        H = image_npy.shape[-2]
        W = image_npy.shape[-1]
        if H < W:
            dif = W - H
            if dif % 2 == 0:
                up_pad = bot_pad = int(dif / 2)
            else:
                up_pad = int(dif / 2)
                bot_pad = int(dif - up_pad)
            #p2d = (up_pad, bot_pad, 0, 0)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((up_pad, bot_pad), (0, 0)),
                    mode='constant', constant_values=0.) 
            
        if H > W:
            dif = H - W
            if dif % 2 == 0:
                left_pad = right_pad = int(dif / 2)
            else:
                left_pad = int(dif / 2)
                right_pad = int(dif - left_pad)
            #p2d = (0, 0, left_pad, right_pad)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0.) 
        return image_npy

    def remove_nan(self, im):
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image
        return im

class FitsCustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None, datalist='info.json'):
        self.root = Path(root)
        self.transform = transform
        self.info = self.load_info(datalist)
        self.images = list(self.info["target_path"])
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, _ = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), -1
        else:
            return img, -1
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return self.removeNaNs(img), -1
    
    def add_norm_channels(self, im):
        nbr_bins = 1024
        # obtain the image histogram
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image # set NaN values to min_values
        imhist,bins = np.histogram(im.flatten(),nbr_bins, density=True)
        # derive the cumulative distribution function, CDF
        cdf = imhist.cumsum()      
        # normalise the CDF
        cdf = cdf / cdf[-1]
        
        im2 = np.interp(im.flatten(),bins[:-1],cdf).reshape(im.shape)

        sigma = 3.0
        #current_min = np.min(t)
        #masked_image = sigma_clip(t[t != 0.0], sigma=sigma, maxiters=5) # non si può fare perchè restituisce un immagine con una dimensione diversa
        masked_image, lower_bound, upper_bound = sigma_clip(im, sigma=sigma, maxiters=5, return_bounds=True)
        min_image = np.min(masked_image)
        max_image = np.max(masked_image)

        norm_npy_image = np.zeros(im.shape)
        norm_npy_image = im

        norm_npy_image = (norm_npy_image - min_image) / (max_image - min_image)
        
        norm_npy_image[masked_image.mask & (im < lower_bound)] = 0.0
        norm_npy_image[masked_image.mask & (im > upper_bound)] = 1.0

        # use linear interpolation of CDF to find new pixel values
        min_image = np.min(im)
        max_image = np.max(im)
        im = (im - min_image) /  (max_image - min_image)
        return_image = np.dstack((im, norm_npy_image, im2))
        return_image_pil = Image.fromarray(np.uint8(return_image*255))
        return return_image_pil

    def removeNaNs(self, im):
        # obtain the image histogram
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image # set NaN values to min_values
        
        return im

    def __len__(self):
        return len(self.images)

class CustomDataset(Dataset, ABC):
    def __init__(self, data_path: str, loader_type: str, transform, datalist='info.json'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.info = pd.read_json(os.path.join(self.data_path, datalist),  orient="index")
        self.loaders = {
            'fits': load_fits_image,
            'npy': load_npy_image,
            'grey_image': load_grey_image,
            'rgb_image': load_rgb_image
        }
        if loader_type in self.loaders:
            self.load_image = self.loaders[loader_type]
        else:
            raise ValueError(f"Loader type '{loader_type}' is not supported. Available loaders: {list(self.loaders.keys())}")

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.info)
    
class CustomUnlabeledDataset(CustomDataset):
    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(os.path.join(self.data_path, self.info.iloc[index]["target_path"]))
        return self.transform(img), -1

class CustomLabeledDataset(CustomDataset):
    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(os.path.join(self.data_path, self.info.iloc[index]["target_path"]))
        label = self.info.iloc[index]["source_type"]
        return self.transform(img), label