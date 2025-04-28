
import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel 
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.utils import shuffle
import glob

cfg = {
        "smax_l": {"coronal":  { "min": {"mean":151,"std":1.899835519},
                                "max": {"mean":198.5,"std":1.414213562}
                                },

                 "sagittal":  { "min": {"mean":39.5,"std":1.322875656},
                                 "max": {"mean": 75.75,"std":1.785357107}
                                },

                 "axial":      { "min": {"mean":68.875,"std":1.964529206},
                                 "max": {"mean": 113.5,"std":1.802775638}
                                }
                 },

        "smax_r": {"coronal":  { "min": {"mean":151,"std":2.175861898},
                                "max": {"mean":198.375,"std":1.316956719}
                                },

                 "sagittal":  { "min": {"mean":95.25,"std":1.71391365},
                                 "max": {"mean": 128.875,"std":2.315032397}
                                },

                 "axial":      { "min": {"mean":66.375,"std":6.479535091},
                                 "max": {"mean": 111.5,"std":7.465145348}
                                }
                 },

        "sphen": {"coronal":  { "min": {"mean":123.75,"std":7.066647013},
                                "max": {"mean":158.375,"std":4.370036867}
                                },

                 "sagittal":  { "min": {"mean":63.625,"std":3.533323506},
                                 "max": {"mean": 103.875,"std":4.0754601}
                                },

                 "axial":      { "min": {"mean":99.625,"std":2.446298224},
                                 "max": {"mean": 127.625,"std":2.287875652}
                                }
                 },

        "sfront": {"coronal":  { "min": {"mean":185,"std":2.618614683},
                                "max": {"mean":208.2857143,"std":1.829464068}
                                },

                 "sagittal":  { "min": {"mean":54.14285714,"std":8.773801447},
                                 "max": {"mean": 109.4285714,"std":10.18201696}
                                },

                 "axial":      { "min": {"mean":126,"std":4.035556255},
                                 "max": {"mean": 156.8571429,"std":6.685347975}
                                }
                 },


        "seth": {"coronal":  { "min": {"mean":152.5714286,"std":2.258769757},
                                "max": {"mean":197.7142857,"std":4.025429372}
                                },

                 "sagittal":  { "min": {"mean":71.57142857,"std":9.897433186},
                                 "max": {"mean":101.8571429,"std":1.456862718}
                                },

                 "axial":      { "min": {"mean":104.5714286,"std":1.916629695},
                                 "max": {"mean": 129.8571429,"std":3.090472522}
                                }
                 },


        "nose": {"coronal":  { "min": {"mean":147.3333333,"std":4.229525847},
                                "max": {"mean":201.6666667,"std":2.924988129}
                                },

                 "sagittal":  { "min": {"mean":68.5,"std":1.802775638},
                                 "max": {"mean":99.33333333,"std":1.885618083}
                                },

                 "axial":      { "min": {"mean":73.16666667,"std":3.89087251},
                                 "max": {"mean": 123.8333333,"std":2.477678125}
                                }
                 },
        
      }




class ParanasalContrastiveDataModule(pl.LightningDataModule):


    def __init__(self, cfg):
        super().__init__()
        
        
        self.batch_size = cfg["batch_size"]
        self.dimension = cfg["dimension"]
        self.subjects = {"train": [],"val": [],"val_thresh": [],"test":[]}
        self.preprocess = None
        self.augmentations = cfg["augmentations"]
        self.transform = None

        self.trainset = cfg["trainset"]
        self.valset = cfg["valset"]
        self.valset_thresh = cfg["valset_thresh"]
        self.testset = cfg["testset"]

        print("LAPPU TRAIN SET", self.trainset)

        
        self.prepare_data()
       
    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)


    def __get__data__(self,data,y,patient_disease_id):
    
        out = data 
        out = np.expand_dims(out,axis=0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y,patient_disease_id=patient_disease_id)

        return subject


    def __prepare_dataset__(self,df,ds_type):
        
        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)
        for index, row in df.iterrows():
           
            image_path = row["image_path"]
            label      = row["cls_id"]
            assert os.path.isfile(image_path)
            img = nibabel.load(image_path)

            patient_disease_id = image_path.split("/")[-1]

            volume = self.__get__data__(img.get_data(),label,patient_disease_id)
            self.subjects[ds_type].append(volume)


    def read_csv(self,csv_location):

        df = pd.read_csv(csv_location)
        return df
        

    def prepare_data(self):
        
        self.__prepare_dataset__(self.read_csv(self.trainset),"train")
        self.__prepare_dataset__(self.read_csv(self.valset),"val")
        #self.__prepare_dataset__(self.read_csv(self.valset_thresh),"val_thresh")
        self.__prepare_dataset__(self.read_csv(self.testset),"test")
        
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99))),
            tio.Resize((self.dimension, self.dimension,self.dimension)),
            #tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    

    def setup(self, stage=None):

        
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])
        self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform)
        #self.val_thresh   = tio.SubjectsDataset(self.subjects["val_thresh"],transform=self.transform)
        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform)
        
        self.train_aug = None

        if self.augmentations:

            print("AUGMENTATING TRAINING SET")
            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,
                tio.RandomFlip(axes=(0,1,2)): 1.0,
                tio.RandomMotion(): 1.0, 
                tio.RandomBlur(): 1.0, 
                tio.RandomNoise(): 1.0, 
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2))]): 1.0,
               #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomMotion()]): 1.0,
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomBlur()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomNoise()]): 1.0,

                tio.Compose([tio.RandomFlip(axes=(0,1,2)), tio.RandomMotion()]): 1.0,
                #tio.Compose([tio.RandomFlip(axes=(0,1,2)), tio.RandomBlur()]): 1.0,
                tio.Compose([tio.RandomFlip(axes=(0,1,2)), tio.RandomNoise()]): 1.0,

                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomMotion()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomBlur()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomNoise()]): 1.0,

            }




            augmentations = tio.Compose([tio.OneOf(spatial_transforms, p=0.5)])
            self.transform = tio.Compose([tio.Compose(augmentations),self.preprocess])

        else:
            print("NO AUGMENTATION")
        
        self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform)
        

        print(f"#Train Dataset: {len(self.train)} #Val Dataset {len(self.val)} #Test Dataset {len(self.test)}\n")


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #loader = {

        #    "cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
        #    "cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        #return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        #loader = {

            #"cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            #"cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        #return [DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False), DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)]

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

    def val_thresh_eval_dataloader(self):
        return DataLoader(self.val_thresh, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=16, pin_memory=True, shuffle=False)

if __name__ == '__main__':

    p = ParanasalContrastiveDataModule("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset",32, 8, 0.8)

    p.prepare_data()



#Look HERE
def remove_regions_3d_image(image, chunk, total_percentage):
    """
    Remove regions of 3D image randomly.

    Parameters:
        image (ndarray): a 3D numpy array representing the image
        chunk (tuple): a 3-element tuple representing the size of the region to remove
        total_percentage (float): the percentage of the total image to remove

    Returns:
        ndarray: a 3D numpy array representing the modified image
    """

    # Calculate the size of the image in each dimension
    dim_x, dim_y, dim_z = image.shape
    # Calculate the number of voxels to remove
    num_voxels = int(total_percentage / 100 * (dim_x* dim_y* dim_z)/(chunk[0]*chunk[1]*chunk[2]))

    

    # Calculate the number of voxels to remove in each dimension
    num_x = int(chunk[0] / 2)
    num_y = int(chunk[1] / 2)
    num_z = int(chunk[2] / 2)

    # Remove the specified number of voxels randomly
    for i in range(num_voxels):
        x = random.randint(num_x, dim_x - num_x - 1)
        y = random.randint(num_y, dim_y - num_y - 1)
        z = random.randint(num_z, dim_z - num_z - 1)
        image[x-num_x:x+num_x+1, y-num_y:y+num_y+1, z-num_z:z+num_z+1] = 0

    return image



"""

ParanasalAEDataModule: DataModule used to preprocess the maxillary sinus volumes and pass to the neural network
"""

class ParanasalAEDataModule(pl.LightningDataModule):


    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]
        self.dimension = cfg["dimension"]
        self.subjects = {"train": [],"val": [] ,"test":[]}
        self.preprocess = None
        self.label_group = cfg["label_group"] 
        
        self.augmentations = cfg["augmentations"]
        self.crop_size = cfg["crop_size"]
        self.transform = None

        self.trainset = cfg["trainset"]
        self.valset = cfg["valset"]
        self.testset = cfg["testset"]
        self.one_patient_id = cfg["one_patient_id"] 

        self.noisestd = cfg["noise_std"] 
        
        
       
        self.samples_per_patient = cfg["samples_per_patient"]
        #Count labels used for calculating class weights 
        self.cls_weights = []
        self.all_labels =  []
        self.datasetinitialised = False

        #self.prepare_data()
        #self.setup()
       
    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)


    def __get__data__(self,data,y,disease_y,patient_disease_id,image_path,smax=None,crop_size=None):
    
        out = data 
        #out = np.expand_dims(out,axis=0)
        #subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y,patient_disease_id=patient_disease_id,image_path=image_path,smax=smax)
        subject = tio.Subject(one_image=tio.ScalarImage(out),
                              org_image=tio.ScalarImage(out),
                              label=y,disease_label=disease_y,
                              patient_disease_id=patient_disease_id,
                              image_path=image_path,smax=smax,
                              crop_size=crop_size)

        return subject


    def __prepare_dataset__(self,df,ds_type,label_group,crop_size,one_patient_id=None):
        
        
        # This dataframe only contains the folder where extracted volumes of each patient are located 

        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)
       
        dataframe = {"image_path":[],"cls_id":[],"patient_id":[],"patient_diagnosis":[], "smax":[], "cls_id_disease":[], "crop_size":[]}
        #df = df[2500:]
        for index, row in tqdm(df.iterrows(),total=len(df)):
            
            #if index > 10:
            #    continue
            folder_path  = row["folder"]
            disease_label = row["label"] #Labels 0 normal 1 mucosal thickening 2 polyps 3 cysts 4 fully occupied
            #print("before",label)
            label = None
            #Used to group labels in a super label group. eg. mucosal thickening, polyps, cysts, fully occupied can be grouped into anomaly class
            for label_index,label_grp in enumerate(label_group):
                if disease_label in label_grp:
                    label = label_index
            
            
            

            #print("after",label)

            all_volume_locs = list(glob.glob(folder_path + "/*.nii.gz"))

            assert self.samples_per_patient <= len(all_volume_locs)
            
            all_volume_locs = all_volume_locs[:self.samples_per_patient]
            labels_per_vol =  [label]*len(all_volume_locs)
            crop_sizes =  [crop_size]*len(all_volume_locs)
            disease_labels_per_vol =  [disease_label]*len(all_volume_locs)

            #patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            #TODO 
            patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            smax       =  [x.split("/")[-2] for x in all_volume_locs]
            patient_diagnosis =  [x.split("/")[-1] for x in all_volume_locs]
            
            dataframe["image_path"] = dataframe["image_path"] + all_volume_locs
            dataframe["cls_id"] = dataframe["cls_id"] + labels_per_vol
            dataframe["cls_id_disease"] = dataframe["cls_id_disease"] + disease_labels_per_vol
            dataframe["patient_id"] = dataframe["patient_id"] + patient_id
            dataframe["smax"] = dataframe["smax"] + smax
            dataframe["patient_diagnosis"] = dataframe["patient_diagnosis"] + patient_diagnosis
            dataframe["crop_size"] = dataframe["crop_size"] + crop_sizes
        
        df_volumes = pd.DataFrame.from_dict(dataframe) #Dataframe contains the location of all the volumes and their corresponding labels

        for index, row in tqdm(df_volumes.iterrows(),total=len(df_volumes)):
           
            image_path         = row["image_path"]
            label              = row["cls_id"]
            patient_id         = row["patient_id"]
            patient_diagnosis  = row["patient_diagnosis"]
            smax               = row["smax"]
            disease_label      = row["cls_id_disease"] 
            crop_size          = row["crop_size"] 
            
            #Asserting that none of the passed values are None otherwise we will get this error: https://discuss.pytorch.org/t/typeerror-default-collate-batch-must-contain-tensors-numpy-arrays-numbers-dicts-or-lists-found-class-nonetype/87090/5
            assert os.path.isfile(image_path)
            assert label is not None
            assert patient_id is not None
            assert patient_diagnosis is not None
            assert smax is not None
            assert disease_label is not None

            #img = nibabel.load(image_path)

            #patient_disease_id = image_path.split("/")[-1]
            #volume = self.__get__data__(img.get_data(),label,patient_id,image_path,smax)
            volume = self.__get__data__(image_path,label,disease_label,patient_id,image_path,smax,crop_size)
            
            
            #Selects specific patients data from the entire dataset 
            if one_patient_id is not None: 
                if patient_id in one_patient_id:
                    #print("SELECTED ",patient_id,one_patient_id)
                    #print(ds_type, image_path)
                    #assert False
                    self.subjects[ds_type].append(volume)
            else:
                
                self.subjects[ds_type].append(volume)


    def read_csv(self,csv_location):

        df = pd.read_csv(csv_location)
        return df
        

    def prepare_data(self):
        ""

        if not self.datasetinitialised:
            self.__prepare_dataset__(self.read_csv(self.trainset),"train",self.label_group,self.crop_size)
            
            self.__prepare_dataset__(self.read_csv(self.valset),"val",self.label_group,self.crop_size,self.one_patient_id)
            #self.__prepare_dataset__(self.read_csv(self.valset_thresh),"val_thresh")
            self.__prepare_dataset__(self.read_csv(self.testset),"test",self.label_group,self.crop_size,self.one_patient_id)
            self.datasetinitialised = True
        
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1),percentiles=(self.cfg.get('perc_low',1),self.cfg.get('perc_high',99))),
            tio.Resize((64, 64, 64)),
            #tio.Resize((80, 80, 80)),
            #tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def calculate_cls_weights(self):
        
        df = self.read_csv(self.trainset) 
        labels = df["label"]
        all_labels = []
        for disease_label in labels:
            for label_index,label_grp in enumerate(self.label_group):
                if disease_label in label_grp:
                    
                    all_labels.append(label_index)
                    break
            

        
        # Function to calculate class weights 
        unique_count = {}
        for elem in all_labels:
            if elem in unique_count:
                unique_count[elem] += 1
            else:
                unique_count[elem] = 1
        
        #wj=n_samples / (n_classes * n_samplesj)
        n_samples = len(all_labels)
        n_classes = len(list(unique_count.keys()))

        sorted_dict = dict(sorted(unique_count.items()))
        for key in sorted_dict.keys():
            
            n_samplesj = sorted_dict[key]
            assert n_samplesj > 0

            w = n_samples/(n_classes*n_samplesj)

            self.cls_weights.append(w)
        
       
        

    def setup(self, stage=None):

        print("Setting up dataset")
        self.preprocess = self.get_preprocessing_transform()
        
        spatial_transforms = {
            
            tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 0.5,
            tio.RandomFlip(axes=(0,1,2)): 0.5,
            tio.RandomBlur(): 0.5,
            tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2))]): 0.5,
            tio.RandomNoise(mean=0,std=self.noisestd,exclude="org_image"): 1.0, 
            
        }

        augmentations = tio.Compose(spatial_transforms, p=1.0)
        if self.augmentations:
            print("Augmenting Dataset")
            self.transform = tio.Compose([self.preprocess,tio.Compose(augmentations)])
        else:
            print("No Augmentation applied to dataset")
            self.transform = tio.Compose([self.preprocess])

        try:
            self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform,load_getitem=False)
            
        except:
            self.train = [1]

        
        try:
            self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform,load_getitem=False)
        except:
            self.val = [1]

        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform,load_getitem=False)
        
        print(f"#Train Dataset: {len(self.train)} #Val Dataset {len(self.val)} #Test Dataset {len(self.test)}\n")


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #loader = {

        #    "cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
        #    "cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        #return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        #loader = {

            #"cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            #"cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        #return [DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False), DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)]

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, pin_memory=True, shuffle=False)

    def val_thresh_eval_dataloader(self):

        return DataLoader(self.val_thresh, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

"""

ParanasalClassificationDataModule: DataModule used to preprocess the maxillary sinus volumes and pass to the neural network
"""

class ParanasalClassificationDataModule(pl.LightningDataModule):


    def __init__(self, cfg):
        super().__init__()
        
        
        self.batch_size = cfg["batch_size"]
        self.dimension = cfg["dimension"]
        self.subjects = {"train": [],"val": [] ,"test":[]}
        self.preprocess = None
        self.label_group = cfg["label_group"] 
        
        self.augmentations = cfg["augmentations"]
        self.crop_size = cfg["crop_size"]
        self.transform = None

        self.trainset = cfg["trainset"]
        self.valset = cfg["valset"]
        self.testset = cfg["testset"]
        self.one_patient_id = cfg["one_patient_id"] 
        
        
       
        self.samples_per_patient = cfg["samples_per_patient"]
        #Count labels used for calculating class weights 
        self.cls_weights = []
        self.all_labels =  []
        self.datasetinitialised = False

        #self.prepare_data()
        #self.setup()
       
    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)


    def __get__data__(self,data,y,disease_y,patient_disease_id,image_path,smax=None,crop_size=None):
    
        out = data 
        #out = np.expand_dims(out,axis=0)
        #subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y,patient_disease_id=patient_disease_id,image_path=image_path,smax=smax)
        subject = tio.Subject(one_image=tio.ScalarImage(out),label=y,disease_label=disease_y,patient_disease_id=patient_disease_id,image_path=image_path,smax=smax,crop_size=crop_size)

        return subject


    def __prepare_dataset__(self,df,ds_type,label_group,crop_size,one_patient_id=None):
        
        
        # This dataframe only contains the folder where extracted volumes of each patient are located 

        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)
       
        dataframe = {"image_path":[],"cls_id":[],"patient_id":[],"patient_diagnosis":[], "smax":[], "cls_id_disease":[], "crop_size":[]}
        #df = df[2500:]
        for index, row in tqdm(df.iterrows(),total=len(df)):
            
            #if index > 10:
            #    continue
            folder_path  = row["folder"]
            disease_label = row["label"] #Labels 0 normal 1 mucosal thickening 2 polyps 3 cysts 4 fully occupied
            #print("before",label)
            label = None
            #Used to group labels in a super label group. eg. mucosal thickening, polyps, cysts, fully occupied can be grouped into anomaly class
            for label_index,label_grp in enumerate(label_group):
                if disease_label in label_grp:
                    label = label_index
            
            
            
            #print(folder_path)
            #print("after",label)

            all_volume_locs = list(glob.glob(folder_path + "/*.nii.gz"))

            assert self.samples_per_patient <= len(all_volume_locs)
            
            all_volume_locs = all_volume_locs[:self.samples_per_patient]
            labels_per_vol =  [label]*len(all_volume_locs)
            crop_sizes =  [crop_size]*len(all_volume_locs)
            disease_labels_per_vol =  [disease_label]*len(all_volume_locs)

            #patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            #TODO 
            patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            smax       =  [x.split("/")[-2] for x in all_volume_locs]
            patient_diagnosis =  [x.split("/")[-1] for x in all_volume_locs]
            
            dataframe["image_path"] = dataframe["image_path"] + all_volume_locs
            dataframe["cls_id"] = dataframe["cls_id"] + labels_per_vol
            dataframe["cls_id_disease"] = dataframe["cls_id_disease"] + disease_labels_per_vol
            dataframe["patient_id"] = dataframe["patient_id"] + patient_id
            dataframe["smax"] = dataframe["smax"] + smax
            dataframe["patient_diagnosis"] = dataframe["patient_diagnosis"] + patient_diagnosis
            dataframe["crop_size"] = dataframe["crop_size"] + crop_sizes
        
        df_volumes = pd.DataFrame.from_dict(dataframe) #Dataframe contains the location of all the volumes and their corresponding labels

        for index, row in tqdm(df_volumes.iterrows(),total=len(df_volumes)):
           
            image_path         = row["image_path"]
            label              = row["cls_id"]
            patient_id         = row["patient_id"]
            patient_diagnosis  = row["patient_diagnosis"]
            smax               = row["smax"]
            disease_label      = row["cls_id_disease"] 
            crop_size          = row["crop_size"] 
            
            #Asserting that none of the passed values are None otherwise we will get this error: https://discuss.pytorch.org/t/typeerror-default-collate-batch-must-contain-tensors-numpy-arrays-numbers-dicts-or-lists-found-class-nonetype/87090/5
            assert os.path.isfile(image_path)
            assert label is not None
            assert patient_id is not None
            assert patient_diagnosis is not None
            assert smax is not None
            assert disease_label is not None

            #img = nibabel.load(image_path)

            #patient_disease_id = image_path.split("/")[-1]
            #volume = self.__get__data__(img.get_data(),label,patient_id,image_path,smax)
            volume = self.__get__data__(image_path,label,disease_label,patient_id,image_path,smax,crop_size)
            
            
            #Selects specific patients data from the entire dataset 
            if one_patient_id is not None: 
                
                if patient_id in one_patient_id:
                    
                    print("SELECTED ",patient_id,one_patient_id)
                    print(ds_type, image_path)
                    #assert False
                    self.subjects[ds_type].append(volume)
            else:
                
                self.subjects[ds_type].append(volume)


    def read_csv(self,csv_location):

        df = pd.read_csv(csv_location)
        return df
        

    def prepare_data(self):
        ""

        if not self.datasetinitialised:
            self.__prepare_dataset__(self.read_csv(self.trainset),"train",self.label_group,self.crop_size)
            
            self.__prepare_dataset__(self.read_csv(self.valset),"val",self.label_group,self.crop_size,self.one_patient_id)
            #self.__prepare_dataset__(self.read_csv(self.valset_thresh),"val_thresh")
            self.__prepare_dataset__(self.read_csv(self.testset),"test",self.label_group,self.crop_size,self.one_patient_id)
            self.datasetinitialised = True
        
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99))),
            tio.Resize((64, 64, 64)),
            #tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def calculate_cls_weights(self):
        
        df = self.read_csv(self.trainset) 
        labels = df["label"]
        all_labels = []
        for disease_label in labels:
            for label_index,label_grp in enumerate(self.label_group):
                if disease_label in label_grp:
                    
                    all_labels.append(label_index)
                    break
            

        
        # Function to calculate class weights 
        unique_count = {}
        for elem in all_labels:
            if elem in unique_count:
                unique_count[elem] += 1
            else:
                unique_count[elem] = 1
        
        #wj=n_samples / (n_classes * n_samplesj)
        n_samples = len(all_labels)
        n_classes = len(list(unique_count.keys()))

        sorted_dict = dict(sorted(unique_count.items()))
        for key in sorted_dict.keys():
            
            n_samplesj = sorted_dict[key]
            assert n_samplesj > 0

            w = n_samples/(n_classes*n_samplesj)

            self.cls_weights.append(w)
        
       
        

    def setup(self, stage=None):

        print("Setting up dataset")
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])


        try:
            self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform,load_getitem=False)
        except:
            self.val = [1]

        ########
        #spatial_transforms = {
                   
        #    tio.RandomMotion(degrees=40): 1.0, 
                
        #    }

        #augmentations = tio.Compose([tio.OneOf(spatial_transforms, p=1.0)])
        #self.transform = tio.Compose([self.preprocess,tio.Compose(augmentations)])

        ########
        
        #self.val_thresh   = tio.SubjectsDataset(self.subjects["val_thresh"],transform=self.transform)
        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform,load_getitem=False)
        
        self.train_aug = None
        
        if self.augmentations:

            print("AUGMENTATING TRAINING SET")
            """
            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,
                tio.RandomFlip(axes=(0,1,2)): 1.0 

            }
            """

            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,
                tio.RandomFlip(axes=(0,1,2)): 1.0,
                tio.RandomMotion(): 1.0, 
                tio.RandomBlur(): 1.0, 
                tio.RandomNoise(): 1.0, 
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2))]): 1.0,
               #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomMotion()]): 1.0,
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomBlur()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomNoise()]): 1.0,

                tio.Compose([tio.RandomFlip(axes=(0,1,2)), tio.RandomMotion()]): 1.0,
                tio.Compose([tio.RandomFlip(axes=(0,1,2)), tio.RandomNoise()]): 1.0,

                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomMotion()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomBlur()]): 1.0,
                #tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2)),tio.RandomNoise()]): 1.0,

            }

            augmentations = tio.Compose([tio.OneOf(spatial_transforms, p=0.5)])
            self.transform = tio.Compose([self.preprocess,tio.Compose(augmentations)])

        else:
            print("NO AUGMENTATION")
        

        try:
            self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform,load_getitem=False)
            
        except:
            self.train = [1]
        
        

        print(f"#Train Dataset: {len(self.train)} #Val Dataset {len(self.val)} #Test Dataset {len(self.test)}\n")


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #loader = {

        #    "cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
        #    "cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        #return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        #loader = {

            #"cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            #"cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        #return [DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False), DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)]

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):

        return DataLoader(self.val, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def val_thresh_eval_dataloader(self):

        return DataLoader(self.val_thresh, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)
    





class ParanasalClassificationWithResidualDataModule(pl.LightningDataModule):


    def __init__(self, cfg):
        super().__init__()
        
        
        self.batch_size = cfg["batch_size"]
        self.dimension = cfg["dimension"]
        self.subjects = {"train": [],"val": [] ,"test":[]}
        self.preprocess = None
        self.label_group = cfg["label_group"] 
        
        self.augmentations = cfg["augmentations"]
        self.crop_size = cfg["crop_size"]
        self.transform = None

        self.trainset = cfg["trainset"]
        self.valset = cfg["valset"]
        self.testset = cfg["testset"]
        self.one_patient_id = cfg["one_patient_id"] 

        self.ae_model = cfg["ae_model"] 
        self.median_filter  = cfg["median_filter"] 
        
        
       
        self.samples_per_patient = cfg["samples_per_patient"]
        #Count labels used for calculating class weights 
        self.cls_weights = []
        self.all_labels =  []
        self.datasetinitialised = False

        #self.prepare_data()
        #self.setup()
       
    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)


    def __get__data__(self,y,disease_y,patient_disease_id,image_path,image_path_residual,smax=None,crop_size=None):
    
        #out = np.expand_dims(out,axis=0)
        #subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y,patient_disease_id=patient_disease_id,image_path=image_path,smax=smax)

        #Resize torch io ScalarImage to 65x65x65
        #image = tio.ScalarImage(image_path)

        # Define rescaling transform
        #resize_transform = tio.CropOrPad((64, 64, 64)) 
        #rescaled_image = resize_transform(image)

        #print(rescaled_image.shape)


        subject = tio.Subject(one_image= tio.ScalarImage(image_path) , #rescaled_image,
                              one_image_residual= tio.ScalarImage(image_path_residual) ,
                              org_image=tio.ScalarImage(image_path) ,
                              label=y,
                              disease_label=disease_y,
                              patient_disease_id=patient_disease_id,
                              image_path=image_path,
                              image_path_residual=image_path_residual,
                              smax=smax,
                              crop_size=crop_size)

        return subject


    def __prepare_dataset__(self,df,ds_type,label_group,crop_size,one_patient_id=None):
        
        
        # This dataframe only contains the folder where extracted volumes of each patient are located 

        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)
       
        dataframe = {"image_path":[],"image_path_residuals":[],"cls_id":[],"patient_id":[],"patient_diagnosis":[], "smax":[], "cls_id_disease":[], "crop_size":[]}
        #df = df[2500:]
        for index, row in tqdm(df.iterrows(),total=len(df)):
            
            #if index > 10:
            #    continue
            folder_path  = row["folder"]
            disease_label = row["label"] #Labels 0 normal 1 mucosal thickening 2 polyps 3 cysts 4 fully occupied
            #print("before",label)
            label = None
            #Used to group labels in a super label group. eg. mucosal thickening, polyps, cysts, fully occupied can be grouped into anomaly class
            for label_index,label_grp in enumerate(label_group):
                if disease_label in label_grp:
                    label = label_index
            
            
            

            #print("after",label)

            all_volume_locs = list(glob.glob(folder_path + "/*.nii.gz"))

            all_residual_volume_locs = list(glob.glob(folder_path + "/residuals" + f"/{self.ae_model}/{self.median_filter}/" +  "/*.nii.gz"))
            
            
            assert self.samples_per_patient <= len(all_volume_locs)
            
            all_volume_locs = all_volume_locs[:self.samples_per_patient]
            all_residual_volume_locs = all_residual_volume_locs[:self.samples_per_patient]
            labels_per_vol =  [label]*len(all_volume_locs)
            crop_sizes =  [crop_size]*len(all_volume_locs)
            disease_labels_per_vol =  [disease_label]*len(all_volume_locs)

            #patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            #TODO 
            patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            smax       =  [x.split("/")[-2] for x in all_volume_locs]
            patient_diagnosis =  [x.split("/")[-1] for x in all_volume_locs]
            
            dataframe["image_path"] = dataframe["image_path"] + all_volume_locs
            dataframe["image_path_residuals"] = dataframe["image_path_residuals"] + all_residual_volume_locs
            dataframe["cls_id"] = dataframe["cls_id"] + labels_per_vol
            dataframe["cls_id_disease"] = dataframe["cls_id_disease"] + disease_labels_per_vol
            dataframe["patient_id"] = dataframe["patient_id"] + patient_id
            dataframe["smax"] = dataframe["smax"] + smax
            dataframe["patient_diagnosis"] = dataframe["patient_diagnosis"] + patient_diagnosis
            dataframe["crop_size"] = dataframe["crop_size"] + crop_sizes
        
        df_volumes = pd.DataFrame.from_dict(dataframe) #Dataframe contains the location of all the volumes and their corresponding labels

        for index, row in tqdm(df_volumes.iterrows(),total=len(df_volumes)):
           
            image_path           = row["image_path"]
            image_path_residuals = row["image_path_residuals"]
            label                = row["cls_id"]
            patient_id           = row["patient_id"]
            patient_diagnosis    = row["patient_diagnosis"]
            smax                 = row["smax"]
            disease_label        = row["cls_id_disease"] 
            crop_size            = row["crop_size"] 
            
            #Asserting that none of the passed values are None otherwise we will get this error: https://discuss.pytorch.org/t/typeerror-default-collate-batch-must-contain-tensors-numpy-arrays-numbers-dicts-or-lists-found-class-nonetype/87090/5
            assert os.path.isfile(image_path)
            assert label is not None
            assert patient_id is not None
            assert patient_diagnosis is not None
            assert smax is not None
            assert disease_label is not None

            #img = nibabel.load(image_path)

            #patient_disease_id = image_path.split("/")[-1]
            #volume = self.__get__data__(img.get_data(),label,patient_id,image_path,smax)
            volume = self.__get__data__(label,disease_label,patient_id,image_path,image_path_residuals,smax,crop_size)
            
            
            #Selects specific patients data from the entire dataset 
            if one_patient_id is not None: 
                if patient_id in one_patient_id:
                    #print("SELECTED ",patient_id,one_patient_id)
                    #print(ds_type, image_path)
                    #assert False
                    self.subjects[ds_type].append(volume)
            else:
                
                self.subjects[ds_type].append(volume)


    def read_csv(self,csv_location):
        df = pd.read_csv(csv_location)
        return df
        

    def prepare_data(self):
        ""

        if not self.datasetinitialised:
            self.__prepare_dataset__(self.read_csv(self.trainset),"train",self.label_group,self.crop_size)
            
            self.__prepare_dataset__(self.read_csv(self.valset),"val",self.label_group,self.crop_size,self.one_patient_id)
            #self.__prepare_dataset__(self.read_csv(self.valset_thresh),"val_thresh")
            self.__prepare_dataset__(self.read_csv(self.testset),"test",self.label_group,self.crop_size,self.one_patient_id)
            self.datasetinitialised = True
        
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99))),
            tio.Resize((64, 64, 64)),
            #tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def calculate_cls_weights(self):
        
        df = self.read_csv(self.trainset) 
        labels = df["label"]
        all_labels = []
        for disease_label in labels:
            for label_index,label_grp in enumerate(self.label_group):
                if disease_label in label_grp:
                    
                    all_labels.append(label_index)
                    break
            

        
        # Function to calculate class weights 
        unique_count = {}
        for elem in all_labels:
            if elem in unique_count:
                unique_count[elem] += 1
            else:
                unique_count[elem] = 1
        
        #wj=n_samples / (n_classes * n_samplesj)
        n_samples = len(all_labels)
        n_classes = len(list(unique_count.keys()))

        sorted_dict = dict(sorted(unique_count.items()))
        for key in sorted_dict.keys():
            
            n_samplesj = sorted_dict[key]
            assert n_samplesj > 0

            w = n_samples/(n_classes*n_samplesj)

            self.cls_weights.append(w)
        
       
        

    def setup(self, stage=None):

        print("Setting up dataset")
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])


        try:
            self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform,load_getitem=False)
        except:
            self.val = [1]

       
        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform,load_getitem=False)
        
        self.train_aug = None
        
        if self.augmentations:

            print("AUGMENTATING TRAINING SET")
            """
            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,s
                tio.RandomFlip(axes=(0,1,2)): 1.0 

            }
            """

            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,
                tio.RandomFlip(axes=(0,1,2)): 1.0,
                tio.RandomNoise(mean=0,std=0.25): 1.0,
                tio.RandomBlur(): 1.0,
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2))]): 1.0,

            }

            augmentations = tio.Compose([tio.OneOf(spatial_transforms, p=0.5)])
            self.transform = tio.Compose([self.preprocess,tio.Compose(augmentations)])

        else:
            print("NO AUGMENTATION")
        

        try:
            self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform,load_getitem=False)
            
        except:
            self.train = [1]
        
        

        print(f"#Train Dataset: {len(self.train)} #Val Dataset {len(self.val)} #Test Dataset {len(self.test)}\n")


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #loader = {

        #    "cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
        #    "cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        #return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)
        #loader = {

            #"cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            #"cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        #return [DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False), DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)]

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):

        return DataLoader(self.val, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def val_thresh_eval_dataloader(self):

        return DataLoader(self.val_thresh, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):

        return DataLoader(self.test, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)



#Constrastive Loss Data module


class ParanasalSimCLR(pl.LightningDataModule):


    def __init__(self, cfg):
        super().__init__()
        
        
        self.batch_size = cfg["batch_size"]
        self.dimension = cfg["dimension"]
        self.subjects = {"train": [],"val": [] ,"test":[]}
        self.preprocess = None
        self.label_group = cfg["label_group"] 
        
        self.augmentations = cfg["augmentations"]
        self.crop_size = cfg["crop_size"]
        self.transform = None

        self.trainset = cfg["trainset"]
        self.valset = cfg["valset"]
        self.testset = cfg["testset"]
        self.one_patient_id = cfg["one_patient_id"] 

        self.ae_model = cfg["ae_model"] 
        self.median_filter  = cfg["median_filter"] 
        
        
       
        self.samples_per_patient = cfg["samples_per_patient"]
        #Count labels used for calculating class weights 
        self.cls_weights = []
        self.all_labels =  []
        self.datasetinitialised = False

        #self.prepare_data()
        #self.setup()
       
    
    def get_max_shape(self, subjects):
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)


    def __get__data__(self,y,disease_y,patient_disease_id,image_path,image_path_residual,smax=None,crop_size=None):
    
        #out = np.expand_dims(out,axis=0)
        #subject = tio.Subject(image=tio.ScalarImage(tensor=torch.tensor(out)),label=y,patient_disease_id=patient_disease_id,image_path=image_path,smax=smax)
        subject = tio.Subject(one_image_view_one=tio.ScalarImage(image_path),
                              one_image_view_two=tio.ScalarImage(image_path) ,
                              label=y,
                              disease_label=disease_y,
                              patient_disease_id=patient_disease_id,
                              image_path=image_path,
                              image_path_residual=image_path_residual,
                              smax=smax,
                              crop_size=crop_size)

        return subject


    def __prepare_dataset__(self,df,ds_type,label_group,crop_size,one_patient_id=None):
        
        
        # This dataframe only contains the folder where extracted volumes of each patient are located 

        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)
       
        dataframe = {"image_path_view_one":[],"image_path_view_two":[],"cls_id":[],"patient_id":[],"patient_diagnosis":[], "smax":[], "cls_id_disease":[], "crop_size":[]}
        #df = df[2500:]
        for index, row in tqdm(df.iterrows(),total=len(df)):
            
            #if index > 10:
            #    continue
            folder_path  = row["folder"]
            disease_label = row["label"] #Labels 0 normal 1 mucosal thickening 2 polyps 3 cysts 4 fully occupied
            #print("before",label)
            label = None
            #Used to group labels in a super label group. eg. mucosal thickening, polyps, cysts, fully occupied can be grouped into anomaly class
            for label_index,label_grp in enumerate(label_group):
                if disease_label in label_grp:
                    label = label_index
            
            
            

            #print("after",label)

            all_volume_locs = list(glob.glob(folder_path + "/*.nii.gz"))
            
            assert self.samples_per_patient <= len(all_volume_locs)
            
            all_volume_locs = all_volume_locs[:self.samples_per_patient]
            
            labels_per_vol =  [label]*len(all_volume_locs)
            crop_sizes =  [crop_size]*len(all_volume_locs)
            disease_labels_per_vol =  [disease_label]*len(all_volume_locs)

            #patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            #TODO 
            patient_id =  [x.split("/")[-3] for x in all_volume_locs]
            smax       =  [x.split("/")[-2] for x in all_volume_locs]
            patient_diagnosis =  [x.split("/")[-1] for x in all_volume_locs]
            
            dataframe["image_path_view_one"] = dataframe["image_path_view_one"] + all_volume_locs
            dataframe["image_path_view_two"] = dataframe["image_path_view_two"] + all_volume_locs
            dataframe["cls_id"] = dataframe["cls_id"] + labels_per_vol
            dataframe["cls_id_disease"] = dataframe["cls_id_disease"] + disease_labels_per_vol
            dataframe["patient_id"] = dataframe["patient_id"] + patient_id
            dataframe["smax"] = dataframe["smax"] + smax
            dataframe["patient_diagnosis"] = dataframe["patient_diagnosis"] + patient_diagnosis
            dataframe["crop_size"] = dataframe["crop_size"] + crop_sizes
        
        df_volumes = pd.DataFrame.from_dict(dataframe) #Dataframe contains the location of all the volumes and their corresponding labels

        for index, row in tqdm(df_volumes.iterrows(),total=len(df_volumes)):
           
            image_path           = row["image_path_view_one"]
            image_path_residuals = row["image_path_view_two"]
            label                = row["cls_id"]
            patient_id           = row["patient_id"]
            patient_diagnosis    = row["patient_diagnosis"]
            smax                 = row["smax"]
            disease_label        = row["cls_id_disease"] 
            crop_size            = row["crop_size"] 
            
            #Asserting that none of the passed values are None otherwise we will get this error: https://discuss.pytorch.org/t/typeerror-default-collate-batch-must-contain-tensors-numpy-arrays-numbers-dicts-or-lists-found-class-nonetype/87090/5
            assert os.path.isfile(image_path)
            assert label is not None
            assert patient_id is not None
            assert patient_diagnosis is not None
            assert smax is not None
            assert disease_label is not None

            #img = nibabel.load(image_path)

            #patient_disease_id = image_path.split("/")[-1]
            #volume = self.__get__data__(img.get_data(),label,patient_id,image_path,smax)
            volume = self.__get__data__(label,disease_label,patient_id,image_path,image_path_residuals,smax,crop_size)
            
            
            #Selects specific patients data from the entire dataset 
            if one_patient_id is not None: 
                if patient_id in one_patient_id:
                    print("SELECTED ",patient_id,one_patient_id)
                    print(ds_type, image_path)
                    #assert False
                    self.subjects[ds_type].append(volume)
            else:
                
                self.subjects[ds_type].append(volume)


    def read_csv(self,csv_location):
        df = pd.read_csv(csv_location)
        return df
        

    def prepare_data(self):
        ""

        if not self.datasetinitialised:
            self.__prepare_dataset__(self.read_csv(self.trainset),"train",self.label_group,self.crop_size)
            
            self.__prepare_dataset__(self.read_csv(self.valset),"val",self.label_group,self.crop_size,self.one_patient_id)
            #self.__prepare_dataset__(self.read_csv(self.valset_thresh),"val_thresh")
            self.__prepare_dataset__(self.read_csv(self.testset),"test",self.label_group,self.crop_size,self.one_patient_id)
            self.datasetinitialised = True
        
    
    def get_preprocessing_transform(self):

        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99))),
            tio.Resize((64, 64, 64)),
            #tio.EnsureShapeMultiple(8),
        ])
        return preprocess
    
    def calculate_cls_weights(self):
        
        df = self.read_csv(self.trainset) 
        labels = df["label"]
        all_labels = []
        for disease_label in labels:
            for label_index,label_grp in enumerate(self.label_group):
                if disease_label in label_grp:
                    
                    all_labels.append(label_index)
                    break
            

        
        # Function to calculate class weights 
        unique_count = {}
        for elem in all_labels:
            if elem in unique_count:
                unique_count[elem] += 1
            else:
                unique_count[elem] = 1
        
        #wj=n_samples / (n_classes * n_samplesj)
        n_samples = len(all_labels)
        n_classes = len(list(unique_count.keys()))

        sorted_dict = dict(sorted(unique_count.items()))
        for key in sorted_dict.keys():
            
            n_samplesj = sorted_dict[key]
            assert n_samplesj > 0

            w = n_samples/(n_classes*n_samplesj)

            self.cls_weights.append(w)
        
        

    def setup(self, stage=None):

        print("Setting up dataset")
        self.preprocess = self.get_preprocessing_transform()
        self.transform = tio.Compose([self.preprocess])


        

       
        self.test  = tio.SubjectsDataset(self.subjects["test"],  transform=self.transform,load_getitem=False,simclr=True)
        
        self.train_aug = None
        
        if self.augmentations:

            print("AUGMENTATING TRAINING SET")
            """
            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,s
                tio.RandomFlip(axes=(0,1,2)): 1.0 

            }
            """

            spatial_transforms = {
                
                tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10): 1.0,
                tio.RandomFlip(axes=(0,1,2)): 1.0,
                tio.RandomNoise(mean=0,std=0.25): 1.0,
                tio.RandomBlur(): 1.0,
                tio.Compose([tio.RandomAffine(scales=(0.8, 1.2),degrees=10,translation=10), tio.RandomFlip(axes=(0,1,2))]): 1.0,

            }

            augmentations = tio.Compose([tio.OneOf(spatial_transforms, p=0.5)])
            self.transform = tio.Compose([self.preprocess,tio.Compose(augmentations)])

        else:
            print("NO AUGMENTATION")
        

        try:
            self.train = tio.SubjectsDataset(self.subjects["train"], transform=self.transform,load_getitem=False,simclr=True)
            
        except:
            self.train = [1]

        try:
            self.val   = tio.SubjectsDataset(self.subjects["val"],   transform=self.transform,load_getitem=False,simclr=True)
        except:
            self.val = [1]
        
        

        print(f"#Train Dataset: {len(self.train)} #Val Dataset {len(self.val)} #Test Dataset {len(self.test)}\n")


    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        #loader = {

        #    "cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
        #    "cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True,drop_last=True)
        #return loader
        
    def val_dataloader(self):

        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False,drop_last=True)
        #loader = {

            #"cls_1":DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True),
            #"cls_2":DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=True)
        
        #}
        #return [DataLoader(self.train_cls_1, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False), DataLoader(self.train_cls_2, batch_size=self.batch_size//2, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False)]

    def test_dataloader(self):
        
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=16, worker_init_fn=self.worker_init_fn, pin_memory=True, shuffle=False,drop_last=True)

    def val_eval_dataloader(self):

        return DataLoader(self.val, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def val_thresh_eval_dataloader(self):

        return DataLoader(self.val_thresh, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test, batch_size=self.samples_per_patient, num_workers=16, pin_memory=True, shuffle=False)








class ParanasalInferenceDataModule(ParanasalClassificationDataModule):

    def __init__(self, cfg):
        super().__init__(cfg)

    def prepare_data(self):
        
        self.__prepare_dataset__(self.read_csv(self.testset),"test")
        self.setup()


if __name__ == '__main__':

    p = ParanasalContrastiveDataModule("/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset",32, 8, 0.8)

    p.prepare_data()