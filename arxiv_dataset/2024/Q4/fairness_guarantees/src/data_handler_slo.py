import torch
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
from glob import glob
import pandas as pd

dr_disease_mapping = {'not.in.icd.table': 0.,
                    'no.dr.diagnosis': 0.,
                    'mild.npdr': 0.,
                    'moderate.npdr': 0.,
                    'severe.npdr': 1.,
                    'pdr': 1.}

amd_disease_mapping = {'not.in.icd.table': 0.,
                        'no.amd.diagnosis': 0.,
                        'early.dry': 1.,
                        'intermediate.dry': 2.,
                        'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
                        'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
                        'wet.amd.active.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.scar': 3.}

class Harvard_SLO(Dataset):

  def __init__(self, file_path, split, diease_type, resolution):

      self.diease_type = diease_type
      self.resolution = resolution
      df_all = pd.read_csv(file_path + "simple_split.csv")

      # print(file_path + "simple_split.csv")
        
      if self.diease_type == 'dr':
          self.data_path = file_path + '/harvard30k_dr'
          self.disease_mapping = dr_disease_mapping
      elif self.diease_type == 'amd':
          self.data_path = file_path + '/harvard30k_amd'
          self.disease_mapping = amd_disease_mapping
      elif self.diease_type == 'glaucoma':
          self.data_path = file_path + '/harvard30k_glaucoma'

      if split=='train':
          self.df_data = df_all[df_all.use=="training"]
      
      elif split=='val':
          self.df_data = df_all[df_all.use=="validation"]

      elif split=='test':
          self.df_data = df_all[df_all.use=="test"]

      # print(self.df_data)

  def __getitem__(self, index):

      img_data = self.data_path + "/" + self.df_data.iloc[index]['filename']
      # print(img_data)
      data = np.load(img_data)
      slo_fundus = data['slo_fundus']
      # print(slo_fundus.shape)
      # slo_fundus = np.transpose(slo_fundus)
      # slo_fundus = slo_fundus[None,:,:]
      # print(slo_fundus.shape)
      if slo_fundus.shape[1] != self.resolution or slo_fundus.shape[2] != self.resolution:
          slo_fundus = resize(slo_fundus, (self.resolution, self.resolution))
      # print(slo_fundus.shape)
      slo_fundus = slo_fundus[None,:,:]
      # print(slo_fundus.shape)
      slo_fundus = np.repeat(slo_fundus, 3, axis=0)
      # print(slo_fundus.shape)
      img_data = slo_fundus.astype(np.float32)        

      if self.diease_type=="dr":
          label = dr_disease_mapping[data['dr_subtype'].item()]
        
      elif self.diease_type=="amd":
          label = amd_disease_mapping[data['amd_condition'].item()]
          
      elif self.diease_type=="glaucoma":
          label = data['glaucoma']

      # print(label)
      attr_label = []
      attr_race = data['race']
      attr_male = data['male']
      attr_hispanic = data['hispanic']

      attr_label.append(attr_race)
      attr_label.append(attr_male)
      attr_label.append(attr_hispanic)
      
      return img_data, label, attr_label

  def __len__(self):
      return len(self.df_data)