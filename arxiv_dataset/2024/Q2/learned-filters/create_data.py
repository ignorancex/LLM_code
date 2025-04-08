# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:24:20 2022

@author: matth
"""
import numpy as np
from skimage.transform import radon, iradon

import os
import shutil
import argparse
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description= 'Define parameters for training')

parser.add_argument('--path', help="Path to folder were CT data is stored", type=str, default="DATASET_FOLDER/")
parser.add_argument('--number', help="Total amount of CT images to preprocess (max=1.229)", type=int, default=100) 
parser.add_argument('--noise', help= "Noise type", type=str, default ="gaussian")
args = parser.parse_args()


folder = args.path


number=args.number
noise= args.noise
savepath=args.path+'/preprocessed_'+ noise +'/'

text_file = open("path_to_data.txt", "w")
text_file.write(args.path)
text_file.close()

#delete old data
if os.path.exists(savepath):
    shutil.rmtree(savepath)

os.makedirs(savepath)
os.makedirs(savepath+'train/')
os.makedirs(savepath+'test/')

width=256
height=256
angles=512
alpha=np.array([32,28,24,20,16,12,8,4,0])
delta=np.sqrt(185856)*alpha

#files
#only take non covid patients
ct_list=os.listdir(folder+'non-COVID/')[0:number]

#prepare and save data
#only need 500 scans
count=0
for ct in tqdm(ct_list):
    count+=1
    x=cv2.imread(folder+'non-COVID/'+ct,cv2.IMREAD_GRAYSCALE)
    x=x/255
    x=x-(1/2)
    x=2*x
    x=cv2.resize(x, (256,256))
    data={}
    data['alpha']=alpha
    data['x']=x
    data['x_fbp']=[]
    y=radon(x,np.arange(angles)/(angles/180), circle=False)
    for i in range(len(alpha)):    
        if noise =="gaussian":
            sigma=alpha[i]
            #print(sigma)
            z=sigma*np.random.randn(y.shape[0], angles)
        elif noise =="poisson":
            m=np.min(y)
            y=y-m
            z=np.random.poisson(y)-y
            y=y+m
            scale=delta[i]/np.linalg.norm(z)
            z=z*scale
        elif noise =="uniform":
            a=np.sqrt(3)*alpha[i]
            z= np.random.uniform(low=-a, high=a, size=(y.shape[0], angles))
        elif noise == "saltpepper":
            z=np.zeros_like(y)
            ma=np.max(y)
            mi=np.min(y)
            while np.linalg.norm(z)<delta[i]:
                x_coord=np.random.randint(0, y.shape[0])
                y_coord=np.random.randint(0, angles)
                if np.random.uniform() <0.5:
                    z[x_coord,y_coord]=mi-y[x_coord,y_coord]
                else:
                    z[x_coord,y_coord]=ma-y[x_coord,y_coord]
        else: 
            print("Wrong argument for --noise!")
            
        
        y_delta=y+z
        x_fbp=iradon(y_delta, circle=False)
        data['x_fbp'].append(x_fbp)
        

    savename=ct.split('(')[0][0:-1]+'_'+ct.split('(')[1][0:-5]
    if count < int((4/5)*number):
        np.save(savepath+'train/'+savename+'.npy', data)
    else:
        np.save(savepath+'test/'+savename+'.npy', data)
    
print("Created training and testdataset in ", savepath)    
    
