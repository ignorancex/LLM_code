import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import os
from scipy import stats

from preprocess import mean, std
from preprocessing.mod_values import bright_lvl, contr_lvl, sat_lvl, hue_lvl, A, w, texture_h
pd.set_option('display.max_columns', None)

mod_value_dict = dict()
mod_value_dict["contrast"]=contr_lvl
mod_value_dict["saturation"]=sat_lvl
mod_value_dict["hue"]=hue_lvl
mod_value_dict["shape"]=str(A)+"_"+str(w)
mod_value_dict["texture"]=texture_h
mod_value_dict["brightness"]=bright_lvl

print("mod values: ", mod_value_dict)

def weighted_mean(x):
    return pd.Series({'weighted_delta': (x.delta * x.orig_similarity).sum() / x.orig_similarity.sum()})

def forward_get_patch_index(ppnet, x):
    distances = ppnet.prototype_distances(x)
    
    # global min pooling
    min_distances, nearest_patch_indices = F.max_pool2d(-distances,
                                    kernel_size=(distances.size()[2],
                                                distances.size()[3]), return_indices=True)
    min_distances = -min_distances.view(-1, ppnet.num_prototypes) #shape (bs, 2000)
    nearest_patch_indices = nearest_patch_indices.view(ppnet.num_prototypes)
    prototype_similarities = ppnet.distance_2_similarity(min_distances) #shape (1,2000)
    logits = ppnet.last_layer(prototype_similarities) #shape(1,200)
    return logits, min_distances, prototype_similarities, nearest_patch_indices

def forward_particular_patch(ppnet, x, nearest_patch_indices):
    distances = ppnet.prototype_distances(x) #shape (5,2000,7,7)
    patch_distances = distances.view(-1, ppnet.num_prototypes, distances.shape[2]*distances.shape[3])[:, range(distances.shape[1]), nearest_patch_indices] #shape (5,2000)
    prototype_similarities = ppnet.distance_2_similarity(patch_distances)
    logits = ppnet.last_layer(prototype_similarities)
    return logits, patch_distances, prototype_similarities

# Select model to use
load_model_dir = './saved_models/densenet121/003_cub/'  # Model directory to ProtoPNet. 
load_model_name = '30push0.7825.pth'# Model name of trained ProtoPNet


test_dir = './data/CUB_200_2011/dataset/' # Path to dataset
dataset = 'train_crop'
test_dataset = 'test_crop'
# Names of the different kinds of modifications to use
modifications = ['contrast', 'saturation', 'hue', 'shape', 'texture', 'brightness']

# Load model
load_model_path = os.path.join(load_model_dir, load_model_name)

ppnet = torch.load(load_model_path)
if torch.cuda.is_available():
    ppnet = ppnet.cuda()
ppnet.eval()
# Get network properties
img_size = ppnet.img_size  # Image size
prototype_shape = ppnet.prototype_shape # Prototype shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

# Initialize preprocessing function used for prototypes
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   transforms.Normalize(mean=mean, std=std)
])

# Dataframe for storing results
results = pd.DataFrame(columns=['image','prototype', 'modification', 'delta', 'orig_similarity'])

# Prototype indices (assumes 2000 prototypes for CUB-200-2011)
prototypes = range(2000)

max_count = 2
count=0
# Loop through image files in all image folders
for path, subdirs, files in os.walk(os.path.join(test_dir, dataset)):
    for subdir in subdirs:
        count+=1
        # if count > max_count:
        #     break
        print("class: ", subdir, count, "/ 200",flush=True)
        for class_path, class_subdirs, class_files in os.walk(os.path.join(os.path.join(test_dir, dataset),subdir)):
            # Loop through files in folder
            for filename in class_files:
                # print("filename: ", filename)
                img_path = os.path.join(class_path, filename) # Get path of file
                
                mod_tensors = []

                # Open image and convert to RGB
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except:
                    img_path = img_path + '.jpg'
                    img_pil = Image.open(img_path).convert('RGB')
                image_orig = preprocess(img_pil).unsqueeze(0)   # Apply preprocessing function
                if torch.cuda.is_available():
                    image_orig = image_orig.cuda() # Utilize GPU
                # Get network output
                with torch.no_grad():
                   logits, min_distances, orig_similarities, nearest_patch_indices = forward_get_patch_index(ppnet, image_orig)
                
                orig_similarity = orig_similarities[0].cpu().data.numpy() #shape(2000,)
                orig_similarities = orig_similarities[0].repeat(len(modifications),1).cpu().data.numpy() # Shape (5, 2000)
                

                # Load the corresponding modified image and find difference
                # in similarity score for each prototype with respect to a specific
                # modification
                for m in range(len(modifications)):
                    modification = modifications[m]
                    # Modify image path to get the modified image
                    mod_path = img_path.replace(dataset, 
                                                dataset + "_" + modification+"_"+str(mod_value_dict[modification]))
                    # Open image and convert to RGB
                    try:
                        img_pil = Image.open(mod_path).convert('RGB')
                    except:
                        mod_path = mod_path + '.jpg'
                        img_pil = Image.open(mod_path).convert('RGB')
                    
                    img_tensor = preprocess(img_pil)  # Turn image into tensor
                    mod_tensors.append(img_tensor)

                images_mod = torch.stack(mod_tensors) #shape [5, 3, 224, 224]
                if torch.cuda.is_available():
                    images_mod = images_mod.cuda() # Utilize GPU

                # Get network output and convert to similarity scores
                with torch.no_grad():
                    logits, patch_distances, mod_similarities = forward_particular_patch(ppnet, images_mod, nearest_patch_indices)
                mod_similarities = mod_similarities.cpu().data.numpy() # Shape (5, 2000)
                
                delta = orig_similarities - mod_similarities # Get differences (per prototype)
                
                # Make dataframe for results (found difference)
                df = pd.DataFrame(columns=['image', 'prototype', 'modification', 'delta', 'orig_similarity'])
                for row in range(mod_similarities.shape[0]):
                    modification = modifications[row]
                    df['prototype'] = prototypes
                    df['image'] = [filename]*len(prototypes)
                    df['modification'] = modification
                    df['delta'] = delta[row,:]
                    df['orig_similarity'] = orig_similarity
                    # Put row in total results (found differences)
                    results = results.append(df, ignore_index=True)
        
      
    
df_grouped = results.groupby(['prototype', 'modification'])
df_grouped_weighted =  df_grouped.apply(weighted_mean)

# Convert results dataframe to csv format and display
with open(load_model_dir + 'trainingset_weighted_global_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h),str(bright_lvl)), "w") as global_f:
    df_grouped_weighted.reset_index()[['prototype', 'modification', 'weighted_delta']].to_csv(global_f, index=False)
print("Done with global scores. Now saving the local scores...", flush=True)
with open(load_model_dir + 'trainingset_local_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h), str(bright_lvl)), "w") as local_f:
    results.to_csv(local_f, index=False)

print("DONE with training set. Now calculating for the test set...", flush=True)
# Dataframe for storing results
results_testset = pd.DataFrame(columns=['image','prototype', 'modification', 'delta', 'orig_similarity'])
# Prototype indices (assumes 2000 prototypes for CUB-200-2011)
prototypes = range(2000)
max_count = 2
count=0
# Loop through image files in all image folders
for path, subdirs, files in os.walk(os.path.join(test_dir, test_dataset)):
    for subdir in subdirs:
        count+=1
        print("class: ", subdir, count, "/ 200",flush=True)
        for class_path, class_subdirs, class_files in os.walk(os.path.join(os.path.join(test_dir, test_dataset),subdir)):
            # Loop through files in folder
            for filename in class_files:
                img_path = os.path.join(class_path, filename) # Get path of file
                mod_tensors = []
                # Open image and convert to RGB
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except:
                    img_path = img_path + '.jpg'
                    img_pil = Image.open(img_path).convert('RGB')
                image_orig = preprocess(img_pil).unsqueeze(0)   # Apply preprocessing function
                if torch.cuda.is_available():
                    image_orig = image_orig.cuda() # Utilize GPU
                # Get network output
                with torch.no_grad():
                   logits, min_distances, orig_similarities, nearest_patch_indices = forward_get_patch_index(ppnet, image_orig)
                orig_similarity = orig_similarities[0].cpu().data.numpy() #shape(2000,)
                orig_similarities = orig_similarities[0].repeat(len(modifications),1).cpu().data.numpy() # Shape (5, 2000)

                # Load the corresponding modified image and find difference
                # in similarity score for each prototype with respect to a specific
                # modification
                for m in range(len(modifications)):
                    modification = modifications[m]
                    # Modify image path to get the modified image
                    mod_path = img_path.replace(test_dataset, 
                                                test_dataset + "_" + modification+"_"+str(mod_value_dict[modification]))
                    # Open image and convert to RGB
                    try:
                        img_pil = Image.open(mod_path).convert('RGB')
                    except:
                        mod_path = mod_path + '.jpg'
                        img_pil = Image.open(mod_path).convert('RGB')
                    
                    img_tensor = preprocess(img_pil)  # Turn image into tensor
                    mod_tensors.append(img_tensor)

                images_mod = torch.stack(mod_tensors) #shape [5, 3, 224, 224]
                if torch.cuda.is_available():
                    images_mod = images_mod.cuda() # Utilize GPU 
                # Get network output and convert to similarity scores
                with torch.no_grad():
                    # logits, min_distances = ppnet(images_orig_mod)
                    logits, patch_distances, mod_similarities = forward_particular_patch(ppnet, images_mod, nearest_patch_indices)
                
                mod_similarities = mod_similarities.cpu().data.numpy() # Shape (5, 2000)
                delta = orig_similarities - mod_similarities # Get differences (per prototype)
        
                # Make dataframe for results (found difference)
                df = pd.DataFrame(columns=['image', 'prototype', 'modification', 'delta', 'orig_similarity'])
                for row in range(mod_similarities.shape[0]):
                    modification = modifications[row]
                    df['prototype'] = prototypes
                    df['image'] = [filename]*len(prototypes)
                    df['modification'] = modification
                    df['delta'] = delta[row,:]
                    df['orig_similarity'] = orig_similarity
                    # Put row in total results (found differences)
                    results_testset = results_testset.append(df, ignore_index=True)
    
df_grouped_testset = results_testset.groupby(['prototype', 'modification'])
df_grouped_weighted_testset =  df_grouped_testset.apply(weighted_mean)

# Convert results dataframe to csv format and display
with open(load_model_dir + 'testset_weighted_global_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h),str(bright_lvl)), "w") as global_f:
    df_grouped_weighted_testset.reset_index()[['prototype', 'modification', 'weighted_delta']].to_csv(global_f, index=False)
print("Done with global scores. Now saving the local scores...", flush=True)
with open(load_model_dir + 'testset_local_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h), str(bright_lvl)), "w") as local_f:
    results_testset.to_csv(local_f, index=False)

print("DONE!", flush=True)

print(df_grouped_weighted.head())
print("Mean per modification training set: ", df_grouped_weighted.groupby('modification').mean())
print("Mean per modification test set: ", df_grouped_weighted_testset.groupby('modification').mean())
print("std per modification training set: ", df_grouped_weighted.groupby('modification').std())
print("std per modification test set: ", df_grouped_weighted_testset.groupby('modification').std())

scores_train = pd.read_csv(load_model_dir + 'trainingset_weighted_global_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h),str(bright_lvl)))
scores_test = pd.read_csv(load_model_dir + 'testset_weighted_global_prototype_scores_%s_%s_%s_%s_%s_%s.csv'%(str(contr_lvl), str(sat_lvl), str(hue_lvl), str(mod_value_dict["shape"]), str(texture_h),str(bright_lvl)))
for modification in modifications:
    # Each test will return at least two things:
    # Statistic: A quantity calculated by the test that can be interpreted in the context of the test via comparing it to critical values from the distribution of the test statistic.
    # p-value: Used to interpret the test, in this case whether the sample was drawn from a Gaussian distribution.
    # In the SciPy implementation of these tests, you can interpret the p value as follows.
    # p <= alpha: reject H0, not normal.
    # p > alpha: fail to reject H0, normal.

    print("training set", modification, stats.shapiro(scores_train.loc[scores_train['modification'] == modification]['weighted_delta']))
    print("test set", modification, stats.shapiro(scores_test.loc[scores_test['modification'] == modification]['weighted_delta']))
    print("Welch's t-test: ", stats.ttest_ind(scores_train.loc[scores_train['modification'] == modification]['weighted_delta'], scores_test.loc[scores_test['modification'] == modification]['weighted_delta'], axis=0, equal_var=False))
    