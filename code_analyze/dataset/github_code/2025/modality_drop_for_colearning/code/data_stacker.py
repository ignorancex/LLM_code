# -*- coding: utf-8 -*-
import numpy as np
import collections
import pandas as pd
from sklearn.model_selection import train_test_split

'''
This code takes modalities (audio, text, vision) and stacks them together into a dataset
'''

#Function for deleting list of labels from dictionary
def delete_key(keys_to_delete, target_dict):
    for key in keys_to_delete:
        target_dict.pop(key, None)
    return target_dict

#Combine all the features together
def combine_modalities(text, vid, audio):
    #dictioanry holding all modalities
    all_mod_dict = dict()
    
    for k in audio.keys():
        #grabbing all features per modalitiy per key
        l = text[k]#[:,:0]
        v = vid[k]#[:, :0]
        a = audio[k]#[:,:0]
        combined_mod = np.concatenate((l,a,v), axis = 1)
        
        #updating all mod dict
        all_mod_dict[k] = combined_mod
    
    return all_mod_dict

if __name__ == "__main__":
    
    #csv that contains ground truth information on dataset
    utterance_to_label = pd.read_csv("../data/iemocap/non_combined/df_iemocap.csv")
    utterance_to_label = utterance_to_label.filter(['wav_file', 'emotion'], axis=1)

    
    #dropping xxx and other and missing files
    xxx_labels = list(utterance_to_label.loc[utterance_to_label['emotion']== 'xxx'].wav_file.values)
    other_labels = list(utterance_to_label.loc[utterance_to_label['emotion']== 'oth'].wav_file.values)
    labels_to_delete = xxx_labels + other_labels
    labels_to_delete.append('Ses03M_impro03_M001')
    labels_to_delete.append('Ses03M_impro03_M030')
    
    #wav to label
    values_to_drop = ['xxx', 'oth']
    utterance_to_label = utterance_to_label[utterance_to_label.emotion.isin(values_to_drop) == False]
    ground_truth_wav = dict(zip(utterance_to_label.wav_file, utterance_to_label.emotion))
    
    #extracted features 
    text_feat = np.load("../data/iemocap/non_combined/text_feat_10036.npy", allow_pickle = True).item()
    visual_feat = np.load("../data/iemocap/non_combined/visual_feat_10036.npy", allow_pickle = True).item()
    audio_feat = np.load("../data/iemocap/non_combined/audio_feat_10036.npy", allow_pickle = True).item()
    
    #Deleting unused keys
    text_feat = delete_key(labels_to_delete, text_feat)
    visual_feat = delete_key(labels_to_delete, visual_feat)
    audio_feat = delete_key(labels_to_delete, audio_feat)
    
    #Convert ground truth label to integers
    all_labels = set(list(ground_truth_wav.values()))
    emotions_to_int = {emotion: i for i, emotion in enumerate(all_labels)}
    
    for k,v in ground_truth_wav.items():
        ground_truth_wav[k] = emotions_to_int[v]
    
    data = combine_modalities(text_feat, visual_feat, audio_feat)
    #splitting into labels and features
    X = np.array(list(data.values()))
    Y = np.array(list(data.keys()))

    
    #Mapping string emotion to int labels
    Y = np.array([ground_truth_wav[idx] for idx in Y]).astype(np.float64)
    
    X_train_im, X_test_im, Y_train_im, Y_test_im = train_test_split(X, Y, test_size = .25)
    X_test_im, X_val_im, Y_test_im, Y_val_im = train_test_split(X_test_im, Y_test_im, test_size = .5)
    
    #saving data 
    np.save("../data/iemocap/X_train", X_train_im)
    np.save("../data/iemocap/X_val", X_val_im)
    np.save("../data/iemocap/X_test", X_test_im)
    
    np.save("../data/iemocap/Y_train", Y_train_im)
    np.save("../data/iemocap/Y_val", Y_val_im)
    np.save("../data/iemocap/Y_test", Y_test_im)
    np.save("../data/iemocap/labels", list(emotions_to_int.keys()))
    
    
    
    