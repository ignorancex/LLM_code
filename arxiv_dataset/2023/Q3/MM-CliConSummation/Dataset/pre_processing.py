import numpy as np 
import pandas as pd
from glob import glob 
from tqdm import tqdm
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.layers import Input, Dense, Convolution2D,GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import pickle
import re
import ast


# Functions

def df_with_image_filename(data):
    
    data= data.iloc[:, 0:5]
    data.columns = ['Patient', 'Doctor','Symptom','Intent','Image_information']
    data.dropna(how = 'all', inplace = True)
    data = data.reset_index(drop = True)
    data = data[['Patient','Doctor','Image_information']]
    data = data.fillna(' ')
    
    idx = []
    for i in range(len(data)):
        if data.Patient[i] == 'Dialogue ID':
            idx.append(i)
            
    pd_conversation = []
    dialogue_IDs = []
    for i in tqdm(range(len(idx))):
        try:
            df = data.iloc[idx[i] : idx[i+1], :].reset_index()
            dialogue_IDs.append(df.Doctor[0])
            dialogue = ''
            for i in range(1, len(df) - 1):
                if df.Patient[i] != 'Patient' and df.Patient[i] != 'PATIENT':
                    dialogue = dialogue + 'Patient: ' + df.Patient[i] + ' '
                if df.Doctor[i] != 'Doctor' and df.Doctor[i] != 'DOCTOR':
                    dialogue = dialogue + 'Doctor: ' + df.Doctor[i] + ' '
            dialogue = dialogue + df.Patient[len(df)-1]
            pd_conversation.append(dialogue)

        except IndexError:
            df = data.iloc[idx[i] : len(data), :].reset_index()
            dialogue_IDs.append(df.Doctor[0])
            dialogue = ''
            for i in range(1, len(df) - 1):
                if df.Patient[i] != 'Patient' and df.Patient[i] != 'PATIENT':
                    if df.Image_information[i] != '':
                        df.Patient[i] = re.sub(r'\$(.*?)\$', 'I have ' + df.Image_information[i], df.Patient[i])
                    dialogue = dialogue + 'Patient: ' + df.Patient[i] + ' '
                if df.Doctor[i] != 'Doctor' and df.Doctor[i] != 'DOCTOR':
                    dialogue = dialogue + 'Doctor: ' + df.Doctor[i] + ' '
            dialogue = dialogue + df.Patient[len(df)-1]
            pd_conversation.append(dialogue)
            
        df_pd = pd.DataFrame()
        df_pd['Dialogue_ID'] = dialogue_IDs
        df_pd['Conversation'] = pd_conversation
        
    return df_pd
            
def img_count(row):
    p = re.compile('\$(.*?)\$')
    l = p.findall(row)
    return len(l)

def image_model_dense(preprocessed_input):
    vgg_output = vgg.predict(preprocessed_input)
    flat = Flatten()(vgg_output)
    dense1 = Dense(4096, activation='relu')(flat)
    x = Dropout(0.2)(dense1)
    dense2 = Dense(1072, activation='relu')(x)
    x = Dropout(0.2)(dense2)
    dense_3 = Dense(768, activation='relu')(x)
    output = dense_3.numpy()
    return output

def extract_features(directory, batch_size):
    
    vect_dict = dict()
    filepaths = []


    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(directory,
        target_size=(224, 224),
        batch_size = batch_size, 
        class_mode='categorical')

    for filepath in tqdm(generator.filepaths):
        filename = filepath.split('\\')[-1]
        filepaths.append(filename)
        img = image.load_img(filepath, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        features = image_model_dense(x)
        vect_dict[filename] = features

    return vect_dict, filepaths

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


# Loading parent data file and creating dataset for our task


# Creating text columns
data_1 = pd.read_csv("MultimodalMedicalDialogueDataset - SD [punctuation considered in BIO tag].csv", header = None)
data_2 = pd.read_csv("MultimodalMedicalDialogueDataset - SD2 [punctuation not considered in BIO tag].csv", header = None)

data_1_prep = df_with_image_filename(data_1)
data_2_prep = df_with_image_filename(data_2)

full_data_with_filename = pd.concat([data_1_prep, data_2_prep])
full_data_with_filename.to_csv("Multimodal_summary(with filename).csv")

multimodal_DP = pd.read_csv("Multimodal_summary(with filename).csv")
multimodal_summary = pd.read_csv("Multimodal_summary (filename-img info).csv")

df = pd.DataFrame()
df['Dialogue_ID'] = multimodal_DP['Dialogue_ID']
df['Conversation'] = multimodal_DP['Conversation']
df['Patient_Summary'] = multimodal_summary['Patient_Summary']
df['Doctor_Summary'] = multimodal_summary['Doctor_Summary']
df['Overall_Summary'] = multimodal_summary['Overall_Summary']
df['One_line_patient_summary'] = multimodal_summary['One_line_patient_summary']
df['One_line_doctor_summary'] = multimodal_summary['One_line_doctor_summary']

l = []
for item in list(multimodal_DP['Conversation']):
    l.append(img_count(item))
idx = []
for i, item in enumerate(l):
    if item > 1:
        idx.append(i)
df = df.drop(index = idx).reset_index(drop = True)

p = re.compile('\$(.*?)\$')
filenames = []
for row in list(df['Conversation']):
    l = p.findall(row)
    if len(l) == 0:
        filenames.append('')
    else:
        filenames.append(l[0])
df['Img_filenames'] = filenames
df.to_csv("Multimodal_summary(single_filename).csv")

df = pd.read_csv("Multimodal_summary(single_filename).csv")
df = df.drop('Unnamed: 0', axis = 1)
df = df.fillna('')

# Adding image vectors

train_path = 'image_data/train'
test_path = 'image_data/test'
IMAGE_SIZE = [224, 224]
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

train_dict, train_filename = extract_features('image_data/train', batch_size = 1)
test_dict, test_filename = extract_features('image_data/test', batch_size = 1)

vect_dict = Merge(train_dict, test_dict)

with open('img_feature_dict.pkl', 'wb') as fp:
    pickle.dump(vect_dict, fp)
    print('dictionary saved successfully to file')
    
white_noise = np.random.rand(1,768)
img_features = []
for file in list(df.Img_filenames):
    if file == '':
        img_features.append(np.random.rand(1,768))
    else:
        img_features.append(vect_dict[file])
img_features_new = []
for img in img_features:
    img_features_new.append(img.reshape((768,)))
    
df['Img_features'] = img_features_new
df['Intent'] = [i.split('.')[-1].split(',')[0].split(':')[-1].strip() for i in df.Conversation]
df['Group'] = [int(i.split(".")[-1].split(',')[-1].split(':')[-1].strip()) for i in df.Conversation]

df = df.reset_index(drop = True)
l = []
for i in tqdm(range(len(df.Group))):
    if df.Group[i] == 1:
        l.append(0)
    elif df.Group[i] == 4:
        l.append(1)
    elif df.Group[i] == 5:
        l.append(2)
    elif df.Group[i] == 6:
        l.append(3)
    elif df.Group[i] == 7:
        l.append(4)
    elif df.Group[i] == 12:
        l.append(5)
    elif df.Group[i] == 13:
        l.append(6)
    elif df.Group[i] == 14:
        l.append(7)
    elif df.Group[i] == 19:
        l.append(8)
df['Group_mapped'] = l

json = df.to_json()
with open("Multimodal_summary(with_intent).json", "w") as outfile:
    outfile.write(json)
    
df = pd.read_json("Multimodal_summary(with_intent).json")

# Creating dialogue columns with masked group and disease

C_wo_g = []
for i in range(len(df['Conversation'])): 
    l = df.Conversation[i].split('.')
    l[-1] = l[-1].split(',')[0]
    C_wo_g.append('.'.join(l))
df['Dialogue_wo_group'] = C_wo_g

WO_disease = []
token = '<MASK>'
for i in tqdm(range(len(df['Conversation']))):  
    a = df['Conversation'][i].split('.')[:-1]
    b = a[-1]
    c = b.split(":")
    c[-1] = c[-1].lower().strip()
    intent = df['Intent'][i].lower().strip()
    c[-1] = c[-1].replace(intent, token)
    a[-1] = ':'.join(c)
    WO_disease.append('.'.join(a))
df['Dialogue_wo_disease'] = WO_disease

json = df.to_json()
with open("Multimodal_summary_whole.json", "w") as outfile:
    outfile.write(json)