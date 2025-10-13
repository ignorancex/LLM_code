import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121,VGG16
from sklearn.preprocessing import OneHotEncoder
import random

def data_simulate_noniid(settings):
    path_1='/home/yuyingduan/PycharmProjects/post_FFL_skin_cancer_detection/HM10000/hmnist_28_28_RGB.csv'
    path_2='/home/yuyingduan/PycharmProjects/post_FFL_skin_cancer_detection/HM10000/HAM10000_metadata.csv'

    df=pd.read_csv(path_1)
    sf=pd.read_csv(path_2)
    # classes = {
    #     0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),

    #     1: ('bcc', 'basal cell carcinoma'),

    #     2: ('bkl', 'benign keratosis-like lesions'),

    #     3: ('df', 'dermatofibroma'),

    #     4: ('nv', ' melanocytic nevi'),

    #     5: ('vasc', ' pyogenic granulomas and hemorrhage'),

    #     6: ('mel', 'melanoma'),
    # }
    classes = {
    0: [('akiec', 'actinic keratoses and intraepithelial carcinomae'),
        ('bcc', 'basal cell carcinoma'),
        ('mel', 'melanoma')],
    1: [('bkl', 'benign keratosis-like lesions'),
        ('df', 'dermatofibroma')],
    2: [('nv', 'melanocytic nevi')],
    3: [('vasc', 'pyogenic granulomas and hemorrhage')]
    }
    
    class_mapping = {
    0: 0,  # 'akiec' -> class 0 (Pre-cancerous and cancerous lesions)
    1: 0,  # 'bcc' -> class 0 (Pre-cancerous and cancerous lesions)
    6: 0,  # 'mel' -> class 0 (Pre-cancerous and cancerous lesions)
    2: 1,  # 'bkl' -> class 1 (Benign lesions)
    3: 1,  # 'df' -> class 1 (Benign lesions)
    4: 2,  # 'nv' -> class 2 (Nevus-like lesions)
    5: 3   # 'vasc' -> class 3 (Vascular lesions)
    }


    Y=df['label']
    features=df.drop(columns=['label'])
    S=sf['sex']
    C=sf['age']
    features=np.array(features)
    sensitive_attribute=[]
    for i in S:
        if i=='male':
            sensitive_attribute.append(1)
        else:
            sensitive_attribute.append(0)
    sensitive_attributes=np.array(sensitive_attribute)
    label=[]
    for i in Y:
        if i==0 or i==1 or i==6:
            label.append(0)
        elif i==2 or i==3:
            label.append(1)
        elif i==4:
            label.append(2)
        else:
            label.append(3)
    labels=np.array(label)
#     x_1=0
#     x_2=0
#     x_3=0
#     x_4=0
#     x_5=0
#     x_6=0
#     x_7=0
#     x_8=0
#     for i in range(len(label)):
#         if label[i]==0 and sensitive_attribute[i]==1:
#             x_1+=1
#         elif label[i]==1 and sensitive_attribute[i]==1:
#             x_2+=1
#         elif label[i]==2 and  sensitive_attribute[i]==1:
#             x_3+=1
#         elif label[i]==3 and sensitive_attribute[i]==1:
#             x_4+=1
#         elif label[i]==0 and sensitive_attribute[i]==0:
#             x_5+=1
#         elif label[i]==1 and sensitive_attribute[i]==0:
#             x_6+=1
#         elif label[i]==2 and sensitive_attribute[i]==0:
#             x_7+=1
#         elif label[i]==3 and sensitive_attribute[i]==0:
#             x_8+=1
#     print(x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8)
    def split_and_return(features, label, sensitive_attribute):
            train_x, vali_x, train_y, vali_y, train_s, vali_s = train_test_split(features, label, sensitive_attribute, test_size=0.5, random_state=42)
            return train_x, vali_x, train_y, vali_y, train_s, vali_s

    # Step 1: IID distribution
    if settings == 1:
            data = np.concatenate((features, labels.reshape(-1, 1), sensitive_attributes.reshape(-1, 1)),axis=1)
            np.random.shuffle(data)

            # Split data into 5 clients
            clients_data = np.array_split(data, 5)

            # Prepare the return values for 5 clients
            train_x, vali_x, train_y, vali_y, train_s, vali_s = [], [], [], [], [], []
            for client_data in clients_data:
                client_features = client_data[:, :-2]
                client_label = client_data[:, -2]
                client_sensitive = client_data[:, -1]
                tx, vx, ty, vy, ts, vs = split_and_return(client_features, client_label, client_sensitive)
                train_x.append(tx)
                vali_x.append(vx)
                train_y.append(ty)
                vali_y.append(vy)
                train_s.append(ts)
                vali_s.append(vs)
            train_x_1=np.array(train_x[0])
            vali_x_1=np.array(vali_x[0])
            train_y_1=np.array(train_y[0])
            vali_y_1=np.array(vali_y[0])
            train_s_1=np.array(train_s[0])
            vali_s_1=np.array(vali_s[0])

            train_x_2=np.array(train_x[1])
            vali_x_2=np.array(vali_x[1])
            train_y_2=np.array(train_y[1])
            vali_y_2=np.array(vali_y[1])
            train_s_2=np.array(train_s[1])
            vali_s_2=np.array(vali_s[1])

            train_x_3=np.array(train_x[2])
            vali_x_3=np.array(vali_x[2])
            train_y_3=np.array(train_y[2])
            vali_y_3=np.array(vali_y[2])
            train_s_3=np.array(train_s[2])
            vali_s_3=np.array(vali_s[2])
            
            train_x_4=np.array(train_x[3])
            vali_x_4=np.array(vali_x[3])
            train_y_4=np.array(train_y[3])
            vali_y_4=np.array(vali_y[3])
            train_s_4=np.array(train_s[3])
            vali_s_4=np.array(vali_s[3])

            train_x_5=np.array(train_x[4])
            vali_x_5=np.array(vali_x[4])
            train_y_5=np.array(train_y[4])
            vali_y_5=np.array(vali_y[4])
            train_s_5=np.array(train_s[4])
            vali_s_5=np.array(vali_s[4])

            train_x=np.concatenate((train_x_1,train_x_2,train_x_3,train_x_4,train_x_5),axis=0)
            vali_x=np.concatenate((vali_x_1,vali_x_2,vali_x_3,vali_x_4,vali_x_5),axis=0)
            train_y=np.concatenate((train_y_1,train_y_2,train_y_3,train_y_4,train_y_5),axis=0)
            vali_y=np.concatenate((vali_y_1,vali_y_2,vali_y_3,vali_y_4,vali_y_5),axis=0)
            train_s=np.concatenate((train_s_1,train_s_2,train_s_3,train_s_4,train_s_5),axis=0)
            vali_s=np.concatenate((vali_s_1,vali_s_2,vali_s_3,vali_s_4,vali_s_5),axis=0)
     
    elif settings == 2:
        client_proportions = [0.1, 0.3, 0.5, 0.7, 0.9]
        print(client_proportions)  # Proportions of male data for each client
        data = np.concatenate((features, labels.reshape(-1, 1), sensitive_attributes.reshape(-1, 1)), axis=1)

        # Shuffle the combined data
        np.random.shuffle(data)

        # Split the shuffled data back into features, labels, and sensitive attributes
        features = data[:, :-2]  # All columns except the last two are features
        labels = data[:, -2]     # Second to last column is labels
        sensitive_attributes = data[:, -1] # Last column is sensitive attributes

        male_x, male_y, male_s = [], [], []
        female_x, female_y, female_s = [], [], []

        # Separate male and female data
        for i in range(len(labels)):
            if sensitive_attributes[i] == 1:
                male_x.append(features[i])
                male_y.append(labels[i])
                male_s.append(sensitive_attributes[i])
            else:
                female_x.append(features[i])
                female_y.append(labels[i])
                female_s.append(sensitive_attributes[i])

        male_x, male_y, male_s = np.array(male_x), np.array(male_y), np.array(male_s)
        female_x, female_y, female_s = np.array(female_x), np.array(female_y), np.array(female_s)

        train_x, vali_x, train_y, vali_y, train_s, vali_s = [], [], [], [], [], []

        # Keep track of the used indices
        used_male_indices = set()
        used_female_indices = set()

        for proportion in client_proportions:
            num_male = int(1500 * proportion)
            num_female = int(1500 * (1 - proportion))

            # Get available male and female indices by removing the already used ones
            available_male_indices = np.setdiff1d(np.arange(len(male_x)), list(used_male_indices))
            available_female_indices = np.setdiff1d(np.arange(len(female_x)), list(used_female_indices))

            # Select male and female samples for the client
            male_sample_idx = np.random.choice(available_male_indices, num_male, replace=False)
            female_sample_idx = np.random.choice(available_female_indices, num_female, replace=False)

            # Add the selected indices to the used sets
            used_male_indices.update(male_sample_idx)
            used_female_indices.update(female_sample_idx)

            # Combine the selected samples
            client_x = np.vstack((male_x[male_sample_idx], female_x[female_sample_idx]))
            client_y = np.hstack((male_y[male_sample_idx], female_y[female_sample_idx]))
            client_s = np.hstack((male_s[male_sample_idx], female_s[female_sample_idx]))

            # Shuffle the client data
            combined_client_data = np.hstack((client_x, client_y.reshape(-1, 1), client_s.reshape(-1, 1)))
            np.random.shuffle(combined_client_data)

            shuffled_x = combined_client_data[:, :-2]
            shuffled_y = combined_client_data[:, -2]
            shuffled_s = combined_client_data[:, -1]

            tx, vx, ty, vy, ts, vs = split_and_return(shuffled_x, shuffled_y, shuffled_s)
            train_x.append(tx)
            vali_x.append(vx)
            train_y.append(ty)
            vali_y.append(vy)
            train_s.append(ts)
            vali_s.append(vs)



        train_x_1=np.array(train_x[0])
        vali_x_1=np.array(vali_x[0])
        train_y_1=np.array(train_y[0])
        vali_y_1=np.array(vali_y[0])
        train_s_1=np.array(train_s[0])
        vali_s_1=np.array(vali_s[0])

        train_x_2=np.array(train_x[1])
        vali_x_2=np.array(vali_x[1])
        train_y_2=np.array(train_y[1])
        vali_y_2=np.array(vali_y[1])
        train_s_2=np.array(train_s[1])
        vali_s_2=np.array(vali_s[1])

        train_x_3=np.array(train_x[2])
        vali_x_3=np.array(vali_x[2])
        train_y_3=np.array(train_y[2])
        vali_y_3=np.array(vali_y[2])
        train_s_3=np.array(train_s[2])
        vali_s_3=np.array(vali_s[2])
        
        train_x_4=np.array(train_x[3])
        vali_x_4=np.array(vali_x[3])
        train_y_4=np.array(train_y[3])
        vali_y_4=np.array(vali_y[3])
        train_s_4=np.array(train_s[3])
        vali_s_4=np.array(vali_s[3])

        train_x_5=np.array(train_x[4])
        vali_x_5=np.array(vali_x[4])
        train_y_5=np.array(train_y[4])
        vali_y_5=np.array(vali_y[4])
        train_s_5=np.array(train_s[4])
        vali_s_5=np.array(vali_s[4])

        train_x=np.concatenate((train_x_1,train_x_2,train_x_3,train_x_4,train_x_5),axis=0)
        vali_x=np.concatenate((vali_x_1,vali_x_2,vali_x_3,vali_x_4,vali_x_5),axis=0)
        train_y=np.concatenate((train_y_1,train_y_2,train_y_3,train_y_4,train_y_5),axis=0)
        vali_y=np.concatenate((vali_y_1,vali_y_2,vali_y_3,vali_y_4,vali_y_5),axis=0)
        train_s=np.concatenate((train_s_1,train_s_2,train_s_3,train_s_4,train_s_5),axis=0)
        vali_s=np.concatenate((vali_s_1,vali_s_2,vali_s_3,vali_s_4,vali_s_5),axis=0)
    # elif settings == 3:
    #     client_1_data=[65,5,5,5,5,5,5,5]
    #     client_2_data=[5,65,5,5,5,5,5,5]
    #     client_3_data=[5,5,65,5,5,5,5,5]
    #     client_4_data=[5,5,5,5,65,5,5,5]
    #     client_5_data=[5,5,5,5,5,5,65,5]
    #     data = np.concatenate((features, labels.reshape(-1, 1), sensitive_attributes.reshape(-1, 1)), axis=1)

    #     # Shuffle the combined data
    #     np.random.shuffle(data)

    #     # Split the shuffled data back into features, labels, and sensitive attributes
    #     features = data[:, :-2]  # All columns except the last two are features
    #     labels = data[:, -2]     # Second to last column is labels
    #     sensitive_attributes = data[:, -1] # Last column is sensitive attributes
    #     male_1_x=[] 
    #     male_2_x=[]
    #     male_3_x=[]
    #     male_4_x=[]
    #     male_5_x=[]
    #     fema










        # ## male data and female data
        # for i in range(len(labels)):
        #     if sensitive_attributes[i]==1:
        #         male_x.append(features[i])
        #         male_y.append(labels[i])
        #         male_s.append(sensitive_attributes[i])
        #     else:
        #         female_x.append(features[i])
        #         female_y.append(labels[i])
        #         female_s.append(sensitive_attributes[i])
        # male_x, male_y, male_s = np.array(male_x), np.array(male_y), np.array(male_s)
        # female_x, female_y, female_s = np.array(female_x), np.array(female_y), np.array(female_s)
        # train_x, vali_x, train_y, vali_y, train_s, vali_s = [], [], [], [], [], []

        # for proportion in client_proportions:
        #     num_male = int(2000 * proportion)
        #     num_female = int(2000 * (1 - proportion))

        #     male_sample_idx = np.random.choice(len(male_x), num_male, replace=False)
        #     female_sample_idx = np.random.choice(len(female_x), num_female, replace=False)

        #     client_x = np.vstack((male_x[male_sample_idx], female_x[female_sample_idx]))
        #     client_y = np.hstack((male_y[male_sample_idx], female_y[female_sample_idx]))
        #     client_s = np.hstack((male_s[male_sample_idx], female_s[female_sample_idx]))

        #     # Shuffle the client data
        #     combined_client_data = np.hstack((client_x, client_y.reshape(-1, 1), client_s.reshape(-1, 1)))
        #     np.random.shuffle(combined_client_data)

        #     shuffled_x = combined_client_data[:, :-2]
        #     shuffled_y = combined_client_data[:, -2]
        #     shuffled_s = combined_client_data[:, -1]
    
    print(np.shape(train_x_1), np.shape(train_x_2), np.shape(train_x_3), np.shape(train_x_4), np.shape(train_x_5))
    train_x=np.array(train_x,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x=np.array(vali_x,dtype=np.float32).reshape(-1,28,28,3)/255
    train_x_1=np.array(train_x_1,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x_1=np.array(vali_x_1,dtype=np.float32).reshape(-1,28,28,3)/255
    train_x_2=np.array(train_x_2,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x_2=np.array(vali_x_2,dtype=np.float32).reshape(-1,28,28,3)/255
    train_x_3=np.array(train_x_3,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x_3=np.array(vali_x_3,dtype=np.float32).reshape(-1,28,28,3)/255
    train_x_4=np.array(train_x_4,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x_4=np.array(vali_x_4,dtype=np.float32).reshape(-1,28,28,3)/255
    train_x_5=np.array(train_x_5,dtype=np.float32).reshape(-1,28,28,3)/255
    vali_x_5=np.array(vali_x_5,dtype=np.float32).reshape(-1,28,28,3)/255
    
    
    return train_x,train_y,train_s, vali_x, vali_y, vali_s, train_x_1, vali_x_1, train_y_1, vali_y_1, train_s_1, vali_s_1, train_x_2, vali_x_2, train_y_2, vali_y_2, train_s_2, vali_s_2, train_x_3, vali_x_3, train_y_3, vali_y_3, train_s_3, vali_s_3, train_x_4, vali_x_4, train_y_4, vali_y_4, train_s_4, vali_s_4, train_x_5, vali_x_5, train_y_5, vali_y_5, train_s_5, vali_s_5
             
    


