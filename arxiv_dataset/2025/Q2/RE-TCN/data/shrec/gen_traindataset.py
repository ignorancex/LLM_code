import numpy as np
import json
import os

# Root directory of the dataset
root_database_path = './shrec17_dataset/HandGestureDataset_SHREC2017'
# Path to the train_gestures.txt file
train_txt_path = './shrec17_dataset/HandGestureDataset_SHREC2017/train_gestures.txt'
# Load the train_gestures.txt file containing sample indices and labels
train_txt = np.loadtxt(train_txt_path, dtype=int)

# Create directories if they don't exist
json_dir = './shrec17_jsons/train_jsons/'
if not os.path.exists(json_dir):
    os.makedirs(json_dir)

Samples_sum = train_txt.shape[0]  # Number of samples
# print(Samples_sum) # 1960

data_dict = []

# Iterate over each sample
for i in range(Samples_sum):
    idx_gesture = train_txt[i][0]  # Gesture information
    idx_finger = train_txt[i][1]   # Finger information
    idx_subject = train_txt[i][2]  # Subject information
    idx_essai = train_txt[i][3]    # Trial information
    label_14 = train_txt[i][4]     # label_14 tag
    label_28 = train_txt[i][5]     # label_28 tag
    T = train_txt[i][6]            # Number of frames in a single sample

    # Path to the skeleton txt file
    skeleton_path = root_database_path + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai) + '/skeletons_world.txt'

    # Read the skeleton txt file
    skeleton_data = np.loadtxt(skeleton_path)
    # print(skeleton_data.shape) # T * 66

    # T = skeleton_data.shape[0]  # Calculate the number of frames in the sample
    # Reshape to T*N*C (T*22*3)
    skeleton_data = skeleton_data.reshape([T, 22, 3])
    # print(skeleton_data) # T*22*3

    # Get filename
    file_name = "g" + str(idx_gesture).zfill(2) + "f" + str(idx_finger).zfill(2) + \
                "s" + str(idx_subject).zfill(2) + "e" + str(idx_essai).zfill(2)

    # Save each sample's information as a JSON file
    data_json = {"file_name": file_name, "skeletons": skeleton_data.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()}
    # with open("./shrec17_jsons/train_jsons/" + file_name + ".json", 'w') as f:
    with open(json_dir + file_name + ".json", 'w') as f:
        json.dump(data_json, f)

    # Record all samples' information in a dictionary
    tmp_data_dict = {"file_name": file_name, "length": T.tolist(), "label_14": label_14.tolist(), "label_28": label_28.tolist()}
    data_dict.append(tmp_data_dict)

# Save all samples' information as a JSON format file
with open("./shrec17_jsons/" + "train_samples.json", 'w') as t:  # Save all sample information as a JSON file
    json.dump(data_dict, t)
