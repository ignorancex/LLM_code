import numpy as np
import json
import os

# Root directory of the dataset
root_dataset_path = './DHG14-28_dataset'
# Path to the sample information txt file
sample_information_txt = root_dataset_path + '/informations_troncage_sequences.txt'

# Read the sample information txt file
sample_txt = np.loadtxt(sample_information_txt, dtype=int)

# Total number of samples
Samples_sum = sample_txt.shape[0]
# Total number of subjects
num_subject = 20

# Initialize lists to hold training and validation data dictionaries
train_data_dict = [[] for i in range(Samples_sum)]
val_data_dict = [[] for i in range(Samples_sum)]

# Iterate over each sample
for i in range(Samples_sum):
    # Extract sample details
    idx_gesture = sample_txt[i][0]  # Gesture information
    idx_finger = sample_txt[i][1]   # Finger information
    idx_subject = sample_txt[i][2]  # Subject information
    idx_essai = sample_txt[i][3]    # Trial information
    begin_frame = sample_txt[i][4]  # Start frame of valid frames
    end_frame = sample_txt[i][5]    # End frame of valid frames
    T = end_frame - begin_frame + 1  # Number of frames in a single sample

    # Path to the skeleton txt file
    skeleton_path = root_dataset_path + '/gesture_' + str(idx_gesture) + '/finger_' + str(idx_finger) \
                    + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai) + '/skeleton_world.txt'

    # Read the skeleton txt file
    skeleton_data = np.loadtxt(skeleton_path)
    # Extract valid frames
    skeleton_data = skeleton_data[begin_frame:end_frame + 1, :]
    # Reshape to T*N*C (T*22*3)
    skeleton_data = skeleton_data.reshape([T, 22, 3])

    # Generate filename
    file_name = "g" + str(idx_gesture).zfill(2) + "f" + str(idx_finger).zfill(2) \
                + "s" + str(idx_subject).zfill(2) + "e" + str(idx_essai).zfill(2)

    label_14 = idx_gesture
    # Generate label_28 based on the number of fingers used
    if idx_finger == 1:
        label_28 = idx_gesture
    else:
        label_28 = idx_gesture + 14

    # Save each sample's information as a JSON file
    data_json = {
        "file_name": file_name,
        "skeletons": skeleton_data.tolist(),
        "label_14": label_14.tolist(),
        "label_28": label_28.tolist()
    }

    # Record sample's information using a dictionary
    tmp_data_dict = {
        "file_name": file_name,
        "length": T.tolist(),
        "label_14": label_14.tolist(),
        "label_28": label_28.tolist()
    }

    # Split data into training and validation sets for each subject
    for idx in range(num_subject):
        val_dir = "./DHG14-28_sample_json/" + str(idx + 1) + "/val/"
        train_dir = "./DHG14-28_sample_json/" + str(idx + 1) + "/train/"
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)

        # Assign to validation set if subject matches, else to training set
        if idx == int(idx_subject) - 1:
            with open("./DHG14-28_sample_json/" + str(idx + 1) + '/val/' + file_name + ".json", 'w') as f:
                json.dump(data_json, f)
            val_data_dict[idx].append(tmp_data_dict)
        else:
            with open("./DHG14-28_sample_json/" + str(idx + 1) + '/train/' + file_name + ".json", 'w') as f:
                json.dump(data_json, f)
            train_data_dict[idx].append(tmp_data_dict)

# Save the list of training and validation samples for each subject
for idx in range(num_subject):
    with open("./DHG14-28_sample_json/" + str(idx + 1) + "/" + str(idx + 1) + "train_samples.json", 'w') as t1:
        json.dump(train_data_dict[idx], t1)
    with open("./DHG14-28_sample_json/" + str(idx + 1) + "/" + str(idx + 1) + "val_samples.json", 'w') as t2:
        json.dump(val_data_dict[idx], t2)
