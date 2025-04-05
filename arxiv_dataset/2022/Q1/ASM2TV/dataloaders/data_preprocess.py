import os
import pandas as pd
import numpy as np

# dataset_name = 'realword2016_dataset'

def parseDataset(dataset_name, num_tasks, time_slot, skip_slot, train_split_ratio, single_sensor):
    if 'realworld2016' in dataset_name:
        root_path = 'data/{}'.format(dataset_name)
        # sensors = ['acc', 'gps', 'gyr', 'lig', 'mag', 'mic']
        if single_sensor:
            sensors = ['acc']
            dim = 3
        else:
            sensors = ['acc', 'gyr', 'mag'] # For simplicity, we only select acc sensor for this case
            dim = 9
        actions = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
        views = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
        skip_window_size = int(skip_slot * 50.)
        window_size = int(time_slot * 50.) # Frequency of sensor in Hz
        # Determinate the max_len for each action
        len_map = {'climbingdown': 20000, 'climbingup': 20000, 'jumping': 4000, 'lying': 30000, 'running': 30000, 'sitting': 30000,
                    'standing': 30000, 'walking': 30000}
        # Save paths
        if not os.path.exists(os.path.join(root_path, 'multi_sensors_acc_gyr_mag')):
            os.mkdir(os.path.join(root_path, 'multi_sensors_acc_gyr_mag'))
            for folder in ['train-sup', 'train-unsup', 'val', 'test']:
                os.mkdir(os.path.join(root_path, 'multi_sensors_acc_gyr_mag', folder))
        if not os.path.exists(os.path.join(root_path, 'single_sensor_acc')):
            os.mkdir(os.path.join(root_path, 'single_sensor_acc'))
            for folder in ['train-sup', 'train-unsup', 'val', 'test']:
                os.mkdir(os.path.join(root_path, 'single_sensor_acc', folder))

        if not single_sensor:
            task_save_path_data = os.path.join(root_path, 'multi_sensors_acc_gyr_mag')
            task_save_path_label = os.path.join(root_path, 'multi_sensors_acc_gyr_mag')
        else:
            task_save_path_data = os.path.join(root_path, 'single_sensor_acc')
            task_save_path_label = os.path.join(root_path, 'single_sensor_acc')

        for taskid in range(1, num_tasks+1):
            task_path = os.path.join(root_path, 'proband{}/data'.format(taskid))
            
            train_data, unsup_data, val_data, test_data = [], [], [], []
            train_labels, unsup_labels, val_labels, test_labels = [], [], [], []
            print(taskid)
            try:
                for i, action in enumerate(actions):
                    print(action)
                    view_data = []
                    for view in views:
                        print(view)
                        max_len = len_map[action]
                        sensor_data = []
                        for sensor in sensors:
                            if sensor == 'gyr':
                                new_sensor_name = 'Gyroscope'
                            elif sensor == 'lig':
                                new_sensor_name = 'Light'
                            elif sensor == 'mag':
                                new_sensor_name = 'MagneticField'
                            elif sensor == 'mic':
                                new_sensor_name = 'Microphone'
                            elif sensor == 'gps':
                                new_sensor_name = 'GPS'
                            else:
                                new_sensor_name = 'acc'
                            temp_path = os.path.join(task_path, '{}_{}_csv'.format(sensor, action))
                            temp_data_path = os.path.join(temp_path, '{}_{}_{}.csv'.format(new_sensor_name, action, view))
                            if not os.path.exists(temp_data_path):
                                temp_data_path = os.path.join(temp_path, '{}_{}_2_{}.csv'.format(new_sensor_name, action, view))
                            temp_data = pd.read_csv(temp_data_path, header=0)
                            temp_data = temp_data.iloc[:max_len, 2:].values
                            sensor_data.append(temp_data)
                        sensor_data = np.concatenate(sensor_data, axis=1) # shape: [max_len, dim * num_sensors]
                        view_data.append(sensor_data[np.newaxis, ...])
                    view_data = np.concatenate(view_data, axis=0) # shape: [num_views, max_len, dim * num_sensors]
                    print(view_data.shape)
                    total_samples = (max_len - window_size) // skip_window_size
                    train_len = int(total_samples * train_split_ratio[0])
                    val_len = int(total_samples * train_split_ratio[1])
                    for j in range(train_len//5):
                        # print(j)
                        for z in [0]:
                            train_data.append(view_data[np.newaxis, :, (j*5+z)*skip_window_size:(j*5+z)*skip_window_size+window_size, :]) # shape: [num_views, window_size, dim * num_sensors]
                            train_labels.append(i)
                    for j in range(train_len//5):
                        for z in [1, 2, 3, 4]:
                            # print(j)
                            unsup_data.append(view_data[np.newaxis, :, (j*5+z)*skip_window_size:(j*5+z)*skip_window_size+window_size, :]) # shape: [num_views, window_size, dim * num_sensors]
                            unsup_labels.append(i)
                    for j in range(train_len, train_len+val_len):
                        val_data.append(view_data[np.newaxis, :, j*skip_window_size:j*skip_window_size+window_size, :]) # shape: [num_views, window_size, dim * num_sensors]
                        val_labels.append(i)
                    for j in range(train_len+val_len, total_samples):
                        test_data.append(view_data[np.newaxis, :, j*skip_window_size:j*skip_window_size+window_size, :]) # shape: [num_views, window_size, dim * num_sensors]
                        test_labels.append(i)
                train_data, unsup_data, val_data, test_data = np.concatenate(train_data, axis=0), \
                                                                np.concatenate(unsup_data, axis=0), \
                                                                np.concatenate(val_data, axis=0), \
                                                                np.concatenate(test_data, axis=0)
                # print(train_data.shape)
                train_labels, unsup_labels, val_labels, test_labels = np.array(train_labels), \
                                                                np.array(unsup_labels), \
                                                                np.array(val_labels), \
                                                                np.array(test_labels)
                # print(train_data.size(), unsup_data.size())
                
                np.save(os.path.join(task_save_path_data, 'train-sup', 'proband{}_data.npy'.format(taskid)), train_data)
                np.save(os.path.join(task_save_path_data, 'train-unsup', 'proband{}_data.npy'.format(taskid)), unsup_data)
                np.save(os.path.join(task_save_path_data, 'val', 'proband{}_data.npy'.format(taskid)), val_data)
                np.save(os.path.join(task_save_path_data, 'test', 'proband{}_data.npy'.format(taskid)), test_data)
                
                np.save(os.path.join(task_save_path_label, 'train-sup', 'proband{}_label.npy'.format(taskid)), train_labels)
                np.save(os.path.join(task_save_path_label, 'train-unsup', 'proband{}_label.npy'.format(taskid)), unsup_labels) 
                np.save(os.path.join(task_save_path_label, 'val', 'proband{}_label.npy'.format(taskid)), val_labels)
                np.save(os.path.join(task_save_path_label, 'test', 'proband{}_label.npy'.format(taskid)), test_labels)
                
            except:
                continue

if __name__ == '__main__':
    dataset_name = 'realworld2016_dataset'
    single_sensor_or_not = False
    if single_sensor_or_not:
        root_path = 'data/{}/{}'.format(dataset_name, 'single_sensor_acc')
    else:
        root_path = 'data/{}/{}'.format(dataset_name, 'multi_sensors_acc_gyr_mag')
    # taskids = [1, 3, 9, 10, 11, 12, 15]
    if not os.path.exists(os.path.join(root_path, 'train-sup', 'proband3_data.npy')): # They both have proband3 dataset
        max_num_tasks = 15
        time_slot = 5.
        skip_slot = 1.
        train_split_ratio = [0.5, 0.1, 0.4]
        parseDataset(dataset_name, max_num_tasks, time_slot, skip_slot, train_split_ratio, single_sensor_or_not)
    taskid = 3
    task_data = np.load(os.path.join(root_path, 'train-sup', 'proband{}_data.npy'.format(taskid)), allow_pickle=True)
    task_label = np.load(os.path.join(root_path, 'train-sup', 'proband{}_label.npy'.format(taskid)), allow_pickle=True)
    print(task_data.shape) # data shape : [num_samples, num_views, time_len, dim]
    print(task_label.shape)
    