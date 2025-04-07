"""
constructing ucf-101 image-only dataset from
ucf-101 video dataset:
1. name preprocessing
2. for each clip, extract its middle frame as the image data


donwload link
https://www.crcv.ucf.edu/data/UCF101/UCF101.rar ~ 6.7 GB
"""

import os 
import sys
import cv2
from tqdm import tqdm

sys.path.append('../')
from utils.directory import load_config
from utils.directory import get_sub_dir_list


"""
#############################################################
manual download ucf-101 from https://www.crcv.ucf.edu/data/UCF101.php 
link: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar ~ 6.7 GB
unzip the file to the directory specified at the bottom of this script:
src_data_dir = os.path.join(ucf101_download_dir, 'ucf-101-original') 
# rename UCF101 to ucf-101-original, it has one subfolder: UCF-101

manual download ucf-101 train/test split file from https://www.crcv.ucf.edu/data/UCF101.php 
link: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip ~ 1.54 MB
unzip the file to the directory specified at the botoom of this script:
#############################################################
"""


def ucf101_name_preprocessing(org_names):
    classes = [
        'Apply Eye Makeup',
        'Apply Lipstick',
        'Archery',
        'Baby Crawling',
        'Balance Beam',
        'Band Marching',
        'Baseball Pitch',
        'Basketball',
        'Basketball Dunk',
        'Bench Press',
        'Biking',
        'Billiards',
        'Blow Dry Hair',
        'Blowing Candles',
        'Body Weight Squats',
        'Bowling',
        'Boxing Punching Bag',
        'Boxing Speed Bag',
        'Breast Stroke',
        'Brushing Teeth',
        'Clean And Jerk',
        'Cliff Diving',
        'Cricket Bowling',
        'Cricket Shot',
        'Cutting In Kitchen',
        'Diving',
        'Drumming',
        'Fencing',
        'Field Hockey Penalty',
        'Floor Gymnastics',
        'Frisbee Catch',
        'Front Crawl',
        'Golf Swing',
        'Haircut',
        'Hammer Throw',
        'Hammering',
        'Hand Stand Pushups',
        'Handstand Walking',
        'Head Massage',
        'High Jump',
        'Horse Race',
        'Horse Riding',
        'Hula Hoop',
        'Ice Dancing',
        'Javelin Throw',
        'Juggling Balls',
        'Jump Rope',
        'Jumping Jack',
        'Kayaking',
        'Knitting',
        'Long Jump',
        'Lunges',
        'Military Parade',
        'Mixing',
        'Mopping Floor',
        'Nunchucks',
        'Parallel Bars',
        'Pizza Tossing',
        'Playing Cello',
        'Playing Daf',
        'Playing Dhol',
        'Playing Flute',
        'Playing Guitar',
        'Playing Piano',
        'Playing Sitar',
        'Playing Tabla',
        'Playing Violin',
        'Pole Vault',
        'Pommel Horse',
        'Pull Ups',
        'Punch',
        'Push Ups',
        'Rafting',
        'Rock Climbing Indoor',
        'Rope Climbing',
        'Rowing',
        'Salsa Spin',
        'Shaving Beard',
        'Shotput',
        'Skate Boarding',
        'Skiing',
        'Skijet',
        'Sky Diving',
        'Soccer Juggling',
        'Soccer Penalty',
        'Still Rings',
        'Sumo Wrestling',
        'Surfing',
        'Swing',
        'Table Tennis Shot',
        'Tai Chi',
        'Tennis Swing',
        'Throw Discus',
        'Trampoline Jumping',
        'Typing',
        'Uneven Bars',
        'Volleyball Spiking',
        'Walking With Dog',
        'Wall Pushups',
        'Writing On Board',
        'Yo Yo',
    ]

    classes = [name.lower() for name in classes]

    assert len(org_names) == len(classes), f"incorrect number of classes {len(org_names)}"

    org_names = [name.lower() for name in org_names]

    old_to_new_names = dict()
    org_names_set = set(org_names)
    for name in classes:
        name_tp = name.replace(' ','')
        if name_tp in org_names_set:
            old_to_new_names[name_tp] = name
        else:
            raise ValueError(f'{name} has no match in original names')
        
    return old_to_new_names


def extract_middle_frame(path):
    # check for tailing '\n' in path
    if '\n' in path: path = path[:-1] # remove tail '\n'

    video = cv2.VideoCapture(path)
    if video.isOpened():
        # number of frames
        N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        mid_frame = int(N_frames / 2)

        # set mid_frame to read
        video.set(cv2.CAP_PROP_POS_FRAMES, mid_frame-1)
        _, frame = video.read()
        video.release()
    else:
        raise ValueError(f'video at {path} not opened correctly')
        
    
    return frame


def construct_ucf101_img_val_set(src_data_dir, dest_home_dir, original_train_path):
    """
        src_data_dir: video data source directory
        dest_home_dir: our dataset's home directory
        original_train_path: the 1st training set split from the original UCF-101 dataset
        we use the 1st training set split (out of all 3 training set splits) as our UCF-101 image dataset's val split
    """

    # get all folder names (old class names)
    old_classnames = get_sub_dir_list(src_data_dir)

    old_to_new_classnames = ucf101_name_preprocessing(old_classnames)

    with open(original_train_path, 'r') as f:
        paths = list(f)

    # training paths: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
    paths = [path.split(' ')[0] for path in paths]
    
    for path in tqdm(paths, total=len(paths)):
        tp = path.split('/')
        old_name, file_name = tp[0], tp[1]
        old_name = old_name.lower()
        new_name = old_to_new_classnames[old_name]
        
        src_vid_path = os.path.join(src_data_dir,path)
        
        dest_dir = os.path.join(dest_home_dir, 'val', new_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # extract 1 frame per clip -> the middle frame
        dest_frame_path = os.path.join(dest_dir, file_name.split('.')[0] + '.jpg')
        frame = extract_middle_frame(src_vid_path)
        cv2.imwrite(dest_frame_path, frame)


    return


def construct_ucf101_img_test_set(src_data_dir, dest_home_dir, original_test_path):
    """
        src_data_dir: video data source directory
        dest_home_dir: our dataset's home directory
        original_test_path: the 1st test set split from the original UCF-101 dataset
        we use the 1st test set split (out of all 3 test set splits) as our UCF-101 image dataset's test split
    """

    # get all folder names (old class names)
    old_classnames = get_sub_dir_list(src_data_dir)

    old_to_new_classnames = ucf101_name_preprocessing(old_classnames)

    with open(original_test_path, 'r') as f:
        paths = list(f)

    # training paths: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
    paths = [path.split(' ')[0] for path in paths]
    
    for path in tqdm(paths, total=len(paths)):
        tp = path.split('/')
        old_name, file_name = tp[0], tp[1]
        old_name = old_name.lower()
        new_name = old_to_new_classnames[old_name]
        
        src_vid_path = os.path.join(src_data_dir,path)
        
        dest_dir = os.path.join(dest_home_dir, 'test', new_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # extract 1 frame per clip -> the middle frame
        dest_frame_path = os.path.join(dest_dir, file_name.split('.')[0] + '.jpg')
        frame = extract_middle_frame(src_vid_path)
        cv2.imwrite(dest_frame_path, frame)

    return




if __name__ == '__main__':
    config_path = '../data_paths.yml'
    config = load_config(config_path)
    ucf101_download_dir = config['dataset-home-dir']
    

    src_data_dir = os.path.join(ucf101_download_dir, 'ucf-101-original', 'UCF-101')
    dest_home_dir = config['ucf-101']
    src_train_split_path = os.path.join(
        ucf101_download_dir, 'ucf-101-original', 'UCF101TrainTestSplits-RecognitionTask', 
        'ucfTrainTestlist', 'trainlist01.txt')
    src_test_split_path = os.path.join(
        ucf101_download_dir, 'ucf-101-original', 'UCF101TrainTestSplits-RecognitionTask', 
        'ucfTrainTestlist', 'testlist01.txt')
    
    construct_ucf101_img_val_set(src_data_dir, dest_home_dir, src_train_split_path)
    construct_ucf101_img_test_set(src_data_dir, dest_home_dir, src_test_split_path)

    print('done')
    