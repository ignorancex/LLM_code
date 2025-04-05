"""
Dataset Filter

Group COCO images used in VerSe from their original Train/Val splits
to a common folder.
"""
from shutil import copyfile
import pandas as pd

def dataset_filter():
    """
    Copy images listed in '3.5k_verse_gold_image_sense_annotations.csv'
    (from Gella et al. 2019 repository) to 'dataset' directory.
    It is assumed to have such a file located in 'data/labels' folder
    and the images in 'train2014' and 'val2014' folders.
    """
    sense_labels_df = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
    sense_labels_df = sense_labels_df[sense_labels_df['COCO/TUHOI'] == 'COCO']
    #  In VerSe, an image may have multiple verbs, but images are considered only once.
    sense_labels_df = sense_labels_df['image'].drop_duplicates()

    for filename in sense_labels_df:
        if filename.startswith('COCO_train2014'):
            copyfile('train2014/' + filename, 'dataset/' + filename)
        if filename.startswith('COCO_val2014'):
            copyfile('val2014/' + filename, 'dataset/' + filename)


if __name__ == '__main__':
    dataset_filter()
