"""Download TUHOI

VerSe images are downloaded from TUHOI repository to
'data/images/TUHOI/' folder.
"""
import pandas as pd
import requests
from tqdm import tqdm


SENSE_LABELS = pd.read_csv('data/labels/3.5k_verse_gold_image_sense_annotations.csv')
SENSE_LABELS = SENSE_LABELS[SENSE_LABELS['COCO/TUHOI'] == 'TUHOI']
SENSE_LABELS = SENSE_LABELS.drop_duplicates()

URL = 'http://disi.unitn.it/~dle/images/DET/'
DOWNLOAD_FOLDER = 'data/images/TUHOI/'

for i, row in tqdm(enumerate(SENSE_LABELS.itertuples())):
    if row.image.startswith('ILSVRC'):
        folder = 'val/'
    elif row.image.startswith('n'):
        folder = 'train/'
    else:
        print('Unknown prefix, image name: %s' % row.image)

    r = requests.get(URL + folder + row.image)

    with open(DOWNLOAD_FOLDER + row.image, 'wb') as f:
        f.write(r.content)
