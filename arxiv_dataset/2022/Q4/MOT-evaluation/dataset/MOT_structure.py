
import os
import shutil

import numpy as np



def move_public_detections(path, end):

    for folder in os.listdir(path):

        new_f  = os.path.join(end, folder)
        origin = os.path.join(path, folder, 'det')

        print(new_f)
        print(origin)

        os.mkdir(new_f)
        shutil.move(origin, new_f)


def move_gt_detections(path, end, labels=[1, 2, 3, 4, 5, 6, 7], type='MOTChall'):

    for folder in os.listdir(path):

        new_f  = os.path.join(end, folder, 'det/')
        origin = os.path.join(path, folder, 'gt/gt.txt')

        print(origin, '->>', new_f)

        # Create folders
        os.makedirs(new_f)


        # Load GT
        fp_in = np.loadtxt(origin, delimiter=',')


        # Get permited labels
        fp_in[:, 1] = -1
        idx = np.where(np.isin(fp_in[:, 7], labels))
        final = fp_in[idx]

        if type == 'VisDrone':
            final = np.delete(final, 7, 1)

        # Save in file
        np.savetxt(new_f + 'det.txt', final, delimiter=',', fmt=['%d', '%d', '%.0f', '%.0f', '%.0f', '%.0f', '%d', '%d', '%.2f'])




#move_public_detections('dataset/MOT20', 'outputs/detections/public/MOT20')
#move_public_detections('dataset/MOT17', 'outputs/detections/public/MOT17')

#move_gt_detections('dataset/MOT20', 'outputs/detections/gt/MOT20')
#move_gt_detections('dataset/MOT17', 'outputs/detections/gt/MOT17')
move_gt_detections('dataset/VisDrone2019-MOT-val', 'outputs/detections/gt/VisDrone2019-MOT-val', type='VisDrone')

