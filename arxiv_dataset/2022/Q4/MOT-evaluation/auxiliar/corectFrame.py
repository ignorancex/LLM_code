
import os
import numpy as np

def frame_number(file, plus=1):

    file[:, 0] = file[:, 0] + 1

    return file



def change_order(file):

    file[:, [2, 3, 4, 5]] = file[:, [3, 2, 5, 4]]

    return file

def resize_detections(file):

    file[:, [2, 4]] = file[:, [2, 4]] * 4
    file[:, [3, 5]] = file[:, [3, 5]] * 2

    return file


def center2xy(file):

    file[:, 2] = file[:, 2] - file[:, 4] / 2
    file[:, 3] = file[:, 3] - file[:, 5] / 2

    return file


def selectClass(file, label):

    idx = np.where(file[:, 7] == label)[0]

    new_f = file[idx]

    return new_f


def setThreshold(file, threshold):

    idx = np.where(file[:, 6] >= threshold)[0]

    new_f = file[idx]

    return new_f


if __name__ == '__main__':

    detector = 'efficientdet-d0-mod-2'
    sets = ['MOT17', 'MOT20', 'VisDrone2019-MOT-val']

    print(detector)


    for set_name in sets:

        list_subsets = os.listdir( os.path.join('outputs/detections', detector, set_name) )

        print(set_name)

        for subset in list_subsets:

            if not os.path.isdir(os.path.join('outputs/detections', detector, set_name, subset)): continue

            print('->', subset)

            path = os.path.join('outputs/detections', detector, set_name, subset, 'det/det.txt')

            file = np.loadtxt(path, delimiter=',')


            #print(file.shape)

            # file = frame_number(file)
            # file = change_order(file)
            # file = resize_detections(file)
            # file = center2xy(file)
            #file = selectClass(file, 1)
            #file = setThreshold(file, threshold=0.5)
            file = setThreshold(file, threshold=0.8)

            #print(file.shape)


            np.savetxt(path, file, delimiter=',', fmt=['%d', '%d', '%.0f', '%.0f', '%.0f', '%.0f', '%.4f', '%d', '%d', '%d'])


        print()
