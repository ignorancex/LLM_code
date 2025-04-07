
import os
import cv2
import argparse
import numpy as np

# Example of execution:
# $ python dataset/create_set.py --data dataset/MSS_SecuenciasVideo2 --name MSS --verbose 1


def parseInput():
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Create a set of data from a folder.')

    # Select an able detector from list.
    parser.add_argument("--data", help="Path to the folder with the content.", required=True)
    parser.add_argument("--name", help="Name for the dataset folder.", required=True)

    # Important arguments, but optional.
    # parser.add_argument("--gt", help="Path to the GT of the set", default=None)
    # parser.add_argument("--det", help="Path to the public detections of the set.", default=None)

    # Other optional arguments
    parser.add_argument("-v", "--verbose", help="Show debug information.", default=0, type=int)

    return parser.parse_args()



def video2img(path_in, path_out, verbose=0):

    # Video read.
    vidcap = cv2.VideoCapture(path_in)

    count = 1
    image_prev = None


    while True:

        # Read frame
        success, image = vidcap.read()


        if not success: break
        elif np.array_equal(image, image_prev): continue


        # Save frame as JPEG file
        file_name = os.path.join(path_out, "%.6d.jpg" % count)
        cv2.imwrite(file_name, image)


        image_prev = image

        count += 1




if __name__ == '__main__':

    args = parseInput()

    files = os.listdir(args.data)

    data_folder = os.path.join('dataset/', args.name)
    os.makedirs( data_folder )


    for f in files:

        if args.verbose: print('Processing:', f)

        path_out = os.path.join(data_folder, os.path.splitext(f)[0], 'img1')
        os.makedirs( path_out )

        path_in = os.path.join(args.data, f)

        video2img(path_in, path_out)