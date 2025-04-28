import argparse

import cv2 as cv
import numpy as np


def VideoToSlidingVideo(videofile_in: str, videofile_out: str, boxwidth=None):
    """
    Crop each frame with a floating bounding box that moves around the image.
    The size of the crop is defined by "2*boxwidth".
    The crop box "floats" around the video in a linear trajectory between two random start and end points.
    boxwidth = (rows, columns)  # rows x columns the size of the video will be twice this
    """
    cap = cv.VideoCapture(videofile_in)
    # Check if video opened successfully
    if (cap.isOpened() == False):
        print('Unable to read video ' + videofile_in)

    # get parameters of input video, which will be the same in the video output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)

    print('frame_height:', frame_height, 'frame_width:', frame_width, 'fps', fps, 'nframes', nframes)

    p0 = np.array((int(np.random.uniform(boxwidth[0], frame_width - boxwidth[0])),
                   int(np.random.uniform(boxwidth[1], frame_height - boxwidth[1]))))
    p1 = np.array((int(np.random.uniform(boxwidth[0], frame_width - boxwidth[0])),
                   int(np.random.uniform(boxwidth[1], frame_height - boxwidth[1]))))

    out = cv.VideoWriter(videofile_out, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                         (2 * boxwidth[0], 2 * boxwidth[1]))

    # checks whether frames were extracted
    success = True
    i = 0
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = cap.read()
        if not success:
            break

        # get image region
        p = ((p0 * (nframes - i) + p1 * i) / nframes).astype(np.int64)  # centre of the box
        imCrop = image[p[1] - boxwidth[1]:p[1] + boxwidth[1], p[0] - boxwidth[0]:p[0] + boxwidth[0], :]
        out.write(imCrop)
        print('Frame index / number of frames ', i, '/', nframes)
        i += 1

    cap.release()
    out.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videofile_in', required=True, help='Specify videofile_in')
    parser.add_argument('--videofile_out', required=True, help='Specify videofile_out')
    parser.add_argument('--bounds', required=False, help='Specify bounds', nargs='+', type=int)
    args = parser.parse_args()

    VideoToSlidingVideo(args.videofile_in, args.videofile_out, args.bounds)

if __name__ == '__main__':
    main()
