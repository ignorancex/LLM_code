import argparse

import cv2 as cv


def AnonymiseVideo(videofile_in: str, videofile_out: str, bounds=None):
    """
    Select a bounding box and erase all content outside of it.
    Use the bounding box to define the content that you want to preserve.
    If no bounding box is given, then a UI will show on the first frame
    """
    cap = cv.VideoCapture(videofile_in)
    if (cap.isOpened() == False):
        print('Unable to read video ' + videofile_in)

    # get parameters of input video, which will be the same in the video output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    if bounds == None:
        # Read the first frame to define manually the bounding box
        success, image = cap.read()
        if not success:
            print('I could not read the video')
            exit(-1)

        # Select ROI
        bounds = cv.selectROI(image, showCrosshair=False, fromCenter=False)
        # go back to the first frame
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        print("Bounds are " + str(bounds))
    elif not len(bounds) == 4:
        print("Bounds should be given as a tuple of 4 elements")
        exit(-1)

    # Remove outside of the bounding box for all video
    # Here MJPG is used but other codecs are also available here: http://www.fourcc.org/codecs.php
    out = cv.VideoWriter(videofile_out, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    # checks whether frames were extracted
    success = True
    while True:
        # vidObj object calls read
        # function extract frames
        success, image = cap.read()
        if not success:
            break
        # Crop image
        imCrop = image * 0
        imCrop[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :] = \
            image[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]

        # Write the frame into the file 'output.avi'
        out.write(imCrop)

    cap.release()
    out.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videofile_in', required=True, help='Specify videofile_in')
    parser.add_argument('--videofile_out', required=True, help='Specify videofile_out')
    parser.add_argument('--bounds', required=False, help='Specify bounds', nargs='+', type=int)
    args = parser.parse_args()

    AnonymiseVideo(args.videofile_in, args.videofile_out, args.bounds)

if __name__ == '__main__':
    main()
