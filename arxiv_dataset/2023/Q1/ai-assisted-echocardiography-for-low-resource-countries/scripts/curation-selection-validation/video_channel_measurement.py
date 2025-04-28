import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def VideoChannelMeasurement(videofile_in: str, image_frames_path: str, bounds=None):
    """
     Computes Channel Measurements per Frame
     bounds: (start_x  ,start_y, width, heigh )
    """
    cap = cv.VideoCapture(videofile_in)
    # Check if video opened successfully
    if cap.isOpened() == False:
        print('Unable to read video ' + videofile_in)

    # get parameters of input video, which will be the same in the video output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)

    print('frame_height:', frame_height, 'frame_width:', frame_width, 'fps', fps, 'nframes', nframes)

    i = 0
    rg, rb, gb = [], [], []
    nnz_rg, nnz_rb, nnz_gb = [], [], []
    nz_rg, nz_rb, nz_gb = [], [], []

    while True:
        success, image_ = cap.read()
        if not success:
            break

        image = image_[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]
        r, g, b = image[..., 2].astype(float), image[..., 1].astype(float), image[..., 0].astype(float)

        # image wide rgb
        rg.append(np.mean(np.abs(r - g)))
        rb.append(np.mean(np.abs(r - b)))
        gb.append(np.mean(np.abs(g - b)))
        # n pixels not gray
        nnz_rg.append(np.count_nonzero(np.abs(r - g) > 1))
        nnz_rb.append(np.count_nonzero(np.abs(r - b) > 1))
        nnz_gb.append(np.count_nonzero(np.abs(g - b) > 1))
        # statistics of non gray pixels
        nz_rg.append(np.mean((r - g)[np.abs(r - g) > 1]))
        nz_rb.append(np.mean((r - b)[np.abs(r - b) > 1]))
        nz_gb.append(np.mean((g - b)[np.abs(g - b) > 1]))

        font = cv.FONT_HERSHEY_SIMPLEX

        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        image = cv.putText(image, 'R-G: {:.2f}'.format(rg[-1]), (50, 100), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'R-B: {:.2f}'.format(rb[-1]), (50, 150), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'G-B: {:.2f}'.format(gb[-1]), (50, 200), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'R-G nnz: {}'.format(nnz_rg[-1]), (50, 250), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'R-B nnz: {}'.format(nnz_rb[-1]), (50, 300), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'G-B nnz: {}'.format(nnz_gb[-1]), (50, 350), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'R-G nz: {:.2f}'.format(nz_rg[-1]), (50, 400), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'R-B nz: {:.2f}'.format(nz_rb[-1]), (50, 450), font, fontScale, color, thickness,
                           cv.LINE_AA)
        image = cv.putText(image, 'G-B nz: {:.2f}'.format(nz_gb[-1]), (50, 500), font, fontScale, color, thickness,
                           cv.LINE_AA)

        # print(image.shape)
        # plt.imshow(r)
        # plt.show()

        cv.imwrite(image_frames_path+'/nframes{:05d}.png'.format(i), image)

        print('Frame_index/number_of_frames: ', i, '/', nframes)
        i += 1

    plt.subplot(1, 3, 1)
    plt.plot(rg, 'r-', label='r-g')
    plt.plot(rb, 'g-', label='r-b')
    plt.plot(gb, 'b-', label='g-b')
    plt.legend()
    plt.title('Average Absolute difference')
    plt.subplot(1, 3, 2)
    plt.plot(nnz_rg, 'r-', label='r-g')
    plt.plot(nnz_rb, 'g-', label='r-b')
    plt.plot(nnz_gb, 'b-', label='g-b')
    plt.title('NNZ in subtraction')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(nz_rg, 'r-', label='r-g')
    plt.plot(nz_rb, 'g-', label='r-b')
    plt.plot(nz_gb, 'b-', label='g-b')
    plt.title('Average difference over NZNNZ')
    plt.legend()
    plt.savefig(image_frames_path+"output.jpg")
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videofile_in', required=True, help='Specify videofile_in')
    parser.add_argument('--image_frames_path', required=True, help='Specify image_frames_path')
    parser.add_argument('--bounds', required=False, help='Specify bounds', nargs='+', type=int)
    args = parser.parse_args()

    VideoChannelMeasurement(args.videofile_in, args.image_frames_path, args.bounds)

    # #bounds =
    # (638, 135, 821, 722)?
    # (618, 202, 848, 646)?
    # 200, 50, 1200, 1000?

if __name__ == '__main__':
    main()
