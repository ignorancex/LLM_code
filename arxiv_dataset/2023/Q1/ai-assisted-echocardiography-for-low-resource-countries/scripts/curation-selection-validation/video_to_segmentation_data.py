"""
This is a brief script to take data from videos and try to compute images from it.
"""

import cv2 as cv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

global verbose
verbose = False


def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """Iteratively expand the markers white keeping them limited by the mask during each iteration.

    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    :returns A copy of the last expansion.
    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        cv.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded


def contour_to_mask(frame, th=20):

    # by default BGR

    #cv.imwrite('/home/ag09/data/VITAL/frame.png', frame)
    R = frame[..., 2].astype(np.float)
    G = frame[..., 1].astype(np.float)
    B = frame[..., 0].astype(np.float)

    BR = B-R
    #cv.imwrite('/home/ag09/data/VITAL/BR.png', BR)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    BR_closed = cv.morphologyEx(BR, cv.MORPH_CLOSE, kernel)
    #cv.imwrite('/home/ag09/data/VITAL/BR_closed.png', BR_closed)

    BR_closed_th = (BR_closed > th).astype(np.uint8)
    #cv.imwrite('/home/ag09/data/VITAL/BR_closed_th.png', BR_closed_th*255)
    marker = np.zeros_like(BR_closed_th)
    marker[0, 0] = 1
    mask_all = 1-imreconstruct(marker, 1-BR_closed_th, radius=1)
    #cv.imwrite('/home/ag09/data/VITAL/mask_all.png', mask_all * 255)

    # get largest connected component
    # Connected components with stats.
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask_all, connectivity=4)

    if nb_components > 1:
        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        max_label, mask_size = max([(i, stats[i, cv.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])

        mask = np.zeros_like(mask_all)
        mask[output ==  max_label] = 1
    else:
        mask = mask_all
        mask_size = np.count_nonzero(mask_all)
    #cv.imwrite('/home/ag09/data/VITAL/mask.png', mask * 255)
    return mask, mask_size


def msec_to_timestamp(current_timestamp):
    minutes = int(current_timestamp / 1000 / 60)
    seconds = int(np.floor(current_timestamp / 1000) % 60)
    ms = current_timestamp - np.floor(current_timestamp / 1000) * 1000
    current_contour_frame_time = '{:02d}:{:02d}:{:.3f}'.format(minutes, seconds, ms)
    return current_contour_frame_time


def remove_moving_objects(no_contour_images, idx = None, path_out = None):

    moving_images = []
    for i, im in enumerate(no_contour_images):
        if idx is not None and verbose == True:
            os.makedirs('{}/nocontour'.format(path_out), exist_ok=True)
            cv.imwrite('{}/nocontour/nocontour_{}_{}.png'.format(path_out, idx, i), im)
        if i == 0:
            moving_images.append(im)
        else:
            #d = np.max(np.abs(moving_images[-1].astype(np.float)-im.astype(np.float)))
            #
            #if d > 20:# and d_top < 5:
            r, g, b = im[..., 2].astype(np.float), im[..., 1].astype(np.float), im[..., 0].astype(np.float)
            std_rgb = np.std(im, axis=2)
            th = 0.5
            r[std_rgb > th] = np.nan
            g[std_rgb > th] = np.nan
            b[std_rgb > th] = np.nan

            im_nan = np.stack([b, g, r], axis=2)

            moving_images.append(im_nan)

    no_contour_images = np.stack(moving_images, axis=3)

    M = np.nanmedian(no_contour_images, axis=3).astype(np.uint8)
    return M


def not_suitable_image(image):
    """
    Checks if the image, for example if there is mmode in the image
    """
    empty_block = image[500:600,380:480,:]
    th = 10
    if np.mean(empty_block) > th:
        return True
    return False

def search_for_no_annot_frame(cap, img, bounds, th_BR_contour, th_GR_contour, mode='back', max_frame=-1):
    """
    max frame is the furthest frame to search
    """
    found=-1
    found_ms, found_img = None, None
    while True:
        current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
        if mode == 'back':
            previous_frame = current_frame - 1
            if max_frame > -1 and previous_frame < max_frame:
                break

        else:
            previous_frame = current_frame + 1
            if max_frame > -1 and previous_frame > max_frame:
                break
        if previous_frame >= 0 and previous_frame < cap.get(cv.CAP_PROP_FRAME_COUNT):
            cap.set(cv.CAP_PROP_POS_FRAMES, previous_frame-1)
            success, image = cap.read()
            if not success:
                break

            # Crop image
            cropped_image = image[int(bounds[1]):int(bounds[1] + bounds[3]),
                            int(bounds[0]):int(bounds[0] + bounds[2]), :]

            # see if the image is similar to the current frame
            cropped_img = img[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]
            imdiff = np.abs(cropped_image.astype(np.float32)/1 - cropped_img.astype(np.float32)/1)

            if np.mean(imdiff[:100, ...]) > 5:
                # they are different, exit
                break

            if not_suitable_image(image):
                break

            # see if cropped image has annotation
            R = cropped_image[..., 2].astype(np.float)
            G = cropped_image[..., 1].astype(np.float)
            B = cropped_image[..., 0].astype(np.float)
            th_br = 20
            th_gr = 20
            BR = (B - R)[np.abs(B - R) > th_br]
            GR = (G - R)[np.abs(G - R) > th_gr]

            current_frame_has_blue_contour = bool(np.mean(BR) > th_BR_contour)
            current_frame_has_green_contour = bool(np.mean(GR) > th_GR_contour)
            #
            if current_frame_has_blue_contour is not True and current_frame_has_green_contour is not True:
                found=previous_frame
                found_ms = cap.get(cv.CAP_PROP_POS_MSEC)
                found_img = image
                break
        else:
            break

    return found, found_ms, found_img


def ProcessVideo(videofile_in, bounds=(), th_d = 25, th_GB_contour = -5, th_BR_contour = 20, th_GR_contour = 20, dsize = (256, 256), min_mask_size=50000):
    """Select a bounding box and erase all content outside of it. Use the bounding
    box to define the content that you want to preserve.
    If no bounding box is given, then a UI will show on the first frame"""
    # min_contour_pixels = 6000
    # max_no_contour_pixels = 2000

    path, videoname  = os.path.split(videofile_in)
    path_prefix = os.path.splitext(videoname)[0].replace(" ", "_")

    path_out = '{}{}{}_annotations'.format(path, os.path.sep, path_prefix)


    cap = cv.VideoCapture(videofile_in)
    # Check if video opened successfully
    if (cap.isOpened() == False):
        print('Unable to read video ' + videofile_in)

    # get parameters of input video, which will be the same in the video output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)

    if len(bounds) == 0:
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
        cv.destroyAllWindows()

        bounds_detect = cv.selectROI(image, showCrosshair=False, fromCenter=False)
        # go back to the first frame
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        print("Bounds detect are  " + str(bounds_detect))
        cv.destroyAllWindows()
    elif not len(bounds) == 4:
        print("Bounds should be given as a tuple of 4 elements")
        exit(-1)

    # Now redo the bounds to fit a desired  aspect ratio
    if bounds[2] > bounds[3]*dsize[0]/dsize[1]:
        diff = bounds[2]-int(bounds[3]*dsize[0]/dsize[1])
        dif_l = int(diff/2)
        dif_r = diff-dif_l
        bounds = (bounds[0]+dif_l, bounds[1], bounds[2]-(dif_l + dif_r), bounds[3])
    else:
        diff = int(bounds[3]*dsize[0]/dsize[1]) - bounds[2]
        dif_l = int(diff / 2)
        dif_r = diff - dif_l
        bounds = (bounds[0], bounds[1]+dif_l , bounds[2], bounds[3]-(dif_l + dif_r) )

    print('Bounds adapted to {} to maintain aspect ratio'.format(bounds))

    os.makedirs(path_out, exist_ok=True)

    # for now, just get an image
    i = 0
    annotation_count = 0
    #rm, gm, bm, frameno =[], [], [], []

    with open('{}/{}_annotations.txt'.format(path_out, path_prefix), 'w') as outfile:
        outfile.write('{}\t{}\t{}\t{}\n'.format('Annotation', 'Frame','Time','#imgs'))

    current_contour_frame = None
    no_contour_images = []
    last_found = -1
    looking_for_annotated_frame = True
    found_in_frame = -1
    while True:
        success, image = cap.read()
        if not success:
            break
        # Crop image

        if not_suitable_image(image):
            continue

        cropped_image = image[int(bounds[1]):int(bounds[1] + bounds[3]), int(bounds[0]):int(bounds[0] + bounds[2]), :]

        # see if cropped image has annotation
        R = cropped_image[..., 2].astype(np.float)
        G = cropped_image[..., 1].astype(np.float)
        B = cropped_image[..., 0].astype(np.float)
        th_br = 20
        BR = (B - R)[np.abs(B - R) > th_br]

        current_frame_has_blue_contour = bool(np.mean(BR) > th_BR_contour) if len(BR)>50 else False

        if current_frame_has_blue_contour is True:
            if looking_for_annotated_frame is not True:
                continue
            # We have found a blue contour.
            # Here we should save the current frame, and go back in time until we find the label-less contour.
            #current_contour_frame_idx = i
            current_contour_frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
            found_in_frame, found_ms, found_img = search_for_no_annot_frame(cap, image, bounds, th_BR_contour, th_GR_contour, mode='back', max_frame=found_in_frame)
            cap.set(cv.CAP_PROP_POS_MSEC, current_contour_frame_msec)
            if found_in_frame < 0:
                found_in_frame, found_ms, found_img = search_for_no_annot_frame(cap, image, bounds, th_BR_contour, th_GR_contour, mode='forward', max_frame=found_in_frame)
            # continue where we left it
            cap.set(cv.CAP_PROP_POS_MSEC, current_contour_frame_msec+100)
            if found_in_frame < 0:
                print('Could not find annotation for frame at {}'.format(msec_to_timestamp(current_contour_frame_msec)))
            else:
                if last_found == found_ms:
                    continue
                looking_for_annotated_frame = False
                last_found = found_ms
                print('Annotation for frame at {} found at {}'.format(msec_to_timestamp(current_contour_frame_msec), msec_to_timestamp(found_ms)))
                # todo: if the found annotation is the same, then ignore
                os.makedirs('{}/annotated'.format(path_out), exist_ok=True)
                cv.imwrite('{}/annotated/{}_annotated_{}.png'.format(path_out, path_prefix, annotation_count), cv.resize(image, dsize))
                cv.imwrite('{}/{}_input_{}.png'.format(path_out, path_prefix, annotation_count), found_img)
                binary_mask, mask_size = contour_to_mask(image)
                cv.imwrite('{}/{}_label_{}.png'.format(path_out, path_prefix, annotation_count), binary_mask)
                # todo: now scroll forward until we go to a different annotation

                #with open('{}/{}_annotations.txt'.format(path_out, path_prefix), 'a+') as outfile:
                #    outfile.write('{}\t{}\t{}\t{}\n'.format(annotation_count, current_contour_frame_idx, current_contour_frame_time,len(no_contour_images)))
                annotation_count += 1
        else:
            looking_for_annotated_frame = True

        i = i+1
        #previous_cropped_image = cropped_image

        if i % int(nframes/100) == 0:
            print('{}% {} (frame {} / {})'.format(np.round(i/nframes*100), msec_to_timestamp(cap.get(cv.CAP_PROP_POS_MSEC)), i, nframes))


    # plt.subplot(3,2,1)
    # plt.plot(frameno, rm, 'r-')
    # plt.plot(frameno, gm, 'g-')
    # plt.plot(frameno, bm, 'b-')
    # plt.subplot(3, 2, 2)
    # plt.plot(frameno, rm, 'r-')
    # #plt.plot(frameno, np.abs(np.array(gm)-np.array(rm)), 'r-')
    # plt.title('|r-g|')
    # plt.subplot(3, 2, 3)
    # #plt.plot(frameno, np.abs(np.array(bm) -np.array(rm)), 'g-')
    # plt.plot(frameno, gm, 'g-')
    # plt.title('|r-b|')
    # plt.subplot(3, 2, 4)
    # #plt.plot(frameno, np.abs(np.array(bm)-np.array(gm)), 'b-')
    # plt.plot(frameno, bm, 'b-')
    # plt.title('|g-b|')
    # plt.subplot(3, 2, 5)
    # plt.plot(frameno, np.abs(np.array(gm) - np.array(rm)), 'r-')
    # plt.plot(frameno, np.abs(np.array(bm) - np.array(rm)), 'g-')
    # plt.plot(frameno, np.abs(np.array(bm) - np.array(gm)), 'b-')
    # plt.show()


if __name__ == '__main__':
    args = sys.argv[1:]
    # Calling the function
    # Input parameters
    if len(args) < 1:
        print('Usage: {} <video file name>'.format(sys.argv[0]))
        exit(-1)

    videofile_in = args[0]

    if os.path.isfile(videofile_in) == False:
        print('Video not found {}'.format(videofile_in))
        exit(-1)

    #videofile_in = '/home/ag09/data/VITAL/muscle/CNS/01NVb-003-281/T2/01NVb-003-281-2.mp4'
    #videofile_in = '/home/ag09/data/VITAL/muscle/CNS/01NVb-003-281/T2/01NVb-003-281-2 cont.mp4'
    #videofile_in = '/home/ag09/data/VITAL/muscle/CNS/01NVb-003-281/T2/small.mp4'

    # optional input parameters
    #centred around 1050, 500
    bounds = (500, 200, 1100, 650)
    min_mask_size = 50000 # minimum number of pixels for a mask, to remove false positives
    #min_mask_size = -1
    #bounds = ()

    dsize = (275, 175)
    th_GB_contour = -15 # threshold to detect blue contours
    th_BR_contour = 20
    th_GR_contour = 20
    th_d = 1 # minimum distance between consecutive images to remove input artefacts.

    ProcessVideo(videofile_in, bounds=bounds, th_GB_contour=th_GB_contour, th_BR_contour=th_BR_contour, th_GR_contour=th_GR_contour, th_d=th_d, min_mask_size=min_mask_size, dsize = dsize)
