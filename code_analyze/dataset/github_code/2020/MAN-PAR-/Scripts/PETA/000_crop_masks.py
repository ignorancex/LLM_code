import cv2
import os
import numpy as np

def resizeAndPad(img, size, padColor=0):

    h, w, channel = img.shape
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    if h==0:
        print("height of the image is =0")
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    #cv2.imshow("_cropped_img", scaled_img)
    #cv2.waitKey(0)
    return scaled_img

# input dirs
Peta_imgs_with_border = "/home/eshan/PycharmProjects/PRLetter_PETA/DATA/Bordered_Images/"
List_of_images = os.listdir(Peta_imgs_with_border)

Peta_mask_dir =  "/home/eshan/PycharmProjects/PRLetter_PETA/DATA/Masks/"
List_of_PETA_masks = os.listdir(Peta_mask_dir)

# Output dirs
IMG_dir= "/home/eshan/PycharmProjects/PRLetter_PETA/DATA/resized_imgs/"
MASK_dir = "/home/eshan/PycharmProjects/PRLetter_PETA/DATA/resized_masks/"

aspect_ratio = []

for i, name in enumerate(List_of_PETA_masks):
    if os.path.exists(IMG_dir+name):
        print("the resized IMAGE is already existed:   {}/{}".format(i,len(List_of_PETA_masks)))
        continue
    else:
        massk = cv2.imread(Peta_mask_dir + name)
        image = cv2.imread(Peta_imgs_with_border + name)
        h,w,d = massk.shape
        crop_mask= massk[200:h - 200, 237:w - 237]
        crop_img = image[200:h - 200, 237:w - 237]

        resized_mask = resizeAndPad(crop_mask, (32,32), padColor=0)
        resized_img = resizeAndPad(crop_img, (500,500), padColor=0)

        cv2.imwrite(filename=MASK_dir+name,img=resized_mask)
        cv2.imwrite(filename=IMG_dir+name, img=resized_img)
        print(resized_img.shape,"  ", resized_mask.shape)
        #cv2.imshow("cropped_image",resized_mask)
        #cv2.imshow("resized_cropped_image", resized_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if i % 500 == 0:
            print("preparing {}/{}".format(i,len(List_of_PETA_masks)))
print("FINISHED, Go to the next step")
