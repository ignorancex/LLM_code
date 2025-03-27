#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import numpy as np
import cv2


def isotrop_resize(img, output_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    md = max(w, h)
    if md != output_size:
        h = h * output_size // md
        w = w * output_size // md
        interpolation = interpolation_up if output_size > md else interpolation_down
        img = cv2.resize(img, (w, h), interpolation=interpolation)

    img = img[:output_size, :output_size]
    h, w = img.shape[:2]

    if h == output_size and w == output_size:
        return img
    else:
        img0 = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        pw = (output_size - w) // 2
        ph = (output_size - h) // 2
        img0[ph:ph + h, pw: pw + w, :] = img
        return img0


class FaceExtractor:
    def __init__(self, face_size=None, square=False, return_frame=False, factor_border=3, factor_up_border=1):
        self.face_size = face_size
        self.square = square
        self.return_frame = return_frame
        self.factor_border = int(factor_border)
        self.factor_up_border = int(factor_up_border)

    def reset(self):
        return self

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            imgs = [inp['frame_bgr'], ]
            boxes = inp['boxes']
            image_inds = [0, ] * len(boxes)
            if not self.return_frame:
                del inp['frame_bgr']
        else:
            imgs = inp['frames_bgr']
            ids = inp['frames_inds']
            boxes = inp['boxes']
            image_inds = [ids.index(x) for x in inp['image_inds']]
            if not self.return_frame:
                del inp['frames_bgr']
                del inp['frames_inds']

        face_bgr = list()
        face_start = list()
        for bbox, iframe in zip(boxes, image_inds):
            frame = imgs[iframe]
            xmin, ymin, xmax, ymax = np.int64(bbox)
            w = xmax - xmin
            h = ymax - ymin
            if self.square:
                if h > w:
                    p_h = self.factor_up_border * h // self.factor_border
                    p_w = self.factor_up_border * h // self.factor_border + (h - w) // 2
                    ll = h + 2 * (self.factor_up_border * h // self.factor_border)
                else:
                    p_h = self.factor_up_border * w // self.factor_border + (w - h) // 2
                    p_w = self.factor_up_border * w // self.factor_border
                    ll = w + 2 * (self.factor_up_border * w // self.factor_border)

                xmin = xmin - p_w
                xmax = xmin + ll
                ymin = ymin - p_h
                ymax = ymin + ll

                h, w = frame.shape[:2]
                p_h = max(max(-ymin, ymax - h), 0)
                p_w = max(max(-xmin, xmax - w), 0)
                if (p_h != 0) or (p_w != 0):
                    #print('\nborder face\n')
                    if (2 * p_w) < (xmax - xmin - 5):
                        xmin = xmin + p_w
                        xmax = xmax - p_w
                    else:
                        print('\nerror face W\n')
                    if (2 * p_h) < (ymax - ymin - 5):
                        ymin = ymin + p_h
                        ymax = ymax - p_h
                    else:
                        print('\nerror face H\n')

            else:
                p_h = self.factor_up_border * h // self.factor_border
                p_w = self.factor_up_border * w // self.factor_border
                ymin = ymin - p_h
                ymax = ymax + p_h
                xmin = xmin - p_w
                xmax = xmax + p_w

            ymin = max(ymin, 0)
            xmin = max(xmin, 0)
            crop = frame[ymin:ymax, xmin:xmax]
            if self.face_size is not None:
                crop = isotrop_resize(crop, self.face_size)
            face_bgr.append(crop)
            face_start.append((xmin, ymin))

        inp['face_bgr'] = face_bgr
        inp['face_start'] = face_start
        return inp

