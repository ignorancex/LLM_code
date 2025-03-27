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
import torch
import cv2
from skimage.transform import resize


def get_crop_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.21  # 0.14
    #size = int(np.round(old_size * 1.975))
    size = int(np.round(old_size * 1.56))
    #size = int(old_size * 1.58) * 5 // 4

    roi_box = [0, 0, 0, 0]
    roi_box[0] = int(np.round(center_x - (size / 2)))
    roi_box[1] = int(np.round(center_y - (size / 2)))
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def extract_clip(list_frames, list_start, list_boxes, output_size=224, stride=1, from_bgr=False, return_box=True):
    box = get_crop_from_bbox(list_boxes[len(list_boxes)//2])
    out_size = (box[2] - box[0], box[3] - box[1])
    assert out_size[0] == out_size[1]

    frames = list()
    for index in range(0, len(list_frames), stride):
        start = list_start[index]
        box1 = [box[0]-start[0], box[1]-start[1], box[2]-start[0], box[3]-start[1]]
        frame = list_frames[index][max(box1[1], 0):box1[3],
                                   max(box1[0], 0):box1[2], :]

        if frame.size>0:
            if from_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = np.pad(frame, ((max(-box1[1], 0), 0),
                                   (max(-box1[0], 0), 0),
                                   (0, 0)), mode='edge')
            frame = np.pad(frame, ((0, out_size[1] - frame.shape[0]),
                                  (0, out_size[0] - frame.shape[1]),
                                  (0, 0)), mode='edge')
        else:
            frame = np.zeros((1,1,3), dtype=frame.dtype)

        frame = np.uint8(resize(frame, (output_size, output_size), order=1, mode='reflect',
                                preserve_range=True, anti_aliasing=True))
        frames.append(frame)

    if return_box:
        return np.stack(frames, 0), box
    else:
        return np.stack(frames, 0)


class AlignAudio:

    def input_keys(self):
        return ['spec', ]

    def output_keys(self):
        return ['spec', 'boxes']

    def __call__(self, spec):
        spec = np.concatenate(spec, 0).transpose((1, 0))
        return {'spec': spec, 'boxes': [np.nan, np.nan, np.nan, np.nan]}


class AlignFaces:

    def __init__(self, image_size=224, video_stride=1, return_points=True):
        self.image_size = image_size
        self.video_stride = video_stride
        self.return_points = return_points

    def input_keys(self):
        if self.return_points:
            return ['face_bgr', 'face_start', 'boxes', 'points']
        else:
            return ['face_bgr', 'face_start', 'boxes', ]

    def output_keys(self):
        if self.return_points:
            return ['face3_rgb', 'boxes', 'points']
        else:
            return ['face3_rgb', 'boxes']

    def __call__(self, **keys):
        out_dict = dict()
        out_dict['face3_rgb'], out_dict['boxes'] = extract_clip(
            keys['face_bgr'], keys['face_start'], keys['boxes'],
            stride=self.video_stride, from_bgr=True, return_box=True)
        if self.return_points:
            out_dict['points'] = np.stack(keys['points'], 0)
        return out_dict


class AlignFacesAudio:

    def __init__(self, image_size=224, video_stride=1, return_points=True):
        self.image_size = image_size
        self.video_stride = video_stride
        self.return_points = return_points

    def input_keys(self):
        if self.return_points:
            return ['face_bgr', 'face_start', 'boxes', 'spec', 'points']
        else:
            return ['face_bgr', 'face_start', 'boxes', 'spec']

    def output_keys(self):
        if self.return_points:
            return ['face3_rgb', 'spec', 'boxes', 'points']
        else:
            return ['face3_rgb', 'spec', 'boxes']

    def __call__(self, **keys):
        out_dict = dict()
        out_dict['face3_rgb'], out_dict['boxes'] = extract_clip(
            keys['face_bgr'], keys['face_start'], keys['boxes'],
            stride=self.video_stride, from_bgr=True, return_box=True)
        out_dict['spec'] = np.concatenate(keys['spec'], 0).transpose((1, 0))
        if self.return_points:
            out_dict['points'] = np.stack(keys['points'], 0)
        return out_dict


class ApplyModel3d:
    def __init__(self, device, model, half=False, transform=None, batch_size=8,
                 input_key='embs_face3_bgr', output_key='embs_feat', remove_keys=None):
        self.device = device
        self.transform = transform
        self.half = half
        self.batch_size = batch_size
        self.input_key = input_key
        self.output_key = output_key
        if remove_keys is None:
            self.remove_keys = [self.input_key, ]
        else:
            self.remove_keys = remove_keys
        if self.half:
            self.model = model.half().to(self.device)
        else:
            self.model = model.float().to(self.device)

        self.model.eval()

    def reset(self):
        return self

    def __call__(self, inp):
        if len(inp[self.input_key]) == 0:
            inp[self.output_key] = list()
            return inp

        with torch.no_grad():
            x = inp[self.input_key]
            if self.transform is not None:
                x = [self.transform(_) for _ in x]

            x = torch.stack(x, 0)

            if self.half:
                x = x.half()

            preds = torch.cat([self.model(item.to(self.device)).cpu()
                               for item in torch.split(x, self.batch_size, dim=0)], 0).numpy()

            if self.remove_keys is not None:
                for _ in self.remove_keys:
                    del inp[_]
            inp[self.output_key] = preds

        return inp
