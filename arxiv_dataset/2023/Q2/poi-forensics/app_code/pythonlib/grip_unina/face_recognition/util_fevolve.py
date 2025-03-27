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


import torch
from fevolve.align.align_trans import get_reference_facial_points, warp_and_crop_face
from fevolve.backbone.model_irse import IR_50 as get_model
from grip_unina.util_read import BGR2RGBs


def l2_norm(inp, axis=1):
    """ l2 normalize
    """
    norm = torch.norm(inp, 2, axis, True)
    output = torch.div(inp, norm)
    return output


class ComputeFevolve:
    def __init__(self, device, weights_file, return_frame=False, face_size=(112, 112)):
        self.device = device
        self.face_size = face_size
        self.return_frame = return_frame
        self.model = get_model(self.face_size).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.reference = get_reference_facial_points(default_square=True) * max(self.face_size) / 112

    def reset(self):
        return self

    def __call__(self, inp):
        if 'frames_inds' not in inp:
            imgs = [inp['frame_bgr'], ]
            boxes = inp['points']
            image_inds = [0, ] * len(boxes)
            if not self.return_frame:
                del inp['frame_bgr']
        else:
            imgs = inp['frames_bgr']
            ids = inp['frames_inds']
            boxes = inp['points']
            image_inds = [ids.index(x) for x in inp['image_inds']]
            if not self.return_frame:
                del inp['frames_bgr']
                del inp['frames_inds']

        if len(boxes) == 0:
            inp['fevolve'] = list()
            return inp
        boxes = [[[a[2 * j], a[2 * j + 1]] for j in range(5)] for a in boxes]

        with torch.no_grad():
            # extract face
            faces = [warp_and_crop_face(imgs[i],
                                        kpt,
                                        self.reference,
                                        crop_size=self.face_size)
                     for i, kpt in zip(image_inds, boxes)]
            faces = BGR2RGBs(faces)

            # convert to torch
            faces = (torch.from_numpy(faces).permute(0, 3, 1, 2).float().to(self.device) - 128) / 128.0

            embeddings = l2_norm(self.model(faces)).cpu().numpy()
            del faces

        inp['fevolve'] = embeddings
        return inp
