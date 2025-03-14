from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
from skimage import io
from skimage import color
from numba import prange
import numpy as np
import cv2
import time
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}


class FaceAlignment:
    # summarize batch继续优化大概会快2.5倍，在lrs3-ted上，主要瓶颈便不在神经网络运算了，用单张图的检测即可
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)

        fan_weights = load_url(models_urls[network_name], map_location=lambda storage, loc: storage)
        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()

            depth_weights = load_url(models_urls['depth'], map_location=lambda storage, loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces)

    @torch.no_grad()
    def get_landmarks_from_video_batch(self, image_batch, detected_faces = None):
        # image shape has shape of B, H, W, C
        image_batch = image_batch.float()
        B, H, W, C = image_batch.shape
        t1 = time.time()
        if detected_faces is None:
            detect_image_batch = torch.flip(image_batch, dims = (3,))
            detect_image_batch = detect_image_batch - torch.FloatTensor([104, 117, 123])
            detect_image_batch = detect_image_batch.permute(0, 3, 1, 2)
            if self.device == 'cuda':
                detect_image_batch = detect_image_batch.cuda()
            detected_faces = self.face_detector.detect_from_batch(detect_image_batch)

        t2 = time.time()

        # select the center and max face for alignment
        max_box = []
        max_face_batch = []
        center_batch = []
        scale_batch = []

        for i, faces in enumerate(detected_faces):
            metric_list = []
            for face in faces:
                # for each face: left, top, right, bottom
                box_size = (face[2] - face[0]) * (face[3] - face[1])
                offsets = np.vstack([(face[2] + face[0]) / 2 - W / 2, (face[3] + face[1]) / 2 - H / 2])
                offset_dist_squared = np.sum(np.power(offsets, 2))
                metric_list.append(box_size - offset_dist_squared * 0.5)
            max_idx = np.argmax(metric_list)
            max_face = faces[max_idx]
            max_box.append(max_face)

            center = torch.FloatTensor(
                    [(max_face[2] + max_face[0]) / 2.0,
                     (max_face[3] + max_face[1]) / 2.0])

            center[1] = center[1] - (max_face[3] - max_face[1]) * 0.12
            scale = (max_face[2] - max_face[0] + max_face[3] - max_face[1]) / self.face_detector.reference_scale
            image = image_batch[i].cpu().numpy()
            inp = crop(image, center, scale)


            max_face_batch.append(inp)
            center_batch.append(center)
            scale_batch.append(scale)

        max_face_batch = np.array(max_face_batch)
        max_face_batch = torch.from_numpy(max_face_batch).permute(0, 3, 1, 2).float()
        max_face_batch = max_face_batch.to(self.device)
        max_face_batch.div_(255.0)
        out_batch = self.face_alignment_net(max_face_batch)[-1].detach().cpu()

        t3 = time.time()

        pts_img_batch = []
        heatmap_batch = []
        for i in prange(len(max_face_batch)):
            out = out_batch[i]
            center = center_batch[i]
            scale = scale_batch[i]
            pts, pts_img = get_preds_fromhm(out.unsqueeze(0), center, scale)
            pts, pts_img = pts.view(68, 2).numpy() * 4, pts_img.view(68, 2).numpy()
            pts_img_batch.append(pts_img)
            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for j in prange(68):
                    if pts[j, 0] > 0:
                        heatmaps[j] = draw_gaussian(
                            heatmaps[j], pts[j], 2)
                heatmap_batch.append(heatmaps)
        t4 = time.time()
        if self.landmarks_type == LandmarksType._2D:
            return pts_img_batch
        else:
            landmark_batch = []
            heatmap_batch = np.array(heatmap_batch)
            heatmap_batch = torch.from_numpy(heatmap_batch).to(self.device)
            depth_pred_batch = self.depth_prediciton_net(
                        torch.cat((max_face_batch, heatmap_batch), 1)).detach().cpu().view(B, 68, 1)

            for i in range(len(max_face_batch)):
                depth_pred = depth_pred_batch[i].numpy()
                pts_img = pts_img_batch[i]
                scale = scale_batch[i]
                landmark_batch.append(np.concatenate(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1))
            t5 = time.time()
            print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)
            return landmark_batch


    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        elif isinstance(image_or_path, torch.Tensor):
            image = image_or_path.detach().cpu().numpy()
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if detected_faces is None:
            # rgb to bgr here
            detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())
        
        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        detected_faces = [detected_faces[0]]

        landmarks = []
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp)[-1].detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())

        return landmarks

    @torch.no_grad()
    def get_landmarks_from_batch(self, image_batch, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_batch {torch.tensor} -- The input images batch

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None

        landmarks = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            landmark_set = []
            for face in faces:
                center = torch.FloatTensor(
                    [(face[2] + face[0]) / 2.0,
                     (face[3] + face[1]) / 2.0])

                center[1] = center[1] - (face[3] - face[1]) * 0.12
                scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale
                image = image_batch[i].cpu().numpy()

                image = image.transpose(1, 2, 0)

                inp = crop(image, center, scale)
                inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()

                inp = inp.to(self.device)
                inp.div_(255.0).unsqueeze_(0)

                out = self.face_alignment_net(inp)[-1].detach()
                if self.flip_input:
                    out += flip(self.face_alignment_net(flip(inp))
                                [-1].detach(), is_label=True)  # patched inp_batch undefined variable error
                out = out.cpu()
                pts, pts_img = get_preds_fromhm(out, center, scale)

                # Added 3D landmark support
                if self.landmarks_type == LandmarksType._3D:
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
                    heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                    for i in range(68):
                        if pts[i, 0] > 0:
                            heatmaps[i] = draw_gaussian(
                                heatmaps[i], pts[i], 2)
                    heatmaps = torch.from_numpy(
                        heatmaps).unsqueeze_(0)

                    heatmaps = heatmaps.to(self.device)
                    depth_pred = self.depth_prediciton_net(
                        torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                    pts_img = torch.cat(
                        (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)
                else:
                    pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)
                landmark_set.append(pts_img.numpy())

            landmark_set = np.concatenate(landmark_set, axis=0)
            landmarks.append(landmark_set)
        return landmarks

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True):
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            preds = self.get_landmarks_from_image(image, bounding_boxes)
            predictions[image_path] = preds

        return predictions

    @staticmethod
    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)
