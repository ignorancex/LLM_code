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


import cv2
import os
import numpy as np
from skvideo.io import FFmpegWriter
from scipy.special import expit as sigmoid


def drawBox(frame_np, box, color_box=(0, 0, 255)):
    return cv2.rectangle(frame_np, (int(box[0]), int(box[1])), (int(box[2]) + 1, int(box[3]) + 1), color_box, 10)


def drawText(frame_np, box, txt, color_txt=(255, 255, 255)):
    if int(box[1]) < 50:
        frame_np = cv2.putText(frame_np, txt, (int(box[0]) + 5, int(box[1]) + (int(box[3]) - int(box[1]))//4),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_txt, 5)
    else:
        frame_np = cv2.putText(frame_np, txt, (int(box[0]) + 3, int(box[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_txt, 5)

    return frame_np


def drawPoints(frame_np, points, color=(255, 255, 255)):
    for p in points:
        frame_np = cv2.drawMarker(frame_np, (p[0], p[1]), color,  cv2.MARKER_TILTED_CROSS, thickness=5)
    return frame_np


def drawHeat(frame_np, box, x):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    x = np.float32(x)
    img = frame_np[y1:y2, x1:x2]
    x = x + np.mean(img, -1, keepdims=True) - np.mean(x, -1, keepdims=True)
    frame_np[y1:y2, x1:x2] = np.uint8(x.clip(0, 255))

    return frame_np


class WritingVideo:
    def __init__(self, filename, fps, tag_frame='frames_out_bgr',
                 vid_configure={'-c:v': 'libx264', '-preset': 'ultrafast', '-crf': '35'}):
        self.filename = filename
        self.fps = fps
        self.tag_frame = tag_frame
        self.vid_configure = vid_configure

        self.count_frame = 0
        self.video_out = None

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

    def __enter__(self):
        self.count_frame = 0
        self.video_out = FFmpegWriter(self.filename, inputdict={'-r': str(self.fps)},
                                      outputdict=self.vid_configure, verbosity=0)
        return self

    def __call__(self, inp):
        assert self.tag_frame in inp
        if len(self.tag_frame)==0:
            return inp

        for index_f in np.argsort(inp['frames_inds']):
            index = inp['frames_inds'][index_f]
            frame = cv2.cvtColor(inp[self.tag_frame][index_f], cv2.COLOR_BGR2RGB)
            while self.count_frame <= index:
                self.video_out.writeFrame(frame)
                self.count_frame = self.count_frame + 1

        return inp

    def __exit__(self, type, value, tb):
        self.count_frame = 0
        try:
            self.video_out.close()
        except:
            pass
        self.video_out = None


class GenFrameBoxes:
    def __init__(self, tag_boxes='boxes', return_frame=False):
        self.tag_boxes = tag_boxes
        self.return_frame = return_frame
        self.color_loop = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def reset(self):
        return self

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if 'image_inds' in inp:
            image_inds = inp['image_inds']
            boxes = inp[self.tag_boxes]
            tracks = inp['id_track']
        else:
            image_inds = list()
            boxes = list()
            tracks = list()
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)
            for index in range(len(image_inds)):
                if image_inds[index] == index_f:
                    frame = drawBox(frame, boxes[index], self.color_loop[tracks[index] % len(self.color_loop)])
                    frame = drawText(frame, boxes[index], '%d' % tracks[index], color_txt=(255, 255, 255))

            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class GenFrameHeat:
    def __init__(self, cmap, fit_lim=sigmoid, tag_boxes='face_boxes', return_frame=False):
        self.tag_boxes = tag_boxes
        self.return_frame = return_frame
        self.cmap_bgr = 255*cmap[:, ::-1]
        self.fit_lim = fit_lim

    def reset(self):
        return self

    def fit_size(self, box, x):
        h = int(box[3]) - int(box[1])
        w = int(box[2]) - int(box[0])
        if h > w:
            wm = int(round(h * x.shape[1] / x.shape[0]))
            x = cv2.resize(x, (wm, h), interpolation=cv2.INTER_LINEAR)
        else:
            hm = int(round(w * x.shape[0] / x.shape[1]))
            x = cv2.resize(x, (w, hm), interpolation=cv2.INTER_LINEAR)

        s_w = (x.shape[1] - w) // 2
        s_h = (x.shape[0] - h) // 2
        x = x[s_h:(s_h+h), s_w:(s_w+w)]
        return x

    def apply_cmap(self, x_bgr):
        x_bgr = np.int64(np.round(x_bgr * len(self.cmap_bgr)).clip(0, len(self.cmap_bgr)-1))
        return self.cmap_bgr[x_bgr, :]

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if 'image_inds' in inp:
            image_inds = inp['image_inds']
            boxes = inp[self.tag_boxes]
            mappreds = inp['mappreds']
        else:
            image_inds = list()
            boxes = list()
            mappreds = list()
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)
            for index in range(len(image_inds)):
                if image_inds[index] == index_f:
                    x = self.apply_cmap(self.fit_lim(self.fit_size(boxes[index], mappreds[index])))
                    frame = drawHeat(frame, boxes[index], x)

            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class OutputBoxes:
    def __init__(self, outputfile, margin=25, return_frame=False, color=(180, 180, 180)):
        dat = np.load(outputfile)
        self.boxes = dat['embs_boxes']
        self.ranges = np.asarray([[_[0] + margin, _[1] - margin] for _ in dat['embs_range'].astype(np.int64)])
        if 'embs_points' in dat:
            self.points = np.asarray([_[margin: len(_)-margin] for _ in dat['embs_points']])
        else:
            self.points = None
        self.dists = dat['embs_dists']
        self.id_track = dat['embs_track']
        while len(self.dists.shape) > 1:
            self.dists = self.dists[..., -1]
        del dat
        self.return_frame = return_frame
        self.color = color

    def reset(self):
        return self

    def __call__(self, inp):
        assert 'frames_bgr' in inp
        imgs = inp['frames_bgr']
        ids = inp['frames_inds']
        if not self.return_frame:
            del inp['frames_bgr']

        list_frames_out_bgr = list()
        for index_f, frame in zip(ids, imgs):
            frame = np.copy(frame)

            sel = (self.ranges[:, 0] <= index_f) & (index_f < self.ranges[:, 1])
            boxf = self.boxes[sel]
            tracks = self.id_track[sel]
            distsf = self.dists[sel]
            rp = self.ranges[sel]
            if self.points is not None:
                pp = self.points[sel]
            else:
                pp = None

            for index in range(len(boxf)):
                if np.isnan(boxf[index][0]):
                    frame = drawText(frame, (0, 50, 50, 100), '%.3f' % distsf[index], color_txt=self.color)
                else:
                    frame = drawBox(frame, boxf[index], self.color)
                    frame = drawText(frame, boxf[index], '%.3f' % distsf[index], color_txt=self.color)
                if pp is not None:
                    try:
                        xy = pp[index][index_f-rp[index][0]].reshape(-1, 2)
                        frame = drawPoints(frame, xy, self.color)
                    except:
                        pass


            list_frames_out_bgr.append(frame)

        inp['frames_out_bgr'] = list_frames_out_bgr
        return inp


class WritingClips:
    def __init__(self, filedir, write_one=False):
        self.list_track = dict()
        self.write_one = write_one
        self.filedir = filedir
        os.makedirs(self.filedir, exist_ok=True)

    def __enter__(self):
        self.list_track = dict()
        return self

    def __call__(self, inp):
        assert 'boxes' in inp
        assert 'image_inds' in inp
        assert 'id_track' in inp
        assert 'points' in inp
        assert 'face_bgr' in inp
        assert 'face_start' in inp
        flag_landmarks = 'landmarks68' in inp

        for index_f in np.argsort(inp['image_inds']):
            index = inp['image_inds'][index_f]
            track = inp['id_track'][index_f]
            start = inp['face_start'][index_f]
            point = [[inp['points'][index_f][2 * j] - start[0],
                      inp['points'][index_f][2 * j + 1] - start[1]] for j in range(5)]
            if flag_landmarks:
                lm68 = np.asarray(inp['landmarks68'][index_f]) - [start, ]
            else:
                lm68 = None
            box = inp['boxes'][index_f]
            box = [box[0]-start[0], box[1]-start[1],
                   box[2]-start[0], box[3]-start[1]]

            if track not in self.list_track:
                self.list_track[track] = index

            count = index - self.list_track[track]
            if self.write_one:
                if count == 0:
                    filepng = os.path.join(self.filedir, 'track_%d.png' % track)
                    cv2.imwrite(filepng, inp['face_bgr'][index_f])
                continue

            filepng = os.path.join(self.filedir, 'crop_%d_%d.png' % (track, count))
            filenpz = os.path.join(self.filedir, 'crop_%d_%d.png.npz' % (track, count))

            cv2.imwrite(filepng, inp['face_bgr'][index_f])

            data_storage = {"ldm": (box, point, lm68) if lm68 else (box, point),
                            "idx": index,
                            "box": start}
            np.savez(filenpz, **data_storage)

        return inp

    def __exit__(self, type, value, tb):
        self.list_track = dict()
