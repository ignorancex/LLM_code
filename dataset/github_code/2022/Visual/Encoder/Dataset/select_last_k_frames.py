""" This class implements the algorithm to select the last k frames.
"""

import numpy as np
import cv2
import os

from utils.json_utils import fill_annotation


class SelectLastKFrames:
    def __init__(self):
        self.k = 5
        self.rgb_patches_list = []
        self.dpt_patches_list = []
        self.prediction_list = []
        self.selected_rgb_patches = []
        self.selected_dpt_patches = []
        # create json as a dictionary
        keys = {"annotations"}
        self.gt_json = {key: [] for key in keys}

    def update_frames(self, rgb_patches_list, dpt_patches_list, prediction_list, mask_patches_list):
        self.rgb_patches_list = rgb_patches_list
        self.dpt_patches_list = dpt_patches_list
        self.prediction_list = prediction_list
        self.mask_patches_list = mask_patches_list
        self.selected_rgb_patches = []
        self.selected_dpt_patches = []
        self.selected_predictions = []
        self.selected_mask_patches = []

    def select_frames(self):
        """ Selects the last k patches based on the minimum distance"""

        # prediction_list = [cls, score, ar_w, ar_h, avg_d]
        pred_list = self.prediction_list.copy()
        k = min(self.k, len(pred_list))
        if k == 0:
            return
        # sort distances in ascending order
        dist_array = np.asarray(pred_list)[:, 4]
        dist_array_incr = np.sort(dist_array)
        for i in range(k):
            min_val = dist_array_incr[i]
            index_min = np.where(dist_array == min_val)[0][0]
            self.selected_rgb_patches.append(self.rgb_patches_list[index_min])
            self.selected_dpt_patches.append(self.dpt_patches_list[index_min])
            self.selected_predictions.append(self.prediction_list[index_min])
            self.selected_mask_patches.append(self.mask_patches_list[index_min])
            # cv2.imshow("", np.vstack((self.rgb_patches_list[index_min][:, :, ::-1],
            #                           cv2.applyColorMap(cv2.convertScaleAbs(self.dpt_patches_list[index_min], alpha=0.03), cv2.COLORMAP_JET))))
            #
            # cv2.waitKey(0)

    def save_frames_and_annotations(self, path_to_dest, file_name, video_annotation):
        """ Saves selected frames and the corresponding annotation.

        :param path_to_dest: path to destination directory
        :param file_name: name of the frame
        :param video_annotation: annotation of the video
        :return: None
        """

        [cont_id, wt, wb, h, cap, m] = video_annotation
        k = min(self.k, len(self.prediction_list))
        path_to_dest_rgb = os.path.join(path_to_dest, "rgb")
        path_to_dest_dpt = os.path.join(path_to_dest, "depth")
        path_to_dest_masks = os.path.join(path_to_dest, "mask")
        for i in range(k):
            final_name = file_name + "_{}.png".format(i)
            # save rgb patch
            cv2.imwrite(os.path.join(path_to_dest_rgb, final_name),
                        self.selected_rgb_patches[i][:, :, ::-1])
            # save depth frames (check values)
            cv2.imwrite(os.path.join(path_to_dest_dpt, final_name),
                        self.selected_dpt_patches[i].astype(np.uint16))
            # save mask
            cv2.imwrite(os.path.join(path_to_dest_masks, final_name),
                        self.selected_mask_patches[i].astype(np.uint8))
            [_, _, ar_w, ar_h, avg_d] = self.selected_predictions[i]
            # cv2.imshow("", np.vstack((self.selected_rgb_patches[i][:, :, ::-1],
            #                           cv2.applyColorMap(cv2.convertScaleAbs(self.selected_dpt_patches[i], alpha=0.03), cv2.COLORMAP_JET))))
            #
            # cv2.waitKey(0)

            fill_annotation(self.gt_json, final_name, cont_id, ar_w, ar_h, avg_d, wt, wb, h, cap, m)
