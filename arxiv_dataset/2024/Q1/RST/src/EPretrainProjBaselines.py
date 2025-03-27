
import numpy as np

import torch
import torch.nn.functional as F

from RST.src.ELib import ELib


class EPretrainProjBaselines:

    @staticmethod
    def update_label_info(unlabeled_bundle, lbls_list, soft_lbl_list, do_softmax, temperature):
        ## I newly added "temperature", some of the previous codes did not use it, which was a bug!
        if len(lbls_list) == 1:
            for tw_ind, _ in enumerate(unlabeled_bundle.tws):
                unlabeled_bundle.input_y[0][tw_ind] = np.array(lbls_list[0][tw_ind]).tolist()
                if do_softmax:
                    unlabeled_bundle.input_y_row[0][tw_ind] = \
                        F.softmax(torch.tensor(soft_lbl_list[0][tw_ind]) / temperature).numpy().tolist()
                else:
                    unlabeled_bundle.input_y_row[0][tw_ind] = soft_lbl_list[0][tw_ind]
        elif len(lbls_list) >= 2:
            for tw_ind, _ in enumerate(unlabeled_bundle.tws):
                if do_softmax:
                    y_row = F.softmax(torch.tensor(soft_lbl_list[0][tw_ind]) / temperature).numpy()
                else:
                    y_row = np.array(soft_lbl_list[0][tw_ind])
                for ind in range(1, len(lbls_list)):
                    if do_softmax:
                        y_row += F.softmax(torch.tensor(soft_lbl_list[ind][tw_ind]) / temperature).numpy()
                    else:
                        y_row += np.array(soft_lbl_list[ind][tw_ind])
                y_row = (y_row / len(lbls_list)).tolist()
                unlabeled_bundle.input_y_row[0][tw_ind] = y_row
                unlabeled_bundle.input_y[0][tw_ind] = y_row.index(max(y_row))
            ELib.PASS()
        else:
            raise Exception('not implemented function!')
        ELib.PASS()

    @staticmethod
    def calculate_self_train_steps(train_size, unlabeled_size, ratio):
        steps = 1
        while unlabeled_size > 0:
            to_add = int(train_size * ratio)
            to_add = min(to_add, unlabeled_size)
            unlabeled_size -= to_add
            train_size += to_add
            steps += 1
        return steps

