import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
from torch.optim import lr_scheduler
import gc
import torch.nn.functional as F
from spynet.Spy_net import ME_Spynet, flow_warp
# from models.backbone import QT_Net_HLG, MTT_Net_HLG_base, MTT_Dire_HLG_base
from models.backbone import QT_Net_HLG, MTT_mask_net, MTT_Net_HLG
import re
import options.options as option
import random
import h5py
from einops import rearrange

work_on_999 = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # torch.backends.cudnn.deterministic = True


# set random seed as 100
setup_seed(100)


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)

        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def get_order(file_path):
    poc_list = []
    reorder_list = []
    refer_id_f = {}
    refer_id_b = {}
    qp_dict = {}
    tid_dict = {}
    with open(file_path, 'r') as f:
        content = f.readlines()
    for line in content:
        if line[:3] == 'POC':
            pattern = r'POC\s+(\d+)'
            match = re.search(pattern, line)
            poc = int(match.group(1))
            # qp
            qp_pattern = r"QP (\d+)"
            qp_match = re.search(qp_pattern, line)
            qp = int(qp_match.group(1))
            qp_dict[poc] = qp
            # tid
            tid_pattern = r'TId:\s*(\d+)'
            tid_match = re.search(tid_pattern, line)
            tid = int(tid_match.group(1))
            tid_dict[poc] = tid

            pattern = r'\[L0 ([\d\sc]+)\] \[L1 ([\d\sc]+)\]'
            matches = re.search(pattern, line)
            if matches:
                L0 = matches.group(1)
                L1 = matches.group(2)
                L0 = [int(item[:-1]) if item[-1] == "c" else int(item) for item in L0.split()]
                L1 = [int(item[:-1]) if item[-1] == "c" else int(item) for item in L1.split()]
                refer_id_f[poc] = L0
                refer_id_b[poc] = L1
            else:
                refer_id_b[poc] = 0
                refer_id_f[poc] = 0
    for i in range(len(poc_list)):
        reorder_list.append(poc_list.index(i))
    return refer_id_f, refer_id_b, reorder_list, qp_dict, tid_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def RA_recurrent(cur_id, a=0, b=32):
    if cur_id == a + (b - a) // 2:
        return a, b
    elif cur_id < a + (b - a) // 2:
        return RA_recurrent(cur_id, a=a, b=a + (b - a) // 2)
    elif cur_id > a + (b - a) // 2:
        return RA_recurrent(cur_id, a=a + (b - a) // 2, b=b)


def get_cand_id_list(cur_id, mode='LDP', gop_size=8, ref_len=4, frm_num=100):
    """get candidate reference frame id list"""
    if mode == 'LDP' or mode == 'LDB':
        cand_id_list = set([max(0, (cur_id // gop_size - i) * gop_size) for i in range(4)])
        tmp_id = max(0, cur_id - 1)
        while len(cand_id_list) < 4:
            cand_id_list.add(tmp_id)
            tmp_id -= 1
            if tmp_id <= 0:
                break
        cand_id_list = list(cand_id_list)
        cand_id_list.sort(reverse=True)
        while len(cand_id_list) < 4:
            cand_id_list.append(cand_id_list[-1])
    elif mode == 'RA':
        base_id, remainder_id = cur_id // 32, cur_id % 32
        if base_id * 32 + 32 >= frm_num:
            f0_id, f1_id = RA_recurrent(cur_id=remainder_id, a=0, b=min(32, frm_num - base_id * 32))
        else:
            f0_id, f1_id = RA_recurrent(cur_id=remainder_id, a=0, b=32)
        cand_id_list = [base_id * 32 + f0_id, base_id * 32 + f0_id if (base_id * 32 + f1_id) >= frm_num else base_id * 32 + f1_id]  # 前向参考列表和后向参考列表的第一个frame id
    return cand_id_list[:ref_len]


class ValidDataset(Dataset):
    def __init__(self, qp, sub_id, mode='train', dataset_dir="G:\\dataset", enc_mode='LDP', split_id=None):
        super().__init__()
        self.qp = qp
        self.dataset_dir = dataset_dir
        assert mode in ['train', 'valid', 'test']
        assert enc_mode in ['LDP', 'LDB', 'RA']
        self.mode = mode + '_dataset'
        self.enc_mode = enc_mode
        self.sub_id = sub_id

        if mode == 'train':
            if work_on_999:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/ghome/fengxm/VVC_LAST/log/qp%d_frame64.log" % self.qp)
            else:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/code/log/qp%d_frame64.log" % self.qp)
        elif mode == 'test' or mode == 'valid':
            if work_on_999:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/ghome/fengxm/VVC_LAST/log/qp%d_frame100.log" % self.qp)
            else:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/code/log/qp%d_frame100.log" % self.qp)

        # load label
        self.__load_label__(sub_id)  # depth -> cls_id
        # load y
        self.__load_dataset_y__(sub_id)  # 加载8bit数据
        print("have loaded dataset y")
        if resample_dataset and self.mode == 'train_dataset':
            # 设计一个重复采样少样本帧的方法
            self.sampled_indices = self.__generate_sampled_indices__()

    # 增加样本数量的resample
    def __generate_sampled_tid_indices__(self, tid):
        total_frames = self.y_content.shape[0] // 64 * 62

        sampled_indices = []
        for i in range(total_frames):
            if self.__get_tid__(i % 62) == tid:
                sampled_indices.extend([i])

        return sampled_indices

    def __get_tid__(self, p_idx):
        if self.enc_mode == 'RA':
            abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
        else:
            abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index
        tid = self.tid_dict[abs_idx]
        return tid

    def __generate_sampled_indices__(self):
        # 保证每种TID的训练机会相等
        total_frames = self.y_content.shape[0] // 64 * 62
        frame_sampling_rates = []

        sampled_indices = []
        for i in range(total_frames):
            sampling_rate = self.__generate_sample_rate__(i % 62)
            sampled_indices.extend([i] * sampling_rate)

        return sampled_indices

    def __generate_sample_rate__(self, p_idx):
        if self.enc_mode == 'RA':
            abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
        else:
            abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index
        tid = self.tid_dict[abs_idx]
        return 16 // 2 ** (tid - 1)

    def __load_dataset_y__(self, sub_id, is_cropped=True):
        # 是否选择cropped sequence作为输入
        if is_cropped and sub_id in ['B', 'C', 'D'] and self.mode == 'train_dataset':
            sub_id += '_cropped'
        self.y_content = np.load(os.path.join(self.dataset_dir, self.mode, 'Input_y', '%s.npy' % sub_id))
        if opt['train']['load_uv']:
            self.uv_content = np.load(os.path.join(self.dataset_dir, self.mode, 'Input_uv', '%s.npy' % sub_id))

    def __load_label__(self, sub_id):
        self.qt_label = np.load(os.path.join(self.dataset_dir, self.mode, self.enc_mode, 'QP%d' % self.qp, 'qt_label', '%s.npy' % sub_id))  # (seq_num, p_id, h, w)
        self.mtt_label = np.load(os.path.join(self.dataset_dir, self.mode, self.enc_mode, 'QP%d' % self.qp, 'mtt_label', '%s.npy' % sub_id))  # (seq_num, p_id, 3, 2, h, w)

    def __random_crop_y__(self, single_frame, crop_x, crop_y, block_width=None, block_height=None):
        if block_width is None:
            block_width = self.block_size
        if block_height is None:
            block_height = self.block_size
        # Calculate Y's coordinates
        left_top_x = crop_x
        left_top_y = crop_y
        right_bottom_x = left_top_x + 128 * block_width
        right_bottom_y = left_top_y + 128 * block_height

        # Ensure Y does not exceed image boundaries
        right_bottom_x = min(right_bottom_x, single_frame.shape[2])
        right_bottom_y = min(right_bottom_y, single_frame.shape[1])

        # Crop Y
        Y = single_frame[:, left_top_y:right_bottom_y, left_top_x:right_bottom_x]

        # If Y is smaller than 128*block_size x 128*block_size, pad it with zeros
        if Y.shape[1] < 128 * block_height or Y.shape[2] < 128 * block_width:
            padding_h = 128 * block_height - Y.shape[1]
            padding_w = 128 * block_width - Y.shape[2]
            Y = np.pad(Y, ((0, 0), (0, padding_h), (0, padding_w)), mode='constant')
        return Y

    def __random_crop_label__(self, single_label, crop_x, crop_y, ratio=4, isqt=False, gt_block_width=None, gt_block_height=None, boundary_len=None):
        # ratio=4 for mtt label, ratio=16 for qt label
        # Crop qt_label and mtt_label based on the same cropping coordinates as X
        if isqt:
            label_crop = single_label[(crop_y + 128 * boundary_len) // ratio:(crop_y + 128 * boundary_len + 128 * gt_block_height) // ratio, (crop_x + 128 * boundary_len) // ratio:(crop_x + 128 * boundary_len + 128 * gt_block_width) // ratio]
        else:
            label_crop = single_label[:, (crop_y + 128 * boundary_len) // ratio:(crop_y + 128 * boundary_len + 128 * gt_block_height) // ratio, (crop_x + 128 * boundary_len) // ratio:(crop_x + 128 * boundary_len + 128 * gt_block_width) // ratio]
        return label_crop

    # 随机裁剪
    def __getitem__(self, index):
        # input: p_frm id
        if resample_dataset and self.mode == "train_dataset":
            index = self.sampled_indices[index]  # 重复利用占比较低的TID帧 or 仅采样特定tid的帧
        if self.mode == "train_dataset":
            seq_idx, p_idx = index // 62, index % 62
            if self.enc_mode == 'RA':
                abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
            else:
                abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index

            cand_frmid = get_cand_id_list(abs_idx, mode='RA', gop_size=32, ref_len=2, frm_num=64)
            qp_list = [self.qp_dict[abs_idx], self.qp_dict[cand_frmid[0]], self.qp_dict[cand_frmid[1]]]
            tid_list = [self.tid_dict[abs_idx], self.tid_dict[cand_frmid[0]], self.tid_dict[cand_frmid[1]]]
            if opt['train']['load_uv']:
                # input_content shape, (3,h,w)
                i_frame = np.concatenate([self.y_content[None, seq_idx * 64 + abs_idx], self.uv_content[seq_idx, abs_idx].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
                p0_frame = np.concatenate([self.y_content[None, seq_idx * 64 + cand_frmid[0]], self.uv_content[seq_idx, cand_frmid[0]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
                p1_frame = np.concatenate([self.y_content[None, seq_idx * 64 + cand_frmid[1]], self.uv_content[seq_idx, cand_frmid[1]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
            else:
                i_frame = self.y_content[None, seq_idx * 64 + abs_idx]
                p0_frame = self.y_content[None, seq_idx * 64 + cand_frmid[0]]
                p1_frame = self.y_content[None, seq_idx * 64 + cand_frmid[1]]

            label_0 = self.qt_label[seq_idx, p_idx]
            label_1 = self.mtt_label[seq_idx, p_idx, 0]
            label_2 = self.mtt_label[seq_idx, p_idx, 1]
            label_3 = self.mtt_label[seq_idx, p_idx, 2]

            if 'A' in self.sub_id:
                # 将A类序列crop成B类分辨率，取1/4 region，以及1-CTU邻域，
                neibo_len = 1
                block_width = i_frame.shape[2] // (2 * 128) + 2 * neibo_len
                block_height = i_frame.shape[1] // (2 * 128) + 2 * neibo_len
                max_x = i_frame.shape[2] - block_width * 128
                max_y = i_frame.shape[1] - block_height * 128

                # Randomly choose a left-top corner that satisfies the requirements
                crop_x = np.random.randint(0, max_x + 1) // 128 * 128
                crop_y = np.random.randint(0, max_y + 1) // 128 * 128

                i_frame = self.__random_crop_y__(i_frame, crop_x, crop_y, block_width, block_height)
                p0_frame = self.__random_crop_y__(p0_frame, crop_x, crop_y, block_width, block_height)
                p1_frame = self.__random_crop_y__(p1_frame, crop_x, crop_y, block_width, block_height)

                label_0 = self.__random_crop_label__(label_0, crop_x, crop_y, ratio=16, isqt=True, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_1 = self.__random_crop_label__(label_1, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_2 = self.__random_crop_label__(label_2, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_3 = self.__random_crop_label__(label_3, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)

            output = {'i_frame': i_frame, 'p0_frame': p0_frame, 'p1_frame': p1_frame, 'label_0': torch.from_numpy(label_0), "label_1": torch.from_numpy(label_1), "label_2": torch.from_numpy(label_2), "label_3": torch.from_numpy(label_3), "qp_list": torch.tensor(qp_list),
                      "tid_list": torch.tensor(tid_list)}  # qt_label, mtt_label_0, mtt_label_1, mtt_label_2

        elif self.mode == "valid_dataset":
            p_idx = index
            if self.enc_mode == 'RA':
                abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1
            else:
                abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1
            cand_frmid = get_cand_id_list(abs_idx, mode='RA', gop_size=32, ref_len=2, frm_num=100)
            qp_list = [self.qp_dict[abs_idx], self.qp_dict[cand_frmid[0]], self.qp_dict[cand_frmid[1]]]
            tid_list = [self.tid_dict[abs_idx], self.tid_dict[cand_frmid[0]], self.tid_dict[cand_frmid[1]]]
            if opt['train']['load_uv']:
                i_frame = np.concatenate([self.y_content[None, abs_idx], self.uv_content[abs_idx].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
                p0_frame = np.concatenate([self.y_content[None, cand_frmid[0]], self.uv_content[cand_frmid[0]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
                p1_frame = np.concatenate([self.y_content[None, cand_frmid[1]], self.uv_content[cand_frmid[1]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
            else:
                i_frame = self.y_content[None, abs_idx]
                p0_frame = self.y_content[None, cand_frmid[0]]
                p1_frame = self.y_content[None, cand_frmid[1]]
            output = {'i_frame': i_frame, 'p0_frame': p0_frame, 'p1_frame': p1_frame, 'label_0': torch.from_numpy(self.qt_label[p_idx]), "label_1": torch.from_numpy(self.mtt_label[p_idx, 0]), "label_2": torch.from_numpy(self.mtt_label[p_idx, 1]), "label_3": torch.from_numpy(self.mtt_label[p_idx, 2]),
                      "qp_list": torch.tensor(qp_list), "tid_list": torch.tensor(tid_list)}

        return output

    def __len__(self):
        if resample_dataset and self.mode == 'train_dataset':
            return len(self.sampled_indices)  # return self.y_content.shape[0] // 64 * 62 // 31 * 80
        else:
            return self.y_content.shape[0] // 64 * 62


# resample 增加样本数量
# hdf5格式的训练数据集
class MyDataset(Dataset):
    def __init__(self, qp, sub_id, mode='train', dataset_dir="G:\\dataset", enc_mode='LDP', split_id=None):
        super().__init__()
        self.qp = qp
        self.dataset_dir = dataset_dir
        assert mode in ['train', 'valid', 'test']
        assert enc_mode in ['LDP', 'LDB', 'RA']
        self.mode = mode + '_dataset'
        self.enc_mode = enc_mode
        self.sub_id = sub_id

        if mode == 'train':
            if work_on_999:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/ghome/fengxm/VVC_LAST/log/qp%d_frame32.log" % self.qp)
            else:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/code/log/qp%d_frame32.log" % self.qp)
        elif mode == 'test' or mode == 'valid':
            if work_on_999:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/ghome/fengxm/VVC_LAST/log/qp%d_frame100.log" % self.qp)
            else:
                self.refer_id_f, self.refer_id_b, _, self.qp_dict, self.tid_dict = get_order("/code/log/qp%d_frame100.log" % self.qp)

        self.label = h5py.File(os.path.join(self.dataset_dir, self.mode, 'vtm10', 'train_qp%d.h5' % self.qp), 'r')
        self.y_content = h5py.File(os.path.join(self.dataset_dir, self.mode, 'vtm10', 'train_seqs.h5'), 'r')
        self.sub_id = sub_id

        self.seq_len = 0
        while True:
            try:
                self.y_content['%s/%d' % (self.sub_id, self.seq_len)]
                self.seq_len += 1
            except:
                break

        if resample_dataset and self.mode == 'train_dataset':
            # 设计一个重复采样少样本帧的方法
            if opt['train']['paired']:
                self.sampled_indices = self.__generate_sampled_indices_paired__()
            else:
                self.sampled_indices = self.__generate_sampled_indices__()
            print('length of the dataset: ', self.__len__())

    # 增加样本数量的resample
    def __generate_sampled_tid_indices__(self, tid):
        total_frames = self.y_content.shape[0] // 64 * 62

        sampled_indices = []
        for i in range(total_frames):
            if self.__get_tid__(i % 62) == tid:
                sampled_indices.extend([i])

        return sampled_indices

    # 另外一种resample策略，采样得到tid=x的帧时，会随机选择一张其他tid的帧，并在训练中，关闭shuffle，提前随机shuffle
    def __generate_sampled_indices_paired__(self):
        # 保证每种TID的训练机会相等
        total_frames = self.seq_len * 31

        sampled_indices = []
        for i in range(total_frames):
            sampling_rate = self.__generate_sample_rate__(i % 31)
            for j in range(sampling_rate):
                sampled_indices.append([i, max(min(i + random.randint(-3, 3), total_frames - 1), 0)])
                if opt['train']['repeat_dataset']:
                    sampled_indices.append([i, max(min(i + random.randint(-3, 3), total_frames - 1), 0)])
        random.shuffle(sampled_indices)
        sampled_indices = [item for sublist in sampled_indices for item in sublist]

        return sampled_indices

    def __get_tid__(self, p_idx):
        if self.enc_mode == 'RA':
            abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
        else:
            abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index
        tid = self.tid_dict[abs_idx]
        return tid

    def __generate_sampled_indices__(self):
        # 保证每种TID的训练机会相等
        total_frames = self.seq_len * 31

        sampled_indices = []
        for i in range(total_frames):
            sampling_rate = self.__generate_sample_rate__(i % 31)
            # if not opt['train']['crop_A']:
            #     # 当各个分辨率的batchsize不同，为了保证训练的均匀，会根据分辨率类别对训练样本进行复制
            #     if self.sub_id == 'B':
            #         sampling_rate *= 4
            #     elif self.sub_id == 'C':
            #         sampling_rate *= 16
            #     elif self.sub_id == 'D':
            #         sampling_rate *= 64
            sampled_indices.extend([i] * sampling_rate)

        return sampled_indices

    def __generate_sample_rate__(self, p_idx):
        if self.enc_mode == 'RA':
            abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
        else:
            abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index
        tid = self.tid_dict[abs_idx]
        return 16 // 2 ** (tid - 1)

    def __random_crop_y__(self, single_frame, crop_x, crop_y, block_width=None, block_height=None):
        if block_width is None:
            block_width = self.block_size
        if block_height is None:
            block_height = self.block_size
        # Calculate Y's coordinates
        left_top_x = crop_x
        left_top_y = crop_y
        right_bottom_x = left_top_x + 128 * block_width
        right_bottom_y = left_top_y + 128 * block_height

        # Ensure Y does not exceed image boundaries
        right_bottom_x = min(right_bottom_x, single_frame.shape[-1])
        right_bottom_y = min(right_bottom_y, single_frame.shape[-2])

        # Crop Y
        Y = single_frame[left_top_y:right_bottom_y, left_top_x:right_bottom_x]

        # If Y is smaller than 128*block_size x 128*block_size, pad it with zeros
        if Y.shape[-2] < 128 * block_height or Y.shape[-1] < 128 * block_width:
            padding_h = 128 * block_height - Y.shape[-2]
            padding_w = 128 * block_width - Y.shape[-1]
            Y = np.pad(Y, ((0, 0), (0, padding_h), (0, padding_w)), mode='constant')
        return Y

    def __random_crop_label__(self, single_label, crop_x, crop_y, ratio=4, isqt=False, gt_block_width=None, gt_block_height=None, boundary_len=None):
        # ratio=4 for mtt label, ratio=16 for qt label
        # Crop qt_label and mtt_label based on the same cropping coordinates as X
        if isqt:
            label_crop = single_label[(crop_y + 128 * boundary_len) // ratio:(crop_y + 128 * boundary_len + 128 * gt_block_height) // ratio, (crop_x + 128 * boundary_len) // ratio:(crop_x + 128 * boundary_len + 128 * gt_block_width) // ratio]
        else:
            label_crop = single_label[:, (crop_y + 128 * boundary_len) // ratio:(crop_y + 128 * boundary_len + 128 * gt_block_height) // ratio, (crop_x + 128 * boundary_len) // ratio:(crop_x + 128 * boundary_len + 128 * gt_block_width) // ratio]
        return label_crop

    def random_flip(self, i_frame, p0_frame, p1_frame, label_0, label_1, label_2, label_3):
        # 确定翻转模式：0表示上下镜像，1表示左右镜像，2表示不翻转
        # flip_mode = np.random.randint(0, 3)

        # if flip_mode == 0:  # 上下镜像
        #     return np.flip(i_frame, axis=0), np.flip(p0_frame, axis=0), np.flip(p1_frame, axis=0), \
        #         np.flip(label_0, axis=0), np.flip(label_1, axis=1), np.flip(label_2, axis=1), np.flip(label_3, axis=1)
        # elif flip_mode == 1:  # 左右镜像
        #     return np.flip(i_frame, axis=1), np.flip(p0_frame, axis=1), np.flip(p1_frame, axis=1), \
        #         np.flip(label_0, axis=1), np.flip(label_1, axis=2), np.flip(label_2, axis=2), np.flip(label_3, axis=2)
        # else:  # 不翻转
        #     return i_frame, p0_frame, p1_frame, label_0, label_1, label_2, label_3

        # 随机选择一个镜像翻转
        # 确定翻转模式：0表示上下镜像，1表示左右镜像
        flip_mode = torch.randint(0, 3, (1,)).item()

        if flip_mode == 0:  # 上下镜像
            return torch.flip(i_frame, dims=[0]), torch.flip(p0_frame, dims=[0]), torch.flip(p1_frame, dims=[0]), torch.flip(label_0, dims=[0]), torch.flip(label_1, dims=[1]), torch.flip(label_2, dims=[1]), torch.flip(label_3, dims=[1])
        elif flip_mode == 1:  # 左右镜像
            return torch.flip(i_frame, dims=[1]), torch.flip(p0_frame, dims=[1]), torch.flip(p1_frame, dims=[1]), torch.flip(label_0, dims=[1]), torch.flip(label_1, dims=[2]), torch.flip(label_2, dims=[2]), torch.flip(label_3, dims=[2])
        else:
            return i_frame, p0_frame, p1_frame, label_0, label_1, label_2, label_3

            # 随机裁剪

    def __getitem__(self, index):
        # input: p_frm id
        if resample_dataset and self.mode == "train_dataset":
            index = self.sampled_indices[index]  # 重复利用占比较低的TID帧 or 仅采样特定tid的帧
        if self.mode == "train_dataset":
            seq_idx, p_idx = index // 31, index % 31
            if self.enc_mode == 'RA':
                abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1  # P-frame index -> all_frame index
            else:
                abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1  # P-frame index -> all_frame index

            cand_frmid = get_cand_id_list(abs_idx, mode='RA', gop_size=32, ref_len=2, frm_num=32)
            qp_list = [self.qp_dict[abs_idx], self.qp_dict[cand_frmid[0]], self.qp_dict[cand_frmid[1]]]
            tid_list = [self.tid_dict[abs_idx], self.tid_dict[cand_frmid[0]], self.tid_dict[cand_frmid[1]]]
            if opt['train']['load_uv']:
                pass  # # input_content shape, (3,h,w)  # i_frame = np.concatenate([self.y_content[None, seq_idx * 64 + abs_idx], self.uv_content[seq_idx, abs_idx].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)  # p0_frame = np.concatenate([self.y_content[None, seq_idx * 64 + cand_frmid[0]], self.uv_content[seq_idx, cand_frmid[0]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)  # p1_frame = np.concatenate([self.y_content[None, seq_idx * 64 + cand_frmid[1]], self.uv_content[seq_idx, cand_frmid[1]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
            else:
                i_frame = self.y_content['%s/%d/%d/y_comp' % (self.sub_id, seq_idx, abs_idx)]
                p0_frame = self.y_content['%s/%d/%d/y_comp' % (self.sub_id, seq_idx, cand_frmid[0])]
                p1_frame = self.y_content['%s/%d/%d/y_comp' % (self.sub_id, seq_idx, cand_frmid[1])]

            label_0 = self.label['%s/%d/%d/qt_label' % (self.sub_id, seq_idx, abs_idx)]
            mtt_label = self.label['%s/%d/%d/mtt_label' % (self.sub_id, seq_idx, abs_idx)]
            label_0, mtt_label = np.array(label_0), np.array(mtt_label)
            label_1, label_2, label_3 = mtt_label[0], mtt_label[1], mtt_label[2]  # TODO: check

            label_0, label_1, label_2, label_3 = torch.from_numpy(label_0), torch.from_numpy(label_1), torch.from_numpy(label_2), torch.from_numpy(label_3)
            i_frame, p0_frame, p1_frame = torch.from_numpy(np.array(i_frame)), torch.from_numpy(np.array(p0_frame)), torch.from_numpy(np.array(p1_frame))

            if 'A' in self.sub_id and opt['train']['crop_A']:
                # 将A类序列crop成B类分辨率，取1/4 region，以及1-CTU邻域，
                neibo_len = 1
                block_width = i_frame.shape[-1] // (2 * 128) + 2 * neibo_len
                block_height = i_frame.shape[-2] // (2 * 128) + 2 * neibo_len
                max_x = i_frame.shape[-1] - block_width * 128
                max_y = i_frame.shape[-2] - block_height * 128

                # Randomly choose a left-top corner that satisfies the requirements
                crop_x = np.random.randint(0, max_x + 1) // 128 * 128
                crop_y = np.random.randint(0, max_y + 1) // 128 * 128

                i_frame = self.__random_crop_y__(i_frame, crop_x, crop_y, block_width, block_height)
                p0_frame = self.__random_crop_y__(p0_frame, crop_x, crop_y, block_width, block_height)
                p1_frame = self.__random_crop_y__(p1_frame, crop_x, crop_y, block_width, block_height)

                label_0 = self.__random_crop_label__(label_0, crop_x, crop_y, ratio=16, isqt=True, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_1 = self.__random_crop_label__(label_1, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_2 = self.__random_crop_label__(label_2, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
                label_3 = self.__random_crop_label__(label_3, crop_x, crop_y, ratio=4, gt_block_width=block_width - 2 * neibo_len, gt_block_height=block_height - 2 * neibo_len, boundary_len=neibo_len)
            if opt['train']['random_flip']:
                i_frame, p0_frame, p1_frame, label_0, label_1, label_2, label_3 = self.random_flip(i_frame, p0_frame, p1_frame, label_0, label_1, label_2, label_3)

            output = {'i_frame': i_frame, 'p0_frame': p0_frame, 'p1_frame': p1_frame, 'label_0': label_0, "label_1": label_1, "label_2": label_2, "label_3": label_3, "qp_list": torch.tensor(qp_list), "tid_list": torch.tensor(tid_list)}  # qt_label, mtt_label_0, mtt_label_1, mtt_label_2

        elif self.mode == "valid_dataset":
            p_idx = index
            if self.enc_mode == 'RA':
                abs_idx = p_idx // 31 * 32 + p_idx % 31 + 1
            else:
                abs_idx = p_idx // 7 * 8 + p_idx % 7 + 1
            cand_frmid = get_cand_id_list(abs_idx, mode='RA', gop_size=32, ref_len=2, frm_num=100)
            qp_list = [self.qp_dict[abs_idx], self.qp_dict[cand_frmid[0]], self.qp_dict[cand_frmid[1]]]
            tid_list = [self.tid_dict[abs_idx], self.tid_dict[cand_frmid[0]], self.tid_dict[cand_frmid[1]]]
            if opt['train']['load_uv']:
                pass  # i_frame = np.concatenate([self.y_content[None, abs_idx], self.uv_content[abs_idx].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)  # p0_frame = np.concatenate([self.y_content[None, cand_frmid[0]], self.uv_content[cand_frmid[0]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)  # p1_frame = np.concatenate([self.y_content[None, cand_frmid[1]], self.uv_content[cand_frmid[1]].repeat(2, axis=-1).repeat(2, axis=-2)], axis=0)
            else:
                i_frame = self.y_content[None, abs_idx]
                p0_frame = self.y_content[None, cand_frmid[0]]
                p1_frame = self.y_content[None, cand_frmid[1]]
            output = {'i_frame': i_frame, 'p0_frame': p0_frame, 'p1_frame': p1_frame, 'label_0': torch.from_numpy(self.qt_label[p_idx]), "label_1": torch.from_numpy(self.mtt_label[p_idx, 0]), "label_2": torch.from_numpy(self.mtt_label[p_idx, 1]), "label_3": torch.from_numpy(self.mtt_label[p_idx, 2]),
                      "qp_list": torch.tensor(qp_list), "tid_list": torch.tensor(tid_list)}

        return output

    def __len__(self):
        if resample_dataset and self.mode == 'train_dataset':
            return len(self.sampled_indices)  # return self.y_content.shape[0] // 64 * 62 // 31 * 80
        else:
            return self.seq_len * 31


@torch.no_grad()
def evaluation(qt_net, mtt_net, epoch, valid_data, writer, global_iter, flow_net, dire=False, ds=1, stage=1, mtt_mask_net=None):
    """stage = 1, qt
    stage = 2, qt + mt0
    stage = 3, qt + mt0 + mt1"""
    step = 0
    # loss = 0
    ce_loss = 0
    all_accu_list = [0, 0, 0]
    res_loss_list = [0, 0, 0]
    accu_list = [0, 0, 0, 0]
    recall_list = [0, 0, 0, 0, ]  # 预测非零中正确样本比例
    zero_rate_list = [0, 0, 0, 0, ]
    zero_label_list = [0, 0, 0, 0, ]
    accu_mask, zero_mask = 0, 0
    # scale_label_list = [2, 1, 2, 4]
    ce_criterion = torch.nn.BCELoss(reduction='mean')
    l1_criterion = nn.L1Loss()
    if stage >= 1:
        qt_net = qt_net.eval()
    elif stage >= 2:
        mtt_mask_net = mtt_mask_net.eval()
    if stage > 2:
        mtt_net = mtt_net.eval()
    flow_net = flow_net.eval()

    qt_loss, mtt_loss, mtt_mask_loss = 0, 0, 0
    for batch in valid_data:
        label_batch_list = [batch['label_' + str(depth)] for depth in range(4)]
        i_frame, p0_frame, p1_frame = batch['i_frame'], batch['p0_frame'], batch['p1_frame']  # input_batch shape(B, ref_frm_num+1, 384, 384), shape(B, ref_frm_num+1, 2, 192, 192)
        i_frame, p0_frame, p1_frame = i_frame.cuda().float(), p0_frame.cuda().float(), p1_frame.cuda().float()
        label_batch_list = [item.cuda().float() for item in label_batch_list]
        if opt['network']['qml']:
            qp_list = batch['qp_list'].cuda().float()
        elif opt['network']['tml']:
            qp_list = batch['tid_list'].cuda().float()
        else:
            qp_list = batch['qp_list'].cuda().float()

        if stage == 1:
            # qt net
            i_frame, p0_frame, p1_frame = i_frame[:, :, ::4, ::4], p0_frame[:, :, ::4, ::4], p1_frame[:, :, ::4, ::4]
            p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
            p0_flow, p1_flow = p0_flow[-1], p1_flow[-1]
            if opt['network']['wo_op']:
                p0_flow, p1_flow = torch.zeros_like(p0_flow), torch.zeros_like(p1_flow)
                trans_flow_DAM = False
            else:
                trans_flow_DAM = True
            qt_pred_list, qt_feature = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow, p1_flow], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=ds)
            qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
            qt_pred = qt_pred_list[-1]
            max_pred_depth = 1
        elif stage == 2:
            # mtt-mask
            if opt['network']['large_qt_model']:
                p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
                qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow[-1], p1_flow[-1]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=0)
                i_frame, p0_frame, p1_frame = i_frame[:, :, ::2, ::2], p0_frame[:, :, ::2, ::2], p1_frame[:, :, ::2, ::2]
            else:
                i_frame, p0_frame, p1_frame = i_frame[:, :, ::2, ::2], p0_frame[:, :, ::2, ::2], p1_frame[:, :, ::2, ::2]
                p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt_mask')
                qt_pred_list, _ = qt_net(i_frame[:, 0:1, ::2, ::2] / 255.0, torch.stack([p0_flow[0], p1_flow[0]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2] / 255.0, p1_frame=p1_frame[:, 0:1, ::2, ::2] / 255.0, out_medium_feat=True, upsample=4)
            qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
            qt_pred = qt_pred_list[-1]
            mtt_mask_list = mtt_mask_net(i_frame[:, 0:1] / 255.0, p0_flow=p0_flow[1], p1_flow=p1_flow[1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame / 255.0, p1_frame=p1_frame / 255.0)
            mtt_mask_pred = mtt_mask_list[-1].round()
            max_pred_depth = 2
            #
            sum_mtt_weight = label_batch_list[1][:, 0] + label_batch_list[2][:, 0] + label_batch_list[3][:, 0]
            sum_mtt_weight = ((F.max_pool2d(sum_mtt_weight.float(), kernel_size=4) + label_batch_list[0]).cuda() > qt_pred_list[-1][:, 0].round())
            mtt_mask_label = rearrange(rearrange(sum_mtt_weight, 'b (hi h) (wi w) -> b hi h wi w', h=8, w=8), 'b hi h wi w -> b hi wi h w').sum(dim=(3, 4))
            mtt_mask_label = torch.clip(mtt_mask_label, min=0, max=1).float()
            mtt_mask_loss += ce_criterion(mtt_mask_pred.flatten(), mtt_mask_label.flatten())
            #
            accu_mask += torch.sum(mtt_mask_pred.flatten() == mtt_mask_label.flatten()) / mtt_mask_label.numel() * 100
            zero_mask += torch.sum(mtt_mask_pred == 0) / mtt_mask_pred.numel() * 100

        elif stage == 3:
            # mtt-net for depth
            if opt['network']['large_qt_model']:
                p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
                qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow[-1], p1_flow[-1]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True,
                                            upsample=0)  # i_frame, p0_frame, p1_frame = i_frame[:,:,::2,::2], p0_frame[:,:,::2,::2], p1_frame[:,:,::2,::2]
            else:
                i_frame, p0_frame, p1_frame = i_frame[:, :, ::4, ::4], p0_frame[:, :, ::4, ::4], p1_frame[:, :, ::4, ::4]
                p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt_mask')
                qt_pred_list, _ = qt_net(i_frame[:, 0:1, ::2, ::2] / 255.0, torch.stack([p0_flow[0], p1_flow[0]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2] / 255.0, p1_frame=p1_frame[:, 0:1, ::2, ::2] / 255.0, out_medium_feat=True, upsample=4)
            qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
            qt_pred = qt_pred_list[-1]

            # TODO: 此处训练模型时没有注意到应该将i_frame,p0_frame,p1_frame归一化
            mtt_mask_list = mtt_mask_net(i_frame[:, 0:1, ::2, ::2], p0_flow=p0_flow[1], p1_flow=p1_flow[1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, :, ::2, ::2], p1_frame=p1_frame[:, :, ::2, ::2])
            # mtt_mask_list = mtt_mask_net(i_frame[:, 0:1, ::2, ::2] / 255.0, p0_flow=p0_flow[1], p1_flow=p1_flow[1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:,:, ::2, ::2] / 255.0, p1_frame=p1_frame[:,:, ::2, ::2] / 255.0)

            # mtt_mask_pred = torch.softmax(mtt_mask_list[-1], dim=-1)
            mtt_mask_pred = mtt_mask_list[-1].view(i_frame.shape[0], -1)

            mtt_depth_map_list, ctu_decision, drop_decision = mtt_net(luma=i_frame[:, 0:1] / 255.0, p0_flow=p0_flow[-1], p1_flow=p1_flow[-1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame / 255.0, p1_frame=p1_frame / 255.0, ctu_decision=mtt_mask_pred, mask_ratio=0.3)

            # 将mtt_depth_map_list还原回去
            mtt_pred = []
            for layer_depth in range(3):
                if dire:
                    save_mtt_depth = torch.argmax(mtt_depth_map_list[layer_depth], dim=-1) - 1
                else:
                    save_mtt_depth = torch.argmax(mtt_depth_map_list[layer_depth], dim=-1)

                    # mtt_pred_single = torch.zeros_like(label_batch_list[layer_depth + 1][:, 0:1, ::2, ::2])
                mtt_pred_single = torch.zeros(i_frame.shape[0], i_frame.shape[-1] // 128 * i_frame.shape[-2] // 128, 32, 32).cuda()
                drop_mtt_depth = torch.zeros(i_frame.shape[0], i_frame.shape[-1] // 128 * i_frame.shape[-2] // 128 - save_mtt_depth.shape[0], 32, 32).cuda()

                mtt_pred_single = rearrange(mtt_pred_single, 'b n h w -> b n (h w)')
                drop_mtt_depth = rearrange(drop_mtt_depth, 'b n h w -> b n (h w)')
                save_mtt_depth = rearrange(save_mtt_depth, '(b n) h w -> b n (h w)', b=i_frame.shape[0]).float()

                mtt_pred_single = batch_index_fill(mtt_pred_single, save_mtt_depth, drop_mtt_depth, ctu_decision, drop_decision)
                mtt_pred.append(rearrange(mtt_pred_single, 'b (hi wi) (h w) -> b (hi h) (wi w)', hi=i_frame.shape[-2] // 128, wi=i_frame.shape[-1] // 128, h=32, w=32))

        # # dire
        # import palettable
        # import matplotlib.pyplot as plt
        # plt.clf()
        # fig, ax = plt.subplots()
        # im = ax.imshow(mtt_pred[0][0].detach().cpu(), cmap=palettable.colorbrewer.diverging.PuOr_3.mpl_colormap)
        # cbar = plt.colorbar(im, fraction=0.025, pad=0.04)
        # plt.savefig('/code/demo1.png')
        # depth
        # plt.imsave('/code/demo1.png', (mtt_pred[0][0] == 2).detach().cpu())
        if dire:
            # depth model
            for pred_depth in range(opt['train']['max_layer_depth']):
                if pred_depth == 0:
                    qt_loss += l1_criterion(qt_pred, label_batch_list[0].unsqueeze(1))
                    mask = 1
                    res_loss_list.append(qt_loss.item())
                    pred_label, GT_label = qt_pred.round(), label_batch_list[0].unsqueeze(1).round()
                if pred_depth >= 1:
                    mtt_depth = pred_depth - 1

                    # accu_depth_map = label_batch_list[pred_depth][:, 0]
                    if mtt_depth == 0:
                        accu_depth_map = label_batch_list[pred_depth][:, 1]
                    else:
                        accu_depth_map = label_batch_list[pred_depth][:, 1]

                    mtt_loss += l1_criterion(mtt_pred[mtt_depth], accu_depth_map)  # accu loss
                    # mask = mtt_pred[mtt_depth][-1] * mask
                    res_loss_list.append(l1_criterion(mtt_pred[mtt_depth], accu_depth_map).item())
                    # pred_label, GT_label = mtt_pred[mtt_depth][-1].round(), label_batch_list[pred_depth][:, 0:1].round()
                    pred_label, GT_label = mtt_pred[mtt_depth].round(), accu_depth_map.round()

                # 累积 metric
                zero_rate_list[pred_depth] += torch.sum(pred_label == 0) / pred_label.numel() * 100
                recall_list[pred_depth] += torch.sum((pred_label == GT_label) * (pred_label != 0)) / torch.sum(pred_label != 0) * 100
                accu_list[pred_depth] += torch.sum(pred_label == GT_label) / pred_label.numel() * 100

            if mtt_mask_net is not None:
                # mtt-mask预测准确率
                ctu_label = F.max_pool2d(label_batch_list[1][:, 0], kernel_size=32).reshape(-1, 1).clamp(max=1).round()
                mtt_mask_pred = torch.round(mtt_mask_pred)
                mtt_mask_pred = mtt_mask_pred.reshape(-1, 1).round()
                accu_mask += torch.sum(ctu_label == mtt_mask_pred) / mtt_mask_pred.numel() * 100
                zero_mask += torch.sum(mtt_mask_pred.round() == 0) / mtt_mask_pred.numel() * 100
        else:
            # depth model
            for pred_depth in range(opt['train']['max_layer_depth']):
                if pred_depth == 0:
                    qt_loss += l1_criterion(qt_pred, label_batch_list[0].unsqueeze(1))
                    mask = 1
                    res_loss_list.append(qt_loss.item())
                    pred_label, GT_label = qt_pred.round(), label_batch_list[0].unsqueeze(1).round()
                if pred_depth >= 1:
                    mtt_depth = pred_depth - 1

                    # accu_depth_map = label_batch_list[pred_depth][:, 0]
                    if mtt_depth == 0:
                        accu_depth_map = label_batch_list[pred_depth][:, 0]
                    else:
                        accu_depth_map = label_batch_list[pred_depth][:, 0]

                    mtt_loss += l1_criterion(mtt_pred[mtt_depth], accu_depth_map)  # accu loss
                    # mask = mtt_pred[mtt_depth][-1] * mask
                    res_loss_list.append(l1_criterion(mtt_pred[mtt_depth], accu_depth_map).item())
                    # pred_label, GT_label = mtt_pred[mtt_depth][-1].round(), label_batch_list[pred_depth][:, 0:1].round()
                    pred_label, GT_label = mtt_pred[mtt_depth].round(), accu_depth_map.round()

                # 累积 metric
                zero_rate_list[pred_depth] += torch.sum(pred_label == 0) / pred_label.numel() * 100
                recall_list[pred_depth] += torch.sum((pred_label == GT_label) * (pred_label != 0)) / torch.sum(pred_label != 0) * 100
                accu_list[pred_depth] += torch.sum(pred_label == GT_label) / pred_label.numel() * 100

            if mtt_mask_net is not None:
                # mtt-mask预测准确率
                ctu_label = F.max_pool2d(label_batch_list[1][:, 0], kernel_size=32).reshape(-1, 1).clamp(max=1).round()
                mtt_mask_pred = torch.round(mtt_mask_pred)
                mtt_mask_pred = mtt_mask_pred.reshape(-1, 1).round()
                accu_mask += torch.sum(ctu_label == mtt_mask_pred) / mtt_mask_pred.numel() * 100
                zero_mask += torch.sum(mtt_mask_pred.round() == 0) / mtt_mask_pred.numel() * 100

                # l1_loss += criterion(pred_label_list[pred_depth], label_batch_list[pred_depth]) * weight_of_loss[pred_depth]

        step += 1

    if opt['depth_model']:
        accu_list = [ele / step for ele in accu_list]
        zero_rate_list = [ele / step for ele in zero_rate_list]
        zero_label_list = [ele / step for ele in zero_label_list]
        recall_list = [ele / step for ele in recall_list]
        accu_mask /= step
        zero_mask /= step
    elif opt['dire_model']:
        accu_list = [ele / step for ele in accu_list]
        res_loss_list = [ele / step for ele in res_loss_list]
        recall_list = [ele / step for ele in recall_list]
        all_accu_list = [ele / step for ele in all_accu_list]
        avg_loss = sum(res_loss_list)

    if opt['log']['use_tensor_board']:
        if opt['depth_model']:
            writer.add_scalar('loss/qt_loss', qt_loss / step, global_iter)
            writer.add_scalar('loss/mtt_mask_loss', mtt_mask_loss / step, global_iter)
            writer.add_scalar('loss/mtt_loss', mtt_loss / step, global_iter)
            for pred_depth in range(4):
                writer.add_scalar('criterion/accu_depth%d' % pred_depth, accu_list[pred_depth], global_iter)
                writer.add_scalar('criterion/recall_depth%d' % pred_depth, recall_list[pred_depth], global_iter)
            writer.add_scalar('criterion/accu_mask', accu_mask, global_iter)
            writer.add_scalar('criterion/zero_depth', zero_mask, global_iter)
        elif opt['dire_model']:
            writer.add_scalar('loss/ce_loss', avg_loss, global_iter)
            for pred_depth in range(3):
                writer.add_scalar('criterion/accu_depth%d' % pred_depth, accu_list[pred_depth], global_iter)
                writer.add_scalar('criterion/recall_depth%d' % pred_depth, recall_list[pred_depth], global_iter)
                writer.add_scalar('criterion/all_accu_depth%d' % pred_depth, all_accu_list[pred_depth], global_iter)

    print('VALID')
    print("epoch:%d\t iter:%d\t qt_loss:%.3f\t mtt_loss:%.3f" % (epoch, global_iter, qt_loss / step, mtt_loss / step))
    for pred_depth in range(4):
        print("depth%d\t accuracy: " % pred_depth, '%.4f%%' % accu_list[pred_depth], "recall: ", '%.4f%%' % recall_list[pred_depth], "\t zero_rate: ", '%.4f%%' % zero_rate_list[pred_depth], "\t zero_label: ", '%.4f%%' % zero_label_list[pred_depth])
    print("mtt-mask accuracy: ", '%.4f%%' % accu_mask, "zero_rate: ", '%.4f%%' % zero_mask)

    return (qt_loss + mtt_loss) / step


def YUV2RGB(Y, U, V, isYUV420=True):
    """Y: (frame_num, height, width), U/V: (frame_num, height//2, width//2)"""
    FRAME_NUM, IMG_HEIGHT, IMG_WIDTH = Y.shape[-3], Y.shape[-2], Y.shape[-1]
    bgr_data = torch.zeros(FRAME_NUM, 3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.uint8).to(Y.device)
    if (isYUV420):
        U = F.interpolate(U, scale_factor=2)
        V = F.interpolate(V, scale_factor=2)

    c = (Y - 16) * 298
    d = U - 128
    e = V - 128

    r = torch.floor((c + 409 * e + 128) / 256).long()
    g = torch.floor((c - 100 * d - 208 * e + 128) / 256).long()
    b = torch.floor((c + 516 * d + 128) / 256).long()

    r = torch.where(r < 0, 0, r)
    r = torch.where(r > 255, 255, r)

    g = torch.where(g < 0, 0, g)
    g = torch.where(g > 255, 255, g)

    b = torch.where(b < 0, 0, b)
    b = torch.where(b > 255, 255, b)

    bgr_data[:, 2, :, :] = r
    bgr_data[:, 1, :, :] = g
    bgr_data[:, 0, :, :] = b
    # return (n,3,h,w)
    return bgr_data


@torch.no_grad()
def flow_norm(tensorFlow):
    Backward_tensorGrid_cpu = {}
    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
    Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cpu()

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorFlow.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorFlow.size(2) - 1.0) / 2.0)], 1)
    return tensorFlow


@torch.no_grad()
def dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, label_batch_list=None, out_type='res', flag=None):
    """return：
    input_batch, 当前帧纹理YUV，前一帧和当前帧warp后的残差，后一帧和当前帧warp后的残差 or 经过64x64对齐后的p_frame_aligned
    flow, 前一帧光流和mdf, 后一帧光流和mdf
    """
    if flag == 'mtt':
        Lm = 0
    elif flag == 'mtt_mask':
        Lm = 1
    elif flag == 'qt':
        Lm = 2
    # YYY spynet下采样
    I_frame_YYY = F.interpolate(i_frame[:, 0:1], scale_factor=1 / ds, mode='nearest').repeat(1, 3, 1, 1) / 255.0
    P0_frame_YYY = F.interpolate(p0_frame[:, 0:1], scale_factor=1 / ds, mode='nearest').repeat(1, 3, 1, 1) / 255.0
    P1_frame_YYY = F.interpolate(p1_frame[:, 0:1], scale_factor=1 / ds, mode='nearest').repeat(1, 3, 1, 1) / 255.0
    P0_flow_list = flow_net(im1=I_frame_YYY, im2=P0_frame_YYY, Lm=Lm)
    P1_flow_list = flow_net(im1=I_frame_YYY, im2=P1_frame_YYY, Lm=Lm)
    if flag == 'qt':
        return P0_flow_list[-3], P1_flow_list[-3]  # 下采样四倍
    elif flag == 'mtt_mask':
        return [P0_flow_list[-3], P0_flow_list[-2]], [P1_flow_list[-3], P1_flow_list[-2]]
    elif flag == 'mtt':
        return [P0_flow_list[-3], P0_flow_list[-2], P0_flow_list[-1]], [P1_flow_list[-3], P1_flow_list[-2], P1_flow_list[-1]]
    else:
        raise Exception('invalid flag')  # return P0_flow_list[-1], P1_flow_list[-1]  # flow


def train_qt_net(qp, writer, init_global_iter, model_path, dataset_id_list, sub_epoch=5, base_epoch=0, lr=1e-3):
    """
    ds=4, 下采样四倍（消融实验，其他倍率的下采样操作）
    """
    qt_net = QT_Net_HLG(qml=opt['network']['qml'], guide=True, text_fe=opt['network']['text_fe'])
    flow_net = ME_Spynet(me_model_dir=opt['path']['me_model_dir'])

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            devices_id = list(range(opt['gpu_num']))
            qt_net = qt_net.cuda(device=devices_id[0])
            qt_net = torch.nn.DataParallel(qt_net, device_ids=devices_id)
            flow_net = flow_net.cuda(device=devices_id[0])
            flow_net = torch.nn.DataParallel(flow_net, device_ids=devices_id)
        else:
            qt_net = qt_net.cuda()
            flow_net = flow_net.cuda()

    if model_path is not None:
        print("loaded weight from ", model_path)
        qt_net.load_state_dict(torch.load(model_path)['qt_net'])

    flow_net = flow_net.eval()
    # criterion and optimizer
    l1_criterion = torch.nn.L1Loss()
    params_list = [{'params': qt_net.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, sub_epoch + 1, 1)), gamma=opt['train']['gamma'])

    global_iter = init_global_iter
    print('start training ...')
    dataset_id = -1

    # load valid dataset
    if opt['datasets']['val']['use_valid']:
        valid_dataset = ValidDataset(qp, sub_id=opt['datasets']['val']['valid_id'], mode='valid', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
        valid_data = DataLoader(valid_dataset, batch_size=opt['datasets']['val']['batchSize'], shuffle=False, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
        print("loaded " + opt['datasets']['val']['valid_id'] + " valid dataset")
    # warmup=True 在训练的最开始，使用ABC轮流训练，在梯度较大时更新到较好的位置，学到分辨率自适应的特征
    if opt['train']['warmup']:
        print('warmup...')
        ori_dataset_id_list = dataset_id_list
        dataset_id_list = ['B', 'A', 'C']
    for epoch in range(1, sub_epoch + 1):
        qt_net = qt_net.train()
        total_qt_loss, total_loss = 0, 0
        if opt['train']['warmup'] and epoch == 16:
            dataset_id_list = ori_dataset_id_list
            print('warmup ending...')
        if epoch % 5 == 0 or epoch == 1:
            dataset_id = (dataset_id + 1) % len(dataset_id_list)
        if opt['train']['paired'] or (epoch % 5 == 0 or epoch == 1):
            train_dataset = MyDataset(qp, dataset_id_list[dataset_id], mode='train', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
            batchSize = opt['datasets']['train']['batchSize']
            # batchSize for multi-scale training, but it compromised the performance
            if not opt['train']['crop_A']:
                if dataset_id_list[dataset_id] == 'A':
                    batchSize = opt['datasets']['train']['batchSize']
                elif dataset_id_list[dataset_id] == 'B':
                    batchSize = opt['datasets']['train']['batchSize'] * 4
                elif dataset_id_list[dataset_id] == 'C':
                    batchSize = opt['datasets']['train']['batchSize'] * 16
                elif dataset_id_list[dataset_id] == 'D':
                    batchSize = opt['datasets']['train']['batchSize'] * 64

            if opt['train']['paired']:
                shuffle = False
            else:
                shuffle = True

            if opt['datasets']['train']['num_workers'] > 0:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'], multiprocessing_context='fork')
            else:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
            print('loaded ' + dataset_id_list[dataset_id] + ' train dataset')

        for step, batch in enumerate(train_data):
            # print(step)
            optimizer.zero_grad()
            label_batch_list = [batch['label_' + str(depth)] for depth in range(4)]
            i_frame, p0_frame, p1_frame = batch['i_frame'], batch['p0_frame'], batch['p1_frame']
            i_frame, p0_frame, p1_frame = i_frame[:, None, ::ds, ::ds], p0_frame[:, None, ::ds, ::ds], p1_frame[:, None, ::ds, ::ds]
            # i_frame, p0_frame, p1_frame = i_frame[:, None], p0_frame[:, None], p1_frame[:, None]
            qp_list = batch['qp_list']
            if torch.cuda.is_available():
                i_frame, p0_frame, p1_frame = i_frame.cuda().float(), p0_frame.cuda().float(), p1_frame.cuda().float()
                label_batch_list = [item.cuda().float() for item in label_batch_list]
                qp_list = qp_list.cuda().float()
            with torch.no_grad():
                # 得到和下采样后的i_frame相同尺寸的光流，避免质量太差
                p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
                p0_flow, p1_flow = p0_flow[-1], p1_flow[-1]

            if opt['network']['wo_op']:
                p0_flow, p1_flow = torch.zeros_like(p0_flow), torch.zeros_like(p1_flow)
                trans_flow_DAM = False
            else:
                trans_flow_DAM = True

            # i_frame, p0_frame, p1_frame = i_frame[:,:,::ds,::ds], p0_frame[:,:,::ds,::ds], p1_frame[:,:,::ds,::ds]
            if 'A' in dataset_id_list[dataset_id] and opt['train']['crop_A']:
                dsb = 128 // ds  # down-sampled block size
                qt_pred_list = qt_net(i_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb, dsb:i_frame.shape[-1] - dsb] / 255.0, torch.stack([p0_flow[:, :, dsb:(i_frame.shape[-2] - dsb), dsb: (i_frame.shape[-1] - dsb)], p1_flow[:, :, dsb:(i_frame.shape[-2] - dsb), dsb: (i_frame.shape[-1] - dsb)]], dim=1), qp=qp_list[:, 0:1], \
                                      trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb, dsb:i_frame.shape[-1] - dsb] / 255.0, p1_frame=p1_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb, dsb:i_frame.shape[-1] - dsb] / 255.0, out_medium_feat=False, upsample=ds)
            else:
                qt_pred_list = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow, p1_flow], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=False, upsample=ds)

            qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]

            qt_loss = 0
            qt_label = label_batch_list[0].unsqueeze(1)
            # if opt['train']['qt_nz_lamba']:
            #     # 以0作为base
            #     weight_0 = 1
            #     weight_1 = ((qt_label==0).sum() / (qt_label==1).sum()) ** opt['train']['qt_nz_lamba'] if (qt_label==1).sum() != 0 else 0
            #     weight_2 = min(((qt_label==0).sum() / (qt_label==2).sum()) ** opt['train']['qt_nz_lamba'], 50) if (qt_label==2).sum() != 0 else 0
            #     weight_3 = min(((qt_label==0).sum() / (qt_label==3).sum()) ** (opt['train']['qt_nz_lamba']), 50) if (qt_label==3).sum() != 0 else 0
            #     # # 以全部元素作为base
            #     # weight_0 = (qt_label.sum() / (qt_label == 0).sum()) ** opt['train']['qt_nz_lamba'] if (qt_label == 0).sum() != 0 else 0
            #     # weight_1 = (qt_label.sum() / (qt_label == 1).sum()) ** opt['train']['qt_nz_lamba'] if (qt_label == 1).sum() != 0 else 0
            #     # weight_2 = (qt_label.sum() / (qt_label == 2).sum()) ** opt['train']['qt_nz_lamba'] if (qt_label == 2).sum() != 0 else 0
            #     # weight_3 = (qt_label.sum() / (qt_label == 3).sum()) ** opt['train']['qt_nz_lamba'] if (qt_label == 3).sum() != 0 else 0
            #     #
            #     total_weight = weight_0 + weight_1 + weight_2 + weight_3
            #     weight_0, weight_1, weight_2, weight_3 = weight_0 / total_weight * 4, weight_1 / total_weight * 4, weight_2 / total_weight * 4, weight_3 / total_weight * 4
            #     # print(weight_0, int(weight_1.item()), int(weight_2.item()), int(weight_3))
            #     reweight_mask = (qt_label == 0) * weight_0 + (qt_label == 1) * weight_1 + (qt_label == 2) * weight_2 + (qt_label == 3) * weight_3
            # else:
            #     reweight_mask = 1
            if opt['train']['qt_nz_weight'] is not None:
                reweight_mask = torch.ones_like(qt_pred_list[0])
                if opt['train']['qt_nz_weight_C'] is not None and 'C' in dataset_id_list[dataset_id]:
                    # 针对C类的特殊权重
                    reweight_mask = reweight_mask + (qt_label == 1) * opt['train']['qt_nz_weight_C'][0] + (qt_label == 2) * opt['train']['qt_nz_weight_C'][1] + (qt_label == 3) * opt['train']['qt_nz_weight_C'][2]
                elif opt['train']['qt_nz_weight_A'] is not None and 'A' in dataset_id_list[dataset_id]:
                    # 针对A类的特殊权重
                    reweight_mask = reweight_mask + (qt_label == 1) * opt['train']['qt_nz_weight_A'][0] + (qt_label == 2) * opt['train']['qt_nz_weight_A'][1] + (qt_label == 3) * opt['train']['qt_nz_weight_A'][2]
                else:
                    reweight_mask = reweight_mask + (qt_label == 1) * opt['train']['qt_nz_weight'][0] + (qt_label == 2) * opt['train']['qt_nz_weight'][1] + (qt_label == 3) * opt['train']['qt_nz_weight'][2]
            else:
                reweight_mask = 1
            if opt['train']['multi_loss']:
                for qt_single in qt_pred_list:
                    qt_loss += l1_criterion(qt_single * reweight_mask, qt_label * reweight_mask)
            else:
                qt_loss += l1_criterion(qt_pred_list[-1] * reweight_mask, qt_label * reweight_mask)
            loss = qt_loss
            loss.backward()
            optimizer.step()
            total_qt_loss += loss.detach().item()
            if (step + 1) % 100 == 0:
                print("iter:%d\t qt_loss:%.5f" % (step, loss))
            del label_batch_list, batch, qt_pred_list, p0_flow, p1_flow
            if (step + 1) % 500 == 0:
                gc.collect()

            if opt['log']['use_tensor_board']:
                if (step + 1) % 100 == 0:
                    writer.add_scalar('loss/100step_loss', total_qt_loss / 100, global_iter)  # 统计每10个epoch的loss
                    total_qt_loss = 0
            global_iter += 1

        if (epoch + 1) % 1 == 0:
            state = {'qt_net': qt_net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + base_epoch}
            torch.save(state, os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth'))
            out_model_path = os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth')

        if ((epoch + 1) % 5 == 0) and opt['datasets']['val']['use_valid']:
            valid_loss = []
            valid_loss_mean = evaluation(qt_net, None, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=1)

            print("Evaluation: loss=%.4f" % (valid_loss_mean))
            valid_loss.append(valid_loss_mean.item())

        print("learning rate : %.3e" % scheduler.get_last_lr()[0])
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_iter)
        scheduler.step()
        gc.collect()

    return global_iter, out_model_path, base_epoch + epoch + 1, scheduler.get_last_lr()[0]


def binary_threshold(tensor, threshold):
    return torch.where(tensor >= threshold, torch.tensor(1), torch.tensor(0))


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict) # sigmoide获取概率
        pt = predict
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def train_mtt_mask_net(qp, writer, init_global_iter, qt_model_path, mtt_mask_model_path, dataset_id_list, sub_epoch=5, base_epoch=0, lr=1e-3):
    if opt['network']['large_qt_model']:
        from models.backbone_large import QT_Net_HLG
        if qp == 37:
            guide = False
        else:
            guide = True
        qt_net = QT_Net_HLG(spp=False, qml=True, tml=True, guide=guide)
    else:
        from models.backbone import QT_Net_HLG
        qt_net = QT_Net_HLG(qml=opt['network']['qml'], guide=True, text_fe=opt['network']['text_fe'])
    flow_net = ME_Spynet(me_model_dir=opt['path']['me_model_dir'])
    mtt_mask_net = MTT_mask_net(qml=opt['network']['qml'], dlm=opt['network']['dlm'])

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            devices_id = list(range(opt['gpu_num']))
            qt_net = qt_net.cuda(device=devices_id[0])
            qt_net = torch.nn.DataParallel(qt_net, device_ids=devices_id)
            flow_net = flow_net.cuda(device=devices_id[0])
            flow_net = torch.nn.DataParallel(flow_net, device_ids=devices_id)
            mtt_mask_net = mtt_mask_net.cuda(device=devices_id[0])
            mtt_mask_net = torch.nn.DataParallel(mtt_mask_net, device_ids=devices_id)

        else:
            qt_net = qt_net.cuda()
            flow_net = flow_net.cuda()
            mtt_mask_net = mtt_mask_net.cuda()

    print("loaded qt_model weight from ", qt_model_path)
    if torch.cuda.device_count() > 1:
        qt_net.load_state_dict(torch.load(qt_model_path)['qt_net'])
    else:
        qt_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(qt_model_path)['qt_net'].items()})

    if mtt_mask_model_path is not None:
        print("loaded mtt_mask model weight from ", mtt_mask_model_path)
        if torch.cuda.device_count() > 1:
            mtt_mask_net.load_state_dict(torch.load(mtt_mask_model_path)['mtt_mask_net'])
        else:
            mtt_mask_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(mtt_mask_model_path)['mtt_mask_net'].items()})

    flow_net = flow_net.eval()
    qt_net = qt_net.eval()
    # criterion and optimizer
    # l1_criterion = torch.nn.L1Loss()
    params_list = [{'params': mtt_mask_net.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, sub_epoch + 1, 1)), gamma=opt['train']['gamma'])

    global_iter = init_global_iter
    print('start training ...')
    dataset_id = -1

    # load valid dataset
    if opt['datasets']['val']['use_valid']:
        valid_dataset = ValidDataset(qp, sub_id=opt['datasets']['val']['valid_id'], mode='valid', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
        valid_data = DataLoader(valid_dataset, batch_size=opt['datasets']['val']['batchSize'], shuffle=False, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
        print("loaded " + opt['datasets']['val']['valid_id'] + " valid dataset")
    # warmup
    if opt['train']['warmup']:
        print('warmup...')
        ori_dataset_id_list = dataset_id_list
        dataset_id_list = ['B', 'A', 'C']

    for epoch in range(1, sub_epoch + 1):
        mtt_mask_net = mtt_mask_net.train()
        total_mtt_mask_loss, total_loss = 0, 0
        if opt['train']['warmup'] and epoch == 16:
            dataset_id_list = ori_dataset_id_list
            print('warmup ending...')
        if epoch % 1 == 0 or epoch == 1:
            dataset_id = (dataset_id + 1) % len(dataset_id_list)
        if opt['train']['paired'] or (epoch % 1 == 0 or epoch == 1):
            train_dataset = MyDataset(qp, dataset_id_list[dataset_id], mode='train', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
            batchSize = opt['datasets']['train']['batchSize']
            # batchSize for multi-scale training, but it compromised the performance
            if not opt['train']['crop_A']:
                if dataset_id_list[dataset_id] == 'A':
                    # batchSize = opt['datasets']['train']['batchSize'] // 3 - 4
                    batchSize = opt['datasets']['train']['batchSize'] // 5
                elif dataset_id_list[dataset_id] == 'C':
                    batchSize = opt['datasets']['train']['batchSize'] * 4
            print("batchsize: ", batchSize)

            if opt['train']['paired']:
                shuffle = False
            else:
                shuffle = True

            if opt['datasets']['train']['num_workers'] > 0:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'], multiprocessing_context='fork')
            else:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
            print('loaded ' + dataset_id_list[dataset_id] + ' train dataset')

        for step, batch in enumerate(train_data):
            qt_net = qt_net.eval()
            mtt_mask_net = mtt_mask_net.train()
            optimizer.zero_grad()
            label_batch_list = [batch['label_' + str(depth)] for depth in range(4)]
            i_frame, p0_frame, p1_frame = batch['i_frame'], batch['p0_frame'], batch['p1_frame']
            i_frame, p0_frame, p1_frame = i_frame[:, None, ::ds, ::ds], p0_frame[:, None, ::ds, ::ds], p1_frame[:, None, ::ds, ::ds]
            qp_list = batch['qp_list']
            if opt['gpu_num'] > 1:
                # i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[2]).float(), p1_frame.cuda(device=devices_id[3]).float()
                i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[0]).float(), p1_frame.cuda(device=devices_id[1]).float()
            else:
                i_frame, p0_frame, p1_frame = i_frame.cuda().float(), p0_frame.cuda().float(), p1_frame.cuda().float()

            # label_batch_list = [item.cuda().float() for item in label_batch_list]
            qp_list = qp_list.cuda().float()
            # qt_net inference
            with torch.no_grad():
                # mtt-mask训练中，默认直接inference整张图
                # if 'A' in dataset_id_list[dataset_id] and opt['train']['crop_A']:
                #     dsb = 128 // 2  # down-sampled block size
                #     qt_pred_list, texture_feat = qt_net(i_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb:2, dsb:i_frame.shape[-1] - dsb:2] / 255.0,
                #                                         torch.stack([p0_flow[0][:, :, dsb:(i_frame.shape[-2] - dsb), dsb: (i_frame.shape[-1] - dsb)], p1_flow[0][:, :, dsb:(i_frame.shape[-2] - dsb), dsb: (i_frame.shape[-1] - dsb)]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True,
                #                                         p0_frame=p0_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb:2, dsb:i_frame.shape[-1] - dsb:2] / 255.0, p1_frame=p1_frame[:, 0:1, dsb:i_frame.shape[-2] - dsb:2, dsb:i_frame.shape[-1] - dsb:2] / 255.0, out_medium_feat=True, upsample=4)
                # else:
                if opt['network']['large_qt_model']:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
                    # p0_flow, p1_flow = p0_flow[-1], p1_flow[-1]
                    qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow[-1], p1_flow[-1]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=0)
                    i_frame, p0_frame, p1_frame = i_frame[:, :, ::2, ::2], p0_frame[:, :, ::2, ::2], p1_frame[:, :, ::2, ::2]
                else:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt_mask')
                    qt_pred_list, _ = qt_net(i_frame[:, 0:1, ::2, ::2] / 255.0, torch.stack([p0_flow[0], p1_flow[0]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2] / 255.0, p1_frame=p1_frame[:, 0:1, ::2, ::2] / 255.0, out_medium_feat=True, upsample=4)
                qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
                qt_pred = qt_pred_list[-1]

            i_frame.requires_grad, p0_frame.requires_grad, p1_frame.requires_grad = True, True, True
            qp_list.requires_grad = True
            mtt_mask_list = mtt_mask_net(i_frame[:, 0:1] / 255.0, p0_flow=p0_flow[1], p1_flow=p1_flow[1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame / 255.0, p1_frame=p1_frame / 255.0)

            # # mtt_mask_label = where(Qt_depth_pred < QT_depth + MT_depth)
            sum_mtt_weight = label_batch_list[1][:, 0] + label_batch_list[2][:, 0] + label_batch_list[3][:, 0]
            sum_mtt_weight = ((F.max_pool2d(sum_mtt_weight.float(), kernel_size=4) + label_batch_list[0]).cuda() > qt_pred_list[-1][:, 0].round())

            mtt_mask_label = rearrange(rearrange(sum_mtt_weight, 'b (hi h) (wi w) -> b hi h wi w', h=8, w=8), 'b hi h wi w -> b hi wi h w').sum(dim=(3, 4)).cuda().float()
            mtt_mask_label = torch.clip(mtt_mask_label, min=0, max=1)
            mtt_mask_label.requires_grad = True

            mtt_mask_loss = 0
            if opt['train']['focal_loss']:
                ce_criterion = BCEFocalLoss(gamma=0, alpha=0.5, reduction='mean')
            elif opt['train']['mtt_mask_weight']:
                # 对于mask=1的区域增加不同的权重
                mtt_mask_weight = F.normalize(mtt_mask_label, p=1, dim=(1, 2)) * mtt_mask_label[0].numel()
                mtt_mask_weight = torch.clip(mtt_mask_weight, min=0.5).flatten()
                ce_criterion = torch.nn.BCELoss(weight=mtt_mask_weight, reduction='mean')
            else:
                ce_criterion = torch.nn.BCELoss(reduction='mean')

            for mtt_mask_single in mtt_mask_list:
                mtt_mask_loss += ce_criterion(mtt_mask_single.flatten(), binary_threshold(mtt_mask_label.flatten(), threshold=0.1).float())
            loss = mtt_mask_loss
            loss.backward()
            optimizer.step()
            total_mtt_mask_loss += loss.detach().item()
            # print(loss.item(), (mtt_mask_single.round()==1).sum().item(), (mtt_mask_label.round()==1).sum().item())
            # if step % 100 == 0:
            #     valid_loss_mean = evaluation(qt_net, None, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=2, mtt_mask_net=mtt_mask_net)

            if (step + 1) % 100 == 0:
                print("iter:%d\t mtt_mask_loss:%.5f" % (step, loss))
            del batch, qt_pred_list, p0_flow, p1_flow, mtt_mask_list
            if (step + 1) % 500 == 0:
                gc.collect()

            if opt['log']['use_tensor_board']:
                if (step + 1) % 100 == 0:
                    writer.add_scalar('loss/100step_loss', total_mtt_mask_loss / 100, global_iter)  # 统计每10个epoch的loss
                    total_mtt_mask_loss = 0
            global_iter += 1

        if (epoch + 1) % 2 == 0:
            state = {'mtt_mask_net': mtt_mask_net.state_dict()}
            torch.save(state, os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth'))
            out_model_path = os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth')

        if ((epoch + 1) % 1 == 0) and opt['datasets']['val']['use_valid']:
            valid_loss = []
            valid_loss_mean = evaluation(qt_net, None, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=2, mtt_mask_net=mtt_mask_net)
            # valid_loss_mean = evaluation(qt_net, None, epoch + base_epoch, valid_data, writer, global_iter, None, mtt_mask_net=mtt_mask_net)

            print("Evaluation: loss=%.4f" % (valid_loss_mean))
            valid_loss.append(valid_loss_mean.item())

        print("learning rate : %.3e" % scheduler.get_last_lr()[0])
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_iter)
        scheduler.step()
        gc.collect()

    return global_iter, out_model_path, base_epoch + epoch + 1, scheduler.get_last_lr()[0]


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B * N, C)

    x[idx1.reshape(-1)] = x1.reshape(B * N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B * N2, C)

    x = x.reshape(B, N, C)
    return x


def train_mtt_depth_net(qp, writer, init_global_iter, qt_model_path, mtt_mask_model_path, mtt_depth_model_path, dataset_id_list, sub_epoch=5, base_epoch=0, lr=1e-3):
    ds = 1
    if opt['network']['large_qt_model']:
        from models.backbone_large import QT_Net_HLG
        if qp == 37:
            guide = False
        else:
            guide = True
        qt_net = QT_Net_HLG(spp=False, qml=True, tml=True, guide=guide)
    else:
        from models.backbone import QT_Net_HLG
        qt_net = QT_Net_HLG(qml=opt['network']['qml'], guide=True, text_fe=opt['network']['text_fe'])
    flow_net = ME_Spynet(me_model_dir=opt['path']['me_model_dir'])
    mtt_mask_net = MTT_mask_net(qml=opt['network']['qml'], dlm=opt['network']['mtt_mask_dlm'])
    mtt_depth_model = MTT_Net_HLG(qml=opt['network']['qml'], residual_type=opt['network']['residual_type'], max_depth=3)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            devices_id = list(range(opt['gpu_num']))
            qt_net = qt_net.cuda(device=devices_id[0])
            qt_net = torch.nn.DataParallel(qt_net, device_ids=devices_id)
            flow_net = flow_net.cuda(device=devices_id[0])
            flow_net = torch.nn.DataParallel(flow_net, device_ids=devices_id)
            mtt_mask_net = mtt_mask_net.cuda(device=devices_id[0])
            mtt_mask_net = torch.nn.DataParallel(mtt_mask_net, device_ids=devices_id)
            mtt_depth_model = mtt_depth_model.cuda(device=devices_id[0])
            mtt_depth_model = torch.nn.DataParallel(mtt_depth_model, device_ids=devices_id)
        else:
            qt_net = qt_net.cuda()
            flow_net = flow_net.cuda()
            mtt_mask_net = mtt_mask_net.cuda()
            mtt_depth_model = mtt_depth_model.cuda()

    print("loaded qt_model weight from ", qt_model_path)
    if torch.cuda.device_count() > 1:
        qt_net.load_state_dict(torch.load(qt_model_path)['qt_net'])
    else:
        qt_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(qt_model_path)['qt_net'].items()})

    if mtt_mask_model_path is not None:
        print("loaded mtt_mask model weight from ", mtt_mask_model_path)
        if torch.cuda.device_count() > 1:
            mtt_mask_net.load_state_dict(torch.load(mtt_mask_model_path)['mtt_mask_net'])
        else:
            mtt_mask_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(mtt_mask_model_path)['mtt_mask_net'].items()})
    else:
        raise Exception("no weight.")

    if mtt_depth_model_path is not None:
        print("loaded mtt_depth model weight from ", mtt_depth_model_path)
        if torch.cuda.device_count() > 1:
            mtt_depth_model.load_state_dict(torch.load(mtt_depth_model_path)['mtt_depth_model'])
        else:
            mtt_depth_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(mtt_depth_model_path)['mtt_mask_net'].items()})

    flow_net = flow_net.eval()
    qt_net = qt_net.eval()
    mtt_mask_net = mtt_mask_net.eval()
    # criterion and optimizer
    # l1_criterion = torch.nn.L1Loss()
    params_list = [{'params': mtt_depth_model.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, sub_epoch + 1, 1)), gamma=opt['train']['gamma'])

    global_iter = init_global_iter
    print('start training ...')
    dataset_id = -1

    # load valid dataset
    if opt['datasets']['val']['use_valid']:
        valid_dataset = ValidDataset(qp, sub_id=opt['datasets']['val']['valid_id'], mode='valid', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
        valid_data = DataLoader(valid_dataset, batch_size=opt['datasets']['val']['batchSize'], shuffle=False, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
        print("loaded " + opt['datasets']['val']['valid_id'] + " valid dataset")
    # warmup
    if opt['train']['warmup']:
        print('warmup...')
        ori_dataset_id_list = dataset_id_list
        dataset_id_list = ['B', 'A', 'C']

    total_mtt_depth_loss = 0
    for epoch in range(1, sub_epoch + 1):
        mtt_depth_model = mtt_depth_model.train()
        total_mtt_mask_loss, total_loss = 0, 0
        if opt['train']['warmup'] and epoch == 16:
            dataset_id_list = ori_dataset_id_list
            print('warmup ending...')
        if epoch % 1 == 0 or epoch == 1:
            dataset_id = (dataset_id + 1) % len(dataset_id_list)
        if opt['train']['paired'] or (epoch % 1 == 0 or epoch == 1):
            train_dataset = MyDataset(qp, dataset_id_list[dataset_id], mode='train', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
            batchSize = opt['datasets']['train']['batchSize']
            # batchSize for multi-scale training, but it compromised the performance
            if not opt['train']['crop_A']:
                if dataset_id_list[dataset_id] == 'A':
                    # batchSize = opt['datasets']['train']['batchSize'] // 3 - 4
                    batchSize = opt['datasets']['train']['batchSize']
                elif dataset_id_list[dataset_id] == 'C':
                    batchSize = opt['datasets']['train']['batchSize']
            print("batchsize: ", batchSize)

            if opt['train']['paired']:
                shuffle = False
            else:
                shuffle = True

            if opt['datasets']['train']['num_workers'] > 0:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'], multiprocessing_context='fork')
            else:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
            print('loaded ' + dataset_id_list[dataset_id] + ' train dataset')

        for step, batch in enumerate(train_data):
            qt_net = qt_net.eval()
            mtt_depth_model = mtt_depth_model.train()
            optimizer.zero_grad()
            label_batch_list = [batch['label_' + str(depth)] for depth in range(4)]
            # label_batch_list = [ele.cuda() for ele in label_batch_list]
            i_frame, p0_frame, p1_frame = batch['i_frame'], batch['p0_frame'], batch['p1_frame']
            i_frame, p0_frame, p1_frame = i_frame[:, None, ::ds, ::ds], p0_frame[:, None, ::ds, ::ds], p1_frame[:, None, ::ds, ::ds]
            qp_list = batch['qp_list']
            if opt['gpu_num'] > 1:
                # i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[2]).float(), p1_frame.cuda(device=devices_id[3]).float()
                i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[2]).float(), p1_frame.cuda(device=devices_id[3]).float()
            else:
                i_frame, p0_frame, p1_frame = i_frame.cuda().float(), p0_frame.cuda().float(), p1_frame.cuda().float()

            label_batch_list = [item.cuda().float() for item in label_batch_list]
            qp_list = qp_list.cuda().float()
            # qt_net inference
            with torch.no_grad():
                if opt['network']['large_qt_model']:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')

                    # p0_flow = [ele.cuda(device=devices_id[2]) for ele in p0_flow]
                    # p1_flow = [ele.cuda(device=devices_id[3]) for ele in p1_flow]

                    p0_flow, p1_flow = p0_flow[-1].cuda(device=devices_id[2]), p1_flow[-1].cuda(device=devices_id[3])
                    # p0_flow, p1_flow = p0_flow[-1].cuda(), p1_flow[-1].cuda()

                    # p0_flow, p1_flow = p0_flow[-1], p1_flow[-1]
                    if opt['train']['crop_A'] and dataset_id_list[dataset_id] == 'A':
                        i_frame, p0_frame, p1_frame = i_frame[:, :, 128:1152, 128:2048], p0_frame[:, :, 128:1152, 128:2048], p1_frame[:, :, 128:1152, 128:2048]
                        p0_flow = [ele[:, :, 128 // 2 ** (2 - i):ele.shape[-2] - 128 // 2 ** (2 - i), 128 // 2 ** (2 - i):ele.shape[-1] - 128 // 2 ** (2 - i)] for i, ele in enumerate(p0_flow)]
                        p1_flow = [ele[:, :, 128 // 2 ** (2 - i):ele.shape[-2] - 128 // 2 ** (2 - i), 128 // 2 ** (2 - i):ele.shape[-1] - 128 // 2 ** (2 - i)] for i, ele in enumerate(p1_flow)]

                    # # 直接将p0_flow, p1_flow拼接，但是训练中为了节省显存，需要将p0_flow, p1_flow放在不同的设备上
                    # qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, torch.stack([p0_flow, p1_flow], dim=1), qp=qp_list[:, 0:1],\
                    #     trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=0)

                    qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, None, qp=qp_list[:, 0:1], trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=0, p0_flow=p0_flow, p1_flow=p1_flow)

                else:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt_mask')
                    qt_pred_list, _ = qt_net(i_frame[:, 0:1, ::2, ::2] / 255.0, torch.stack([p0_flow[0], p1_flow[0]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2] / 255.0, p1_frame=p1_frame[:, 0:1, ::2, ::2] / 255.0, out_medium_feat=True, upsample=4)
                qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
                qt_pred = qt_pred_list[-1]

                # mtt_mask_list = mtt_mask_net(i_frame[:, 0:1, ::2, ::2], p0_flow=p0_flow[1], p1_flow=p1_flow[1], qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2], p1_frame=p1_frame[:, 0:1, ::2, ::2])
                # mtt_mask = mtt_mask_list[-1]

                # # mtt_mask_label = where(Qt_depth_pred < QT_depth + MT_depth)
                sum_mtt_weight = label_batch_list[1][:, 0] + label_batch_list[2][:, 0] + label_batch_list[3][:, 0]
                sum_mtt_weight = ((F.max_pool2d(sum_mtt_weight.float(), kernel_size=4) + label_batch_list[0]).cuda() > qt_pred_list[-1][:, 0].round())
                mtt_mask_label = rearrange(rearrange(sum_mtt_weight, 'b (hi h) (wi w) -> b hi h wi w', h=8, w=8), 'b hi h wi w -> b hi wi h w').sum(dim=(3, 4)).cuda().float()
                mtt_mask = torch.clip(mtt_mask_label, min=0, max=1)
                mtt_mask += torch.rand_like(mtt_mask) * 0.1  # mtt_mask += torch.normal(mean=0, std=0.5, size=mtt_mask.shape).to(mtt_mask.device)

            i_frame.requires_grad, p0_frame.requires_grad, p1_frame.requires_grad = True, True, True
            qp_list.requires_grad = True
            mtt_mask.requires_grad = True

            mtt_depth_map_list, ctu_decision = mtt_depth_model(luma=i_frame[:, 0:1] / 255.0, p0_flow=p0_flow, p1_flow=p1_flow, qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame / 255.0, p1_frame=p1_frame / 255.0, ctu_decision=mtt_mask, mask_ratio=mask_ratio_g[dataset_id_list[dataset_id]])

            mtt_depth_loss = 0
            zero_rate_list, zero_label_list, single_loss = [], [], []
            for layer_depth, mtt_depth_map in enumerate(mtt_depth_map_list):
                mtt_depth_label = rearrange(rearrange(label_batch_list[layer_depth + 1][:, 0].long(), 'b (hi h) (wi w) -> b hi h wi w', h=32, w=32), 'b hi h wi w -> b (hi wi) (h w)')
                mtt_depth_label = batch_index_select(x=mtt_depth_label, idx=ctu_decision)
                mtt_depth_label = rearrange(mtt_depth_label, 'b n (h w) -> b n h w', h=32, w=32)
                # 使用ce_loss 或者 focal loss决定mtt_depth_map
                if opt['train']['focal_loss'][layer_depth]:
                    if opt['train']['open_alpha']:
                        weight_list_b = [1 / max((torch.sum(torch.round(mtt_depth_label.float()) == i).item() / float(mtt_depth_label.numel())), opt['train']['min_ratio'][layer_depth]) for i in range(3)]
                        weight_list_b_sum = sum(weight_list_b)
                        weight_list_b = [ele / weight_list_b_sum for ele in weight_list_b]
                    else:
                        weight_list_b = [1, 1, 1]
                    focal_loss_b = MultiClassFocalLossWithAlpha(alpha=weight_list_b, gamma=opt['train']['focal_gamma'][layer_depth])
                    mtt_depth_loss_single = focal_loss_b(mtt_depth_map.reshape(-1, 3), mtt_depth_label.reshape(-1))
                else:
                    mtt_depth_loss_single = F.cross_entropy(mtt_depth_map.reshape(-1, 3), mtt_depth_label.reshape(-1))  # mtt_depth_loss += ce_criterion(mtt_depth_map.flatten(), binary_threshold(mtt_mask_label.flatten(), threshold=0.1).float())
                mtt_depth_loss += mtt_depth_loss_single
                if (step + 1) % 500 == 0:
                    zero_rate_list.append(100 * torch.sum(torch.argmax(mtt_depth_map.reshape(-1, 3), dim=-1).int() == 0) / float(mtt_depth_label.numel()))
                    zero_label_list.append(100 * torch.sum(mtt_depth_label.int() == 0) / float(mtt_depth_label.numel()))
                    single_loss.append(mtt_depth_loss_single.item())

            loss = mtt_depth_loss
            loss.backward()
            optimizer.step()
            total_mtt_depth_loss += loss.detach().item()
            # print(loss.item(), (torch.argmax(mtt_depth_map_list[0].reshape(-1, 3), dim=-1)==0).sum().item(), (label_batch_list[1][:,0].round()==1).sum().item())
            # if step % 100 == 0:
            #     valid_loss_mean = evaluation(qt_net, None, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=2, mtt_mask_net=mtt_mask_net)

            if (step + 1) % 500 == 0:
                print("iter:%d\t mtt_depth_loss:%.5f [b] %.5f %.5f %.5f [zero_rate] %.2f %.2f %.2f [zero_label] %.2f %.2f %.2f " % (step, loss, single_loss[0], single_loss[1], single_loss[2], zero_rate_list[0], zero_rate_list[1], zero_rate_list[2], zero_label_list[0], zero_label_list[1], zero_label_list[2]))

            # if (step + 1) % 10 == 0:
            #     # validation test
            #     valid_loss_mean = evaluation(qt_net, mtt_depth_model, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=3, mtt_mask_net=mtt_mask_net,)

            del batch, qt_pred_list, p0_flow, p1_flow, mtt_depth_map_list, ctu_decision
            if (step + 1) % 500 == 0:
                gc.collect()

            if opt['log']['use_tensor_board']:
                if (step + 1) % 100 == 0:
                    writer.add_scalar('loss/100step_loss', total_mtt_depth_loss / 100, global_iter)  # 统计每10个epoch的loss
                    total_mtt_depth_loss = 0
            global_iter += 1

        if (epoch + 1) % 1 == 0:
            state = {'mtt_depth_model': mtt_depth_model.state_dict()}
            torch.save(state, os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth'))
            out_model_path = os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth')

        if ((epoch + 1) % 1 == 0) and opt['datasets']['val']['use_valid']:
            valid_loss = []
            valid_loss_mean = evaluation(qt_net, mtt_depth_model, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=3, mtt_mask_net=mtt_mask_net, )

            print("Evaluation: loss=%.4f" % (valid_loss_mean))
            valid_loss.append(valid_loss_mean.item())

        print("learning rate : %.3e" % scheduler.get_last_lr()[0])
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_iter)
        scheduler.step()
        gc.collect()

    return global_iter, out_model_path, base_epoch + epoch + 1, scheduler.get_last_lr()[0]


def train_mtt_direction_net(qp, writer, init_global_iter, qt_model_path, mtt_mask_model_path, mtt_depth_model_path, dataset_id_list, sub_epoch=5, base_epoch=0, lr=1e-3):
    ds = 1
    if opt['network']['large_qt_model']:
        from models.backbone_large import QT_Net_HLG
        if qp == 37:
            guide = False
        else:
            guide = True
        qt_net = QT_Net_HLG(spp=False, qml=True, tml=True, guide=guide)
    else:
        from models.backbone import QT_Net_HLG
        qt_net = QT_Net_HLG(qml=opt['network']['qml'], guide=True, text_fe=opt['network']['text_fe'])
    flow_net = ME_Spynet(me_model_dir=opt['path']['me_model_dir'])
    mtt_mask_net = MTT_mask_net(qml=opt['network']['qml'], dlm=opt['network']['mtt_mask_dlm'])
    mtt_depth_model = MTT_Net_HLG(qml=opt['network']['qml'], residual_type=opt['network']['residual_type'], max_depth=3)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            devices_id = list(range(opt['gpu_num']))
            qt_net = qt_net.cuda(device=devices_id[0])
            qt_net = torch.nn.DataParallel(qt_net, device_ids=devices_id)
            flow_net = flow_net.cuda(device=devices_id[0])
            flow_net = torch.nn.DataParallel(flow_net, device_ids=devices_id)
            mtt_mask_net = mtt_mask_net.cuda(device=devices_id[0])
            mtt_mask_net = torch.nn.DataParallel(mtt_mask_net, device_ids=devices_id)
            mtt_depth_model = mtt_depth_model.cuda(device=devices_id[0])
            mtt_depth_model = torch.nn.DataParallel(mtt_depth_model, device_ids=devices_id)
        else:
            qt_net = qt_net.cuda()
            flow_net = flow_net.cuda()
            mtt_mask_net = mtt_mask_net.cuda()
            mtt_depth_model = mtt_depth_model.cuda()

    print("loaded qt_model weight from ", qt_model_path)
    if torch.cuda.device_count() > 1:
        qt_net.load_state_dict(torch.load(qt_model_path)['qt_net'])
    else:
        qt_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(qt_model_path)['qt_net'].items()})

    if mtt_mask_model_path is not None:
        print("loaded mtt_mask model weight from ", mtt_mask_model_path)
        if torch.cuda.device_count() > 1:
            mtt_mask_net.load_state_dict(torch.load(mtt_mask_model_path)['mtt_mask_net'])
        else:
            mtt_mask_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(mtt_mask_model_path)['mtt_mask_net'].items()})
    else:
        raise Exception("no weight.")

    if mtt_depth_model_path is not None:
        print("loaded mtt_depth model weight from ", mtt_depth_model_path)
        if torch.cuda.device_count() > 1:
            mtt_depth_model.load_state_dict(torch.load(mtt_depth_model_path)['mtt_dire_model'])
        else:
            mtt_depth_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(mtt_depth_model_path)['mtt_mask_net'].items()})

    flow_net = flow_net.eval()
    qt_net = qt_net.eval()
    mtt_mask_net = mtt_mask_net.eval()
    # criterion and optimizer
    # l1_criterion = torch.nn.L1Loss()
    params_list = [{'params': mtt_depth_model.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, sub_epoch + 1, 1)), gamma=opt['train']['gamma'])

    global_iter = init_global_iter
    print('start training ...')
    dataset_id = -1

    # load valid dataset
    if opt['datasets']['val']['use_valid']:
        valid_dataset = ValidDataset(qp, sub_id=opt['datasets']['val']['valid_id'], mode='valid', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
        valid_data = DataLoader(valid_dataset, batch_size=opt['datasets']['val']['batchSize'], shuffle=False, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
        print("loaded " + opt['datasets']['val']['valid_id'] + " valid dataset")
    # warmup
    if opt['train']['warmup']:
        print('warmup...')
        ori_dataset_id_list = dataset_id_list
        dataset_id_list = ['B', 'A', 'C']

    total_mtt_depth_loss = 0
    for epoch in range(1, sub_epoch + 1):
        mtt_depth_model = mtt_depth_model.train()
        total_mtt_mask_loss, total_loss = 0, 0
        if opt['train']['warmup'] and epoch == 16:
            dataset_id_list = ori_dataset_id_list
            print('warmup ending...')
        if epoch % 1 == 0 or epoch == 1:
            dataset_id = (dataset_id + 1) % len(dataset_id_list)
        if opt['train']['paired'] or (epoch % 1 == 0 or epoch == 1):
            train_dataset = MyDataset(qp, dataset_id_list[dataset_id], mode='train', dataset_dir=opt['path']['train_dataset_dir'], enc_mode=opt['enc_mode'], split_id=dataset_id)
            batchSize = opt['datasets']['train']['batchSize']
            # batchSize for multi-scale training, but it compromised the performance
            if not opt['train']['crop_A']:
                if dataset_id_list[dataset_id] == 'A':
                    # batchSize = opt['datasets']['train']['batchSize'] // 3 - 4
                    batchSize = opt['datasets']['train']['batchSize']
                elif dataset_id_list[dataset_id] == 'C':
                    batchSize = opt['datasets']['train']['batchSize']
            print("batchsize: ", batchSize)

            if opt['train']['paired']:
                shuffle = False
            else:
                shuffle = True

            if opt['datasets']['train']['num_workers'] > 0:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'], multiprocessing_context='fork')
            else:
                train_data = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle, num_workers=opt['datasets']['train']['num_workers'], pin_memory=opt['datasets']['train']['pin_memory'])
            print('loaded ' + dataset_id_list[dataset_id] + ' train dataset')

        for step, batch in enumerate(train_data):
            qt_net = qt_net.eval()
            mtt_depth_model = mtt_depth_model.train()
            optimizer.zero_grad()
            label_batch_list = [batch['label_' + str(depth)] for depth in range(4)]
            # label_batch_list = [ele.cuda() for ele in label_batch_list]
            i_frame, p0_frame, p1_frame = batch['i_frame'], batch['p0_frame'], batch['p1_frame']
            i_frame, p0_frame, p1_frame = i_frame[:, None, ::ds, ::ds], p0_frame[:, None, ::ds, ::ds], p1_frame[:, None, ::ds, ::ds]
            qp_list = batch['qp_list']
            if opt['gpu_num'] > 1:
                # i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[2]).float(), p1_frame.cuda(device=devices_id[3]).float()
                i_frame, p0_frame, p1_frame = i_frame.cuda(device=devices_id[1]).float(), p0_frame.cuda(device=devices_id[2]).float(), p1_frame.cuda(device=devices_id[3]).float()
            else:
                i_frame, p0_frame, p1_frame = i_frame.cuda().float(), p0_frame.cuda().float(), p1_frame.cuda().float()

            label_batch_list = [item.cuda().float() for item in label_batch_list]
            qp_list = qp_list.cuda().float()
            # qt_net inference
            with torch.no_grad():
                if opt['network']['large_qt_model']:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt')
                    p0_flow, p1_flow = p0_flow[-1].cuda(device=devices_id[2]), p1_flow[-1].cuda(device=devices_id[3])

                    if opt['train']['crop_A'] and dataset_id_list[dataset_id] == 'A':
                        i_frame, p0_frame, p1_frame = i_frame[:, :, 128:1152, 128:2048], p0_frame[:, :, 128:1152, 128:2048], p1_frame[:, :, 128:1152, 128:2048]
                        p0_flow = [ele[:, :, 128 // 2 ** (2 - i):ele.shape[-2] - 128 // 2 ** (2 - i), 128 // 2 ** (2 - i):ele.shape[-1] - 128 // 2 ** (2 - i)] for i, ele in enumerate(p0_flow)]
                        p1_flow = [ele[:, :, 128 // 2 ** (2 - i):ele.shape[-2] - 128 // 2 ** (2 - i), 128 // 2 ** (2 - i):ele.shape[-1] - 128 // 2 ** (2 - i)] for i, ele in enumerate(p1_flow)]
                    qt_pred_list, _, _ = qt_net(i_frame[:, 0:1] / 255.0, None, qp=qp_list[:, 0:1], trans_flow_DAM=True, make_res=True, p0_frame=p0_frame[:, 0:1] / 255.0, p1_frame=p1_frame[:, 0:1] / 255.0, out_medium_feat=True, upsample=0, p0_flow=p0_flow, p1_flow=p1_flow)

                else:
                    p0_flow, p1_flow = dataset2dataset(i_frame, p0_frame, p1_frame, flow_net, ds=1, flag='mtt_mask')
                    qt_pred_list, _ = qt_net(i_frame[:, 0:1, ::2, ::2] / 255.0, torch.stack([p0_flow[0], p1_flow[0]], dim=1), qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame[:, 0:1, ::2, ::2] / 255.0, p1_frame=p1_frame[:, 0:1, ::2, ::2] / 255.0, out_medium_feat=True, upsample=4)
                qt_pred_list = [ele * 4 - 0.5 for ele in qt_pred_list]
                qt_pred = qt_pred_list[-1]

                sum_mtt_weight = label_batch_list[1][:, 0] + label_batch_list[2][:, 0] + label_batch_list[3][:, 0]
                sum_mtt_weight = ((F.max_pool2d(sum_mtt_weight.float(), kernel_size=4) + label_batch_list[0]).cuda() > qt_pred_list[-1][:, 0].round())
                mtt_mask_label = rearrange(rearrange(sum_mtt_weight, 'b (hi h) (wi w) -> b hi h wi w', h=8, w=8), 'b hi h wi w -> b hi wi h w').sum(dim=(3, 4)).cuda().float()
                mtt_mask = torch.clip(mtt_mask_label, min=0, max=1)
                mtt_mask += torch.rand_like(mtt_mask) * 0.1  # mtt_mask += torch.normal(mean=0, std=0.5, size=mtt_mask.shape).to(mtt_mask.device)

            i_frame.requires_grad, p0_frame.requires_grad, p1_frame.requires_grad = True, True, True
            qp_list.requires_grad = True
            mtt_mask.requires_grad = True

            mtt_depth_map_list, ctu_decision = mtt_depth_model(luma=i_frame[:, 0:1] / 255.0, p0_flow=p0_flow, p1_flow=p1_flow, qt_pred=qt_pred, qp=qp_list[:, 0:1], trans_flow_DAM=True, p0_frame=p0_frame / 255.0, p1_frame=p1_frame / 255.0, ctu_decision=mtt_mask, mask_ratio=mask_ratio_g[dataset_id_list[dataset_id]])

            mtt_depth_loss = 0
            zero_rate_list, zero_label_list, single_loss = [], [], []
            ver_rate_list, ver_label_list, hor_rate_list, hor_label_list = [], [], [], []
            for layer_depth, mtt_depth_map in enumerate(mtt_depth_map_list):
                mtt_depth_label = rearrange(rearrange(label_batch_list[layer_depth + 1][:, 1].long() + 1, 'b (hi h) (wi w) -> b hi h wi w', h=32, w=32), 'b hi h wi w -> b (hi wi) (h w)')
                mtt_depth_label = batch_index_select(x=mtt_depth_label, idx=ctu_decision)
                mtt_depth_label = rearrange(mtt_depth_label, 'b n (h w) -> b n h w', h=32, w=32)
                if opt['train']['focal_loss'][layer_depth]:
                    if opt['train']['open_alpha']:
                        weight_list_b = [1 / max((torch.sum(torch.round(mtt_depth_label.float()) == i).item() / float(mtt_depth_label.numel())), opt['train']['min_ratio'][layer_depth]) for i in range(3)]
                        weight_list_b_sum = sum(weight_list_b)
                        weight_list_b = [ele / weight_list_b_sum for ele in weight_list_b]
                    else:
                        weight_list_b = [1, 1, 1]
                    focal_loss_b = MultiClassFocalLossWithAlpha(alpha=weight_list_b, gamma=opt['train']['focal_gamma'][layer_depth])
                    mtt_depth_loss_single = focal_loss_b(mtt_depth_map.reshape(-1, 3), mtt_depth_label.reshape(-1))
                else:
                    mtt_depth_loss_single = F.cross_entropy(mtt_depth_map.reshape(-1, 3), mtt_depth_label.reshape(-1))  # mtt_depth_loss += ce_criterion(mtt_depth_map.flatten(), binary_threshold(mtt_mask_label.flatten(), threshold=0.1).float())
                mtt_depth_loss += mtt_depth_loss_single
                if (step + 1) % 500 == 0:
                    zero_rate_list.append(100 * torch.sum(torch.argmax(mtt_depth_map.reshape(-1, 3), dim=-1).int() == 1) / float(mtt_depth_label.numel()))
                    zero_label_list.append(100 * torch.sum(mtt_depth_label.int() == 1) / float(mtt_depth_label.numel()))
                    ver_rate_list.append(100 * torch.sum(torch.argmax(mtt_depth_map.reshape(-1, 3), dim=-1).int() == 0) / float(mtt_depth_label.numel()))
                    ver_label_list.append(100 * torch.sum(mtt_depth_label.int() == 0) / float(mtt_depth_label.numel()))
                    single_loss.append(mtt_depth_loss_single.item())

            loss = mtt_depth_loss
            loss.backward()
            optimizer.step()
            total_mtt_depth_loss += loss.detach().item()

            if (step + 1) % 500 == 0:
                print("iter:%d\t mtt_depth_loss:%.5f [b] %.5f %.5f %.5f [zero_rate] %.2f %.2f %.2f [zero_label] %.2f %.2f %.2f [ver_rate] %.2f %.2f %.2f [ver_label] %.2f %.2f %.2f " % (
                step, loss, single_loss[0], single_loss[1], single_loss[2], zero_rate_list[0], zero_rate_list[1], zero_rate_list[2], zero_label_list[0], zero_label_list[1], zero_label_list[2], ver_rate_list[0], ver_rate_list[1], ver_rate_list[2], ver_label_list[0], ver_label_list[1], ver_label_list[2]))

            del batch, qt_pred_list, p0_flow, p1_flow, mtt_depth_map_list, ctu_decision
            if (step + 1) % 500 == 0:
                gc.collect()

            if opt['log']['use_tensor_board']:
                if (step + 1) % 100 == 0:
                    writer.add_scalar('loss/100step_loss', total_mtt_depth_loss / 100, global_iter)  
                    total_mtt_depth_loss = 0
            global_iter += 1

        if (epoch + 1) % 1 == 0:
            state = {'mtt_dire_model': mtt_depth_model.state_dict()}
            torch.save(state, os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth'))
            out_model_path = os.path.join(opt['path']['cp_path'], 'model_qp' + str(opt['qp']) + '_epoch_' + str(epoch + base_epoch) + '.pth')

        if ((epoch + 1) % 1 == 0) and opt['datasets']['val']['use_valid']:
            valid_loss = []
            valid_loss_mean = evaluation(qt_net, mtt_depth_model, epoch + base_epoch, valid_data, writer, global_iter, flow_net, stage=3, mtt_mask_net=mtt_mask_net, dire=True)

            print("Evaluation: loss=%.4f" % (valid_loss_mean))
            valid_loss.append(valid_loss_mean.item())

        print("learning rate : %.3e" % scheduler.get_last_lr()[0])
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_iter)
        scheduler.step()
        gc.collect()

    return global_iter, out_model_path, base_epoch + epoch + 1, scheduler.get_last_lr()[0]


def train_joint():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('-tensor_writer_dir', type=str)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    print(option.dict2str(opt))

    if opt['log']['use_tensor_board']:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.tensor_writer_dir)
    else:
        writer = None

    if opt['path']['pretrained_path']:
        model_path = opt['path']['pretrained_path']
    else:
        model_path = None

    if opt['path']['pretrained_dire_path']:
        dire_model_path = opt['path']['pretrained_dire_path']
    else:
        dire_model_path = None

    stage = opt['train']['init_stage']
    lr = opt['train']['lr']
    ds = opt['ds']

    base_epoch = 0
    if stage == 0:
        # train qt-net
        ds = 4
        resample_dataset = True
        iter_count, model_path, base_epoch, lr = train_qt_net(opt['qp'], writer, dataset_id_list=opt['train']['resolution'], sub_epoch=opt['train']['sub_epoch'][0], base_epoch=base_epoch, lr=lr, init_global_iter=0, model_path=model_path)
    elif stage == 1:
        # train mtt-mask net
        resample_dataset = True
        if opt['network']['large_qt_model']:
            ds = 1
            qt_model_path = os.path.join(opt['path']['large_qt_model_path'], 'model_qp%d.pth' % opt['qp'])
        else:
            ds = 2
            qt_model_path = os.path.join(opt['path']['qt_model_path'], 'model_qp%d.pth' % opt['qp'])

        if opt['path']['pretrained_path'] is None:
            mtt_mask_model_path = None
        else:
            mtt_mask_model_path = opt['path']['pretrained_path']
        iter_count, model_path, base_epoch, lr = train_mtt_mask_net(opt['qp'], writer, dataset_id_list=opt['train']['resolution'], sub_epoch=opt['train']['sub_epoch'][1], base_epoch=base_epoch, lr=lr, mtt_mask_model_path=mtt_mask_model_path, qt_model_path=qt_model_path, init_global_iter=0)
    elif stage == 2:

        mask_ratio_g = {'A': opt['train']['mask_ratio'][0], 'B': opt['train']['mask_ratio'][1], 'C': opt['train']['mask_ratio'][2], }

        ds = 1
        resample_dataset = False
        if opt['network']['large_qt_model']:
            qt_model_path = os.path.join(opt['path']['large_qt_model_path'], 'model_qp%d.pth' % opt['qp'])
        else:
            qt_model_path = os.path.join(opt['path']['qt_model_path'], 'model_qp%d.pth' % opt['qp'])

        mtt_mask_model_path = os.path.join(opt['path']['mtt_mask_model_path'], 'model_qp%d.pth' % opt['qp'])

        if opt['path']['pretrained_path'] is None:
            mtt_depth_model_path = None
        else:
            mtt_depth_model_path = opt['path']['pretrained_path']

        if opt['network']['model_type'] == 'dire':
            iter_count, model_path, base_epoch, lr = train_mtt_direction_net(opt['qp'], writer, dataset_id_list=opt['train']['resolution'], sub_epoch=opt['train']['sub_epoch'][1], base_epoch=base_epoch, lr=lr, mtt_mask_model_path=mtt_mask_model_path, qt_model_path=qt_model_path, init_global_iter=0,
                                                                             mtt_depth_model_path=mtt_depth_model_path)
        else:
            iter_count, model_path, base_epoch, lr = train_mtt_depth_net(opt['qp'], writer, dataset_id_list=opt['train']['resolution'], sub_epoch=opt['train']['sub_epoch'][1], base_epoch=base_epoch, lr=lr, mtt_mask_model_path=mtt_mask_model_path, qt_model_path=qt_model_path, init_global_iter=0,
                                                                         mtt_depth_model_path=mtt_depth_model_path)
    elif stage == 3:
        train_mtt_direction_net()
    elif stage == 4:
        train_joint()