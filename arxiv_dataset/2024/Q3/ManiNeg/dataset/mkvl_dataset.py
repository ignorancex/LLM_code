import os
import re
import random

import numpy as np
import pandas as pd
from PIL import Image

import SimpleITK as sitk
import torch
from transformers import BertTokenizer

# from ALBEF.dataset.utils import pre_caption

VIEW_MAP = {"RCC": 1, "LCC": 2, "LMLO": 3, "LMO": 3, "RMLO": 4, "RMO": 4}
DEL_VIEWS = ["无", "无资料", "不会", np.nan]
MALIGNANT = {"恶性": 1, "良性": 0}

MAM_MAX_PIXEL, MAM_MIN_PIXEL = 4095, 0


class HMBMPretrainDataset:

    def __init__(self,
                 db_root_path,
                 dcm_dir,
                 anno_file,
                 stage="train",
                 transform=None,
                 text_transform=None,
                 output_modal='dal',
                 # dal: double view, attr, language
                 # val: single view, attr language
                 t_backbone='bert-base-chinese',
                 pre_downsample=True,
                 attr_noise=None,
                 partial_data=None
                 ):
        self.attr_noise = attr_noise
        self.pre_downsample = pre_downsample
        if t_backbone is not None:
            self.tokenizer = BertTokenizer.from_pretrained(t_backbone)
        self.output_modal = output_modal
        self.db_root_path = db_root_path
        self.dcm_dir = dcm_dir
        self.anno_file = anno_file

        self.stage = stage
        self.partial_data = partial_data
        attr_anno = pd.read_excel(os.path.join(db_root_path, f"annotation/{stage}_{anno_file}"))

        self.patient_ids = attr_anno["patient_id"].to_list()
        self.study_ids = attr_anno["study_id"].to_list()
        self.labels = attr_anno["label"].to_list()
        self.attrs = attr_anno.iloc[:, 8: -1].to_numpy()

        # generating two different view, if only one view in each study, we generate two views by
        # data augmention
        view_ids = attr_anno["view_id"].to_list()
        view_ids = [v.split("||")[: -1] for v in view_ids]
        for i in range(len(view_ids)):
            for j in range(len(view_ids[i])):
                view_ids[i][j] = random.sample(view_ids[i][j].split("/")[: -1], 1)[0]
        self.view_ids = view_ids
        for i in range(len(self.view_ids)):
            if len(self.view_ids[i]) < 2:
                self.view_ids[i].append(self.view_ids[i][0])

        reports = attr_anno["诊断结果"].to_list()
        for i in range(len(reports)):
            reports[i] = "".join(reports[i].split("\n"))
        self.reports = reports

        self.transform = transform

    def __len__(self):
        if self.partial_data is not None:
            return int(len(self.patient_ids)*self.partial_data)
        else:
            return len(self.patient_ids)

    def __getitem__(self, item):
        view_dir = os.path.join(
            self.db_root_path, self.dcm_dir, self.patient_ids[item], self.study_ids[item]
        )
        ret = {}
        if 'v' in self.output_modal or 'd' in self.output_modal:
            if self.pre_downsample:
                view1 = Image.open(os.path.join(view_dir, f"{self.view_ids[item][0]}.a.png"))
            else:
                view1 = sitk.ReadImage(os.path.join(view_dir, self.view_ids[item][0]))
                view1 = sitk.GetArrayFromImage(view1)
                # view1 = Image.fromarray(view1[0]).convert("RGB")
                view1 = (view1 - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
                view1 = (view1 * 255).astype(np.uint8)
                view1 = np.concatenate([view1, view1, view1], axis=0).transpose((1, 2, 0))
                view1 = Image.fromarray(view1)

            if self.transform:
                view1 = self.transform(view1)
            ret['view1'] = view1
            if 'p' in self.output_modal:
                ret['view1_path'] = os.path.join(view_dir, f"{self.view_ids[item][0]}.a.png")
        if 'd' in self.output_modal:
            if self.pre_downsample:
                view2 = Image.open(os.path.join(view_dir, f"{self.view_ids[item][1]}.a.png"))
            else:
                view2 = sitk.ReadImage(os.path.join(view_dir, self.view_ids[item][1]))
                view2 = sitk.GetArrayFromImage(view2)
                # view1 = Image.fromarray(view1[0]).convert("RGB")
                view2 = (view2 - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
                view2 = (view2 * 255).astype(np.uint8)
                view2 = np.concatenate([view2, view2, view2], axis=0).transpose((1, 2, 0))
                view2 = Image.fromarray(view2)

            if self.transform:
                view2 = self.transform(view2)
            ret['view2'] = view2
            if 'p' in self.output_modal:
                ret['view2_path'] = os.path.join(view_dir, f"{self.view_ids[item][1]}.a.png")
        if 'l' in self.output_modal:
            if not hasattr(self, "tokenizer"):
                raise AttributeError("t_backbone is wrong.")
            report = self.reports[item].strip()

            text = self.tokenizer(
                report, padding='max_length', max_length=512, truncation=True, return_tensors="pt"
            )
            text = {k: v[0] for k, v in text.items()}
            ret.update(**text)
            if 'r' in self.output_modal:  # r must be with language
                ret['raw_text'] = report
        if 'a' in self.output_modal:
            attr = self.attrs[item]
            # attr = [abs(a - 0.1) for a in attr]  # xw: label smoothing?

            attr = torch.tensor(attr).float()
            if self.attr_noise is not None:
                noise = torch.rand(attr.size()) * self.attr_noise
                attr = torch.abs(attr - noise)
            ret['attr'] = attr

        ret['label'] = torch.tensor(MALIGNANT[self.labels[item]])
        return ret


# ====================== downstream task dataset =========================


class HMBMDownstreamDataset:

    def __init__(self,
                 db_root_path,
                 dcm_dir,
                 anno_file,
                 stage="train",
                 transform=None):

        self.db_root_path = db_root_path
        self.dcm_dir = dcm_dir
        self.anno_file = anno_file

        self.stage = stage

        attr_anno = pd.read_excel(os.path.join(db_root_path, f"annotation/{stage}_{anno_file}"))

        self.patient_ids = attr_anno["patient_id"].to_list()
        self.study_ids = attr_anno["study_id"].to_list()
        self.labels = attr_anno["label"].to_list()

        # generating two different view, if only one view in each study, we generate two views by
        # data augmention
        view_ids = attr_anno["view_id"].to_list()
        view_ids = [v.split("||")[: -1] for v in view_ids]
        for i in range(len(view_ids)):
            for j in range(len(view_ids[i])):
                view_ids[i][j] = random.sample(view_ids[i][j].split("/")[: -1], 1)[0]
        self.view_ids = view_ids
        for i in range(len(self.view_ids)):
            if len(self.view_ids[i]) < 2:
                self.view_ids[i].append(self.view_ids[i][0])

        reports = attr_anno["诊断结果"].to_list()
        for i in range(len(reports)):
            reports[i] = "".join(reports[i].split("\n"))
        self.reports = reports

        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, item):
        view_dir = os.path.join(
            self.db_root_path, self.dcm_dir, self.patient_ids[item], self.study_ids[item]
        )

        view1 = sitk.ReadImage(os.path.join(view_dir, self.view_ids[item][0]))
        view1 = sitk.GetArrayFromImage(view1)
        # view1 = Image.fromarray(view1[0]).convert("RGB")
        view1 = (view1 - MAM_MIN_PIXEL) / (MAM_MAX_PIXEL - MAM_MIN_PIXEL)
        view1 = (view1 * 255).astype(np.uint8)
        view1 = np.concatenate([view1, view1, view1], axis=0).transpose((1, 2, 0))
        view1 = Image.fromarray(view1)

        # view2 = sitk.ReadImage(os.path.join(view_dir, self.view_ids[item][2]))
        # view2 = sitk.GetArrayFromImage(view2)
        # view2 = Image.fromarray(view2[0]).convert("RGB")

        if self.transform:
            view1 = self.transform(view1)
            # view2 = self.transform(view2)

        # label = torch.tensor(MALIGNANT[self.labels[item]])
        if MALIGNANT[self.labels[item]] == 0:
            label = torch.tensor([0, 1])
        elif MALIGNANT[self.labels[item]] == 1:
            label = torch.tensor([1, 0])
        else:
            raise ValueError()
        report = self.reports[item].strip()
        #
        # print(view_dir)
        # print(self.view_ids[item])
        # print(label)
        # print(report)

        return view1, label
        # return view1, report, label


def main():
    db_root_path = "/data2/zhai/HMBM/"
    dcm_folder = "20230301/"
    anno_file = "20230301_final_onehot.xlsx"
    # dcm_folder = "20221123/"
    # anno_file = "20221123_.xlsx"

    import torchvision.transforms as T

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    aug = T.Compose([
        # T.Resize((256, 256)),
        T.RandomResizedCrop((256, 256), scale=(0.2, 1.)),
        T.RandomHorizontalFlip(),
        T.RandomRotation((-10, 10)),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize,
    ])
    aug = None

    hmbm_dataset = HMBMPretrainDataset(
        db_root_path, dcm_folder, anno_file, stage="train", transform=aug
    )
    print(len(hmbm_dataset))
    print(hmbm_dataset[0][0])
    print(hmbm_dataset[0][1])
    print(hmbm_dataset[0][2])


if __name__ == '__main__':
    main()
