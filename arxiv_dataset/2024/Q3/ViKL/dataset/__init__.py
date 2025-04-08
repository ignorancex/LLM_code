import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
import re
# from ALBEF.dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
# from ALBEF.dataset.nlvr_dataset import nlvr_dataset
# from ALBEF.dataset.ve_dataset import ve_dataset
# from ALBEF.dataset.vqa_dataset import vqa_dataset
# from ALBEF.dataset.grounding_dataset import grounding_dataset
# # from ALBEF.dataset.hmbm_dataset import HMBMPretrainDataset, HMBMDownstreamDataset, \
# #     HMBMRetrievalEvalDataset
# from ALBEF.experiments.ddsm_dataset import DDSMClassificationDataset
from dataset.ddsm_dataset_cls import DDSMClsDataset
from dataset.inbreast_dataset_cls import InBreastClsDataset
from dataset.mkvl_dataset import HMBMPretrainDataset, HMBMDownstreamDataset

from dataset.randaugment import RandomAugment


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption
# 构建预训练和下游任务数据集
def create_dataset(dataset, config):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )

    # 数据增强方式
    # 1. 预训练数据增强
    # pretrain_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(
    #         config['image_res'], scale=(v, 1.0), interpolation=Image.BICUBIC
    #     ),
    #     transforms.RandomHorizontalFlip(),
    #     RandomAugment(
    #         2, 7, isPIL=True,
    #         augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
    #               'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']
    #     ),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # 2. 下游任务数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config['image_res'], scale=(config['crop_min_scale'], 1.0), interpolation=Image.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2, 7, isPIL=True,
            augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']
        ),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(
            (config['image_res'], config['image_res']), interpolation=Image.BICUBIC
        ),
        transforms.ToTensor(),
        normalize,
    ])

    # 华美医院数据集按照标注日期由四个文件夹组成
    # 华美医院乳腺钼靶预训练数据集
    if dataset == 'hmbm':
        # dataset = pretrain_dataset(config['train_file'], pretrain_transform)

        # HwaMei Hospital Breast Mammography-Report-Attribution dataset
        dataset1 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20220901/",
            # "20220901_final.xlsx",
            "20220901_final_onehot.xlsx",
            stage="train",
            transform=train_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise'],
            partial_data=config['partial_data'],
        )
        dataset2 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20221123/",
            # "20221123_final.xlsx",
            "20221123_final_onehot.xlsx",
            stage="train",
            transform=train_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise'],
            partial_data=config['partial_data'],
        )
        dataset3 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230301/",
            # "20230301_final.xlsx",
            "20230301_final_onehot.xlsx",
            stage="train",
            transform=train_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise'],
            partial_data=config['partial_data'],
        )
        dataset4 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230519/",
            # "20230519_final.xlsx",
            "20230519_final_onehot.xlsx",
            stage="train",
            transform=train_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise'],
            partial_data=config['partial_data'],
        )
        datasets = ConcatDataset([dataset1, dataset2])
        datasets = ConcatDataset([datasets, dataset3])
        datasets = ConcatDataset([datasets, dataset4])

        test_dataset1 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20220901/",
            # "20220901_final.xlsx",
            "20220901_final_onehot.xlsx",
            stage="test",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        test_dataset2 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20221123/",
            # "20221123_final.xlsx",
            "20221123_final_onehot.xlsx",
            stage="test",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        test_dataset3 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230301/",
            # "20230301_final.xlsx",
            "20230301_final_onehot.xlsx",
            stage="test",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        test_dataset4 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230519/",
            # "20230519_final.xlsx",
            "20230519_final_onehot.xlsx",
            stage="test",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        test_datasets = ConcatDataset([test_dataset1, test_dataset2])
        test_datasets = ConcatDataset([test_datasets, test_dataset3])
        test_datasets = ConcatDataset([test_datasets, test_dataset4])

        val_dataset1 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20220901/",
            # "20220901_final.xlsx",
            "20220901_final_onehot.xlsx",
            stage="valid",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        val_dataset2 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20221123/",
            # "20221123_final.xlsx",
            "20221123_final_onehot.xlsx",
            stage="valid",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        val_dataset3 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230301/",
            # "20230301_final.xlsx",
            "20230301_final_onehot.xlsx",
            stage="valid",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        val_dataset4 = HMBMPretrainDataset(
            "/data2/zhai/HMBM/",
            "20230519/",
            # "20230519_final.xlsx",
            "20230519_final_onehot.xlsx",
            stage="valid",
            transform=test_transform,
            output_modal=config['modal'],
            t_backbone=config['t_backbone'],
            pre_downsample=config['pre_downsample'],
            attr_noise=config['attr_noise']  # fixme: at here, the noise is also add to test and valid set
        )
        val_datasets = ConcatDataset([val_dataset1, val_dataset2])
        val_datasets = ConcatDataset([val_datasets, val_dataset3])
        val_datasets = ConcatDataset([val_datasets, val_dataset4])

        return datasets, val_datasets, test_datasets

    # 下游任务数据集
    # 1. 华美医院乳腺钼靶数据集
    elif dataset == "ddsm":
        calc_train_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "calc_train_set.csv",
            stage="train",
            transform=train_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square'],
            partial_data=config['partial_data'],
        )
        mass_train_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "mass_train_set.csv",
            stage="train",
            transform=train_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square'],
            partial_data=config['partial_data'],
        )
        train_set = ConcatDataset([calc_train_set, mass_train_set])

        calc_valid_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "calc_valid_set.csv",
            stage="valid",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square']
        )

        mass_valid_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "mass_valid_set.csv",
            stage="valid",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square']
        )
        valid_set = ConcatDataset([calc_valid_set, mass_valid_set])

        calc_test_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "calc_test_set.csv",
            stage="test",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square']
        )

        mass_test_set = DDSMClsDataset(
            "/data2/zhai/DDSM/",
            "CBIS-DDSM/",
            "mass_test_set.csv",
            stage="test",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label'],
            padding_square=config['padding_square']
        )
        test_set = ConcatDataset([calc_test_set, mass_test_set])

        return train_set, valid_set, test_set

    # 3. InBreast数据集
    elif dataset == "inbreast":
        train_set = InBreastClsDataset(
            "/data2/zhai/InBreast/",
            "image",
            "inbreast_train.csv",
            stage="train",
            transform=train_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label']
        )
        valid_set = InBreastClsDataset(
            "/data2/zhai/InBreast/",
            "image",
            "inbreast_valid.csv",
            stage="valid",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label']
        )
        test_set = InBreastClsDataset(
            "/data2/zhai/InBreast/",
            "image",
            "inbreast_test.csv",
            stage="test",
            transform=test_transform,
            pre_downsample=config['pre_downsample'],
            binary_label=config['binary_label']
        )
    else:
        raise NotImplementedError(f"dataset {dataset} is not found.")

    return train_set, valid_set, test_set


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)

    return samplers


# data loader
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, drop_last, collate_fns, pin_memory=True):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, drop_last, collate_fn in zip(
            datasets, samplers, batch_size, num_workers, is_trains, drop_last, collate_fns
    ):
        if is_train:
            shuffle = (sampler is None)
            # drop_last = True
            persist = True
        else:
            shuffle = True
            # drop_last = False
            persist = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=persist,
        )
        loaders.append(loader)

    return loaders


def generate_attr_unique(attr):
    unique, inv = torch.unique((attr > 0.5).int(), dim=0, return_inverse=True)
    unique_mask = []
    for n in range(torch.max(inv).item() + 1):
        unique_mask.append(
            torch.where(inv == n)[0][0])  # todo: notice the second 0, it can be changed to random indice.
    return torch.tensor(unique_mask)
