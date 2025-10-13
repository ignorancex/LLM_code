from data_provider.data_loader import MultiDatasetsLoader
from data_provider.data_loader import SingleDatasetLoader

from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from utils.tools import CustomGroupSampler

# data type dict to loader mapping
data_type_dict = {
    # loading single dataset
    'SingleDataset': SingleDatasetLoader,

    # loading multiple datasets, concatenating them
    'MultiDatasets': MultiDatasetsLoader,  # datasets folder names presented in args.data_folder_list
}


def data_provider(args, flag):
    Data = data_type_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'supervised'\
                or args.task_name == 'pretrain_lead' \
                or args.task_name == 'pretrain_moco' \
                or args.task_name == 'pretrain_ts2vec' \
                or args.task_name == 'pretrain_biot' \
                or args.task_name == 'pretrain_eeg2rep' \
                or args.task_name == 'finetune':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if args.task_name == 'supervised' \
            or args.task_name == 'pretrain_moco' \
            or args.task_name == 'pretrain_ts2vec' \
            or args.task_name == 'pretrain_biot' \
            or args.task_name == 'pretrain_eeg2rep' \
            or args.task_name == 'finetune':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # only called when yeilding batches
        )
        return data_set, data_loader

    elif args.task_name == 'pretrain_lead':
        data_set = Data(
            root_path=args.root_path,
            args=args,
            flag=flag,
        )

        sampler = CustomGroupSampler(data_set, batch_size=batch_size,  group_size=2)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # only called when yielding batches
        )

        return data_set, data_loader
