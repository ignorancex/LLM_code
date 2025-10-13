from data.cifar10 import *
from data.cifar100 import *
from data.tinyimagenet import *
from data.imagenet import *
from data.office31 import *
from data.offce_home import *
from data.wilds_camelyon17 import *
import torch
from data.breeds import *

def build_dataloader(dataname, args):
    random_seeds = torch.randint(0, 10000, (2,))
    if args['severity'] == 0:
        seed = 1
        datatype = 'train'
        corruption_type = 'clean'
    else:
        corruption_type = args['corruption']
        seed = random_seeds[1]
        datatype = 'test'
    if dataname == 'cifar10':
        valset = load_cifar10_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'cifar100':
        valset = load_cifar100_image(corruption_type,
                                    clean_cifar_path=args['cifar_data_path'],
                                    corruption_cifar_path=args['cifar_corruption_path'],
                                    corruption_severity=args['severity'],
                                    datatype=datatype,
                                    seed=seed,
                                    num_samples=args['num_samples'])
    elif dataname == 'tinyimagenet':
        valset = load_tinyimagenet(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'imagenet':
        valset = load_Imagenet(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               num_classes= args['num_classes'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'office31':
        valset = get_offce31_loader(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'office_home':
        valset = get_offce_home_loader(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif dataname == 'wilds_camelyon17':
        valset = load_wilds_camelyon17(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               corruption_severity=args['severity'],
                               datatype=datatype)
    elif (dataname == 'entity13') or (dataname == 'entity30') or (dataname == 'living17') or (dataname == 'nonliving26'):
        if datatype == 'train':
            name = args['train_data_name']
        else:
            name = args['dataname']

        valset = get_imagenet_breeds(corruption_type,
                               clean_cifar_path=args['cifar_data_path'],
                               corruption_cifar_path=args['cifar_corruption_path'],
                               name=name,
                               corruption_severity=args['severity'],
                               datatype=datatype)
    else:
        raise Exception("Not Implemented Error")

    valset_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=args['batch_size'],
                                                num_workers = 4,
                                                shuffle=True)
    return valset_loader
