

import os
import sys
import time
import torch
import pathlib
import argparse
import importlib

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from DatasetLoaderTrain import Loader_train

newpath = os.path.join(pathlib.Path().absolute(), 'auxiliar')
sys.path.insert(0, newpath)

from Logger import Logger, saveFPS, str2bool


def parseInput(list_detectors):
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Detectors demo')

    # Select an able detector from list.
    parser.add_argument("--detector", help="Name of the detector to use.", choices=list_detectors, required=True)

    # Important arguments, but optional.
    parser.add_argument("--batch", help="Size of the batch.", default='5', type=int)
    parser.add_argument("--name", help="Add a postfix to the detector name.")
    parser.add_argument("--epochs", help="Epochs of training (fine tuning).", default='15', type=int)

    # Other optional arguments
    parser.add_argument("--clean_log", help="Remove logs and create new file.", type=str2bool, default=False)
    parser.add_argument("--path", help="Path were to find the data.", default='dataset')
    parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
    parser.add_argument("--detc_path", help="Name of folder used to store detections.", default='outputs/detections')
    parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='Auto')
    parser.add_argument("-v", "--verbose", help="Print debug information.", default=1, type=int)

    return parser.parse_args()



def readConfigFile():
    '''Read from the configuration file the available detectors.
    '''

    with open('detectors/detectors.conf', 'r') as fp:

        list_detectors = fp.read().split('\n')


    return list_detectors



def eval_sets(detector, loader, batch, device, logger, name, set_data, verbose=0):
    '''Evaluate each set of data.
    '''


    detector.train_model(loader, device, epochs)



def select_device(device):
    '''Select the device to use in next steps.
    '''

    if device == 'Auto':   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:                  device = args.device

    return device


def collate_fn(batch):
    return tuple(zip(*batch))



if __name__ == '__main__':


    # Read configuration.
    list_detectors = readConfigFile()
    args = parseInput(list_detectors)


    # Create logfile.
    logger = Logger('detectors/', 'detectors.log', ['NAME', 'FPS', 'IMAGES', 'TIME', 'DETECTIONS'], clean=args.clean_log)

    # Select the device where it supposed to run.
    device = select_device(args.device)


    # Import the selected detector.
    module = importlib.import_module(args.detector)
    class_ = getattr(module, args.detector)
    detector = class_(args.batch)                              # Create detector object


    # Write header in logger
    logger.writeHeader(detector.detector_name(args.name))


    # Set up the dataset.
    # loader = train_data = Loader_train('dataset/MOT20', set_list=['MOT20-02', 'MOT20-03'])
    loader = train_data = Loader_train('dataset/MOT17', set_list=['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'])
    loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)


    # Run evaluation
    try:
        detector.train_model(loader, args.epochs, device=device)

    except KeyboardInterrupt:

        logger.writeEnd()
        sys.exit(0)

    except Exception as e:

        logger.writeException('EVATUATING DETECTORS', e)


    logger.writeEnd()
