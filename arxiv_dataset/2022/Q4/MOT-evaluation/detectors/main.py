

import os
import sys
import time
import torch
import pathlib
import argparse
import importlib

from torchvision import transforms

from DatasetLoader import DatasetLoader

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
    parser.add_argument("--size", help="Size of the image to enter in the model.", type=int, default=None)

    # Other optional arguments
    parser.add_argument("--clean_log", help="Remove logs and create new file.", type=str2bool, default=False)
    parser.add_argument("--trained", help="Use a trained model (only if coded).", type=str2bool, default=False)
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

    t = [transforms.ToTensor()]


    for setName in loader.listData():

        if verbose: print('Processing set', setName, '...')

        dataset = loader.loadData(setName, transform=t)

        start = time.time()
        detector.eval_set(dataset, loader, device)
        end = time.time()

        # print('>>time', end-start)
        duration = end - start
        total_images = loader.framesData(setName)
        nb_detection = loader.detectionsData()
        
        if duration == 0:  frames = total_images
        else:              frames = total_images / duration

        logger.writeLog([setName, '%.2f' % frames, '%d' % total_images, '%.2f' % duration, '%d' % nb_detection])

        # Save the results
        loader.save(setName)
        saveFPS(name, set_data, setName, frames)



def select_device(device):
    '''Select the device to use in next steps.
    '''

    if device == 'Auto':   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:                  device = args.device

    return device




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


    # Create detector object
    if not args.trained:
        detector = class_(args.batch)

    else:  detector = class_(args.batch, trained=True)

    # Write header in logger
    logger.writeHeader(detector.detector_name(args.name))


    # Set up the dataset.
    loader = DatasetLoader(path=args.path, set_data=args.set_data, savePath=args.detc_path, detectorName=detector.detector_name(args.name), size=args.size)

    
    # Run evaluation
    try:
        eval_sets(detector, loader, args.batch, device, logger, detector.detector_name(args.name), args.set_data, verbose=args.verbose)

    except KeyboardInterrupt:

        logger.writeEnd()
        sys.exit(0)

    except Exception as e:

        logger.writeException('EVATUATING DETECTORS', e)


    logger.writeEnd()



