
import os
import sys
import time
import pathlib
import argparse
import importlib
from Tracker import Tracker_abs

folders = sys.path[0].split('/')
folders = os.path.join('/', *folders[:-1], 'auxiliar')

sys.path.insert(0, folders)

from Logger import Logger, saveFPS, str2bool

def parseInput(list_detectors):
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Trackers demo')

    # Select an able detector from list.
    parser.add_argument("--tracker", help="Name of the tracker to use.", choices=list_detectors, required=True)

    # Important arguments, but optional.
    parser.add_argument("--name", help="Add a postfix to tracker name.")
    parser.add_argument("--detector", help="Detector used to generate detections.", default='public')
    parser.add_argument("--batch", help="Size of the batch.", default='5', type=int)

    # Other optional arguments
    parser.add_argument("--clean_log", help="Remove logs and create new file.", type=str2bool, default=False)
    parser.add_argument("--path", help="Path were to find the data.", default='dataset')
    parser.add_argument("--set_data", help="Name of set of data.", default='MOT20')
    parser.add_argument("--detc_path", help="Path were to find detections.", default='outputs/detections')
    parser.add_argument("--track_path", help="Name of folder used to store the tracks.", default='outputs/tracks')
    parser.add_argument("--aux_path", help="Path were to store auxiliar information of detectors.", default='trackers/auxiliar')
    parser.add_argument("-dvc", "--device", help="Device where is suposed to execute.", default='Auto')
    parser.add_argument("-v", "--verbose", help="Print debug information.", default=1, type=int)

    return parser.parse_args()



def readConfigFile():
    '''Read from the configuration file the available detectors.
    '''

    with open('trackers/trackers.conf', 'r') as fp:

        list_detectors = fp.read().split('\n')


    return list_detectors


def calculateMetrics(logger, img_path, sequences_time, track_path):


    for setName, duration in sequences_time.items():

        path = os.path.join(img_path, setName, 'img1')

        total_images = len(os.listdir(path))

        if duration == 0:  frames = total_images
        else:              frames = total_images / duration

        logger.writeLog([setName, '%.2f' % frames, '%d' % total_images, '%.2f' % duration])


        saveFPS(track_path, None, setName, frames, types='tracker')


if __name__ == '__main__':


    # Read configuration.
    list_trackers= readConfigFile()
    args = parseInput(list_trackers)


    # Import the selected detector.
    module = importlib.import_module(args.tracker)
    class_ = getattr(module, args.tracker)
    tracker = class_(args.batch)                              # Create tracker object


    # Initialice Logger
    logger = Logger('trackers/', 'trackers.log', ['NAME', 'FPS', 'IMAGES', 'TIME'], clean=args.clean_log)
    logger.writeHeader(tracker.tracker_name(args.name) + '  -  ' + args.detector)


    # Setup paths.
    img_path =        os.path.join(args.path, args.set_data)
    detections_path = os.path.join(args.detc_path, args.detector, args.set_data)
    track_path =      os.path.join(args.track_path, tracker.tracker_name(args.name), args.detector, args.set_data)

    Tracker_abs.create_path(track_path)
    Tracker_abs.create_path(args.aux_path)



    try:
        # Load data
        tracker.load_data(img_path, detections_path, args.aux_path)

    except KeyboardInterrupt:

        logger.writeEnd()
        sys.exit(0)

    except Exception as e:

        logger.writeException('PREPROCESSING DETECTIONS', e)
        logger.writeEnd()
        sys.exit(0)



    try:
        # Run Evaluation
        sequences_time = tracker.calculate_tracks(img_path, args.aux_path, track_path)

        frames = calculateMetrics(logger, img_path, sequences_time, track_path)


    except KeyboardInterrupt:

        logger.writeEnd()
        sys.exit(0)

    except Exception as e:

        logger.writeException('CALCULATING TRACES', e)


    logger.writeEnd()
