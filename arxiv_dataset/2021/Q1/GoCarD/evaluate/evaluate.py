from eval.predict import yolo as y
from eval.convert import transform_prediction
from eval.map import main
from utils import create_data_name, anchor_file
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
parser = argparse.ArgumentParser(description='Evaluate models')
parser.add_argument('-d', '--data_path', type=str, dest='data_path', required=False, action='store',
                    default='./data/', help='specify which data path')
parser.add_argument('-t1', '--train1_source', type=str, dest='train1', required=False, action='store',
                    default='muse_part', help='specify train1 source, e.g. mix_car, muse_part')
parser.add_argument('-ds', '--test_source', type=str, dest='test', required=False, action='store',
                    default='muse_part', help='specify which test test source, e.g. mix_car, muse_part')
parser.add_argument('-t2', '--train2_source', type=str, dest='train2', required=False, action='store',
                    default=None, help='specify the train 2 source, e.g. mix_car, muse_part')  # 'mix_car'
parser.add_argument('-tf', '--train2_fraction', type=str, dest='train2_fraction', required=False, action='store',
                    default=None, help='specify the fraction used from source 1')
parser.add_argument('-mix', '--mix_devel', dest='mix_devel', required=False, action='store_true',
                    help='mix devel from both data sets (t1 and t2)')
parser.add_argument('-b', '--backbone', type=str, dest='backbone', required=False, action='store',
                    default='darknet53', help='specify the backbone e.g. darknet53, tiny_darknet, squeezenet')
parser.add_argument('-g', '--gpu', type=str, dest='gpu', required=False, action='store',
                    default=0, help='specify the gpu source')
parser.add_argument('-transfer', '--transfer_mode', type=int, dest='transfer_mode', required=False, action='store',
                    default=0, help='specify the gpu source')
parser.add_argument('-p', '--partition', type=str, dest='partition', required=False, action='store',
                    default='devel', help='specify which partition should be evaluated')
parser.add_argument('-iou', '--iou_threshold', nargs='+', dest='iou_threshold', required=False, action='store',
                    default=[0.2, 0.4, 0.5], help='specify the overlap threshold area of intersection / area of union')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
CONFIG_PATH = os.path.join(args.data_path, 'config')
data_name = create_data_name(args.train1, args.test, args.train2, args.train2_fraction, args.mix_devel)
PREPROCCESSED_PATH = os.path.join(args.data_path, 'preprocessed', data_name)

if args.transfer_mode > 0:
    print('--- TRANSFER LEARNING MODE ---')
    EXPERIMENT_NAME = args.backbone + '_' + data_name + '_' + str(args.transfer_mode)
    EXPERIMENTS_PATH = os.path.join('./experiments', 'transfer')
else:
    EXPERIMENT_NAME = args.backbone + '_' + data_name
    EXPERIMENTS_PATH = os.path.join('./experiments')

BEST_MODEL_PATH = os.path.join(EXPERIMENTS_PATH, 'best_model')
EXPERIMENT_PATH = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME)
OUTPUT_PREDICTION_PATH = os.path.join(EXPERIMENT_PATH, 'predictions')
OUTPUT_GROUND_TRUTH_PATH = os.path.join(EXPERIMENT_PATH, 'ground_truth')

def start_extract_labels_from_test(partition):
    if args.backbone == "darknet53":
        anchors_path = os.path.join(CONFIG_PATH, 'darknet53_anchors.txt')
        model_name = 'darknet'
    elif 'tinydarknet' in args.backbone:
        anchors_path = os.path.join(CONFIG_PATH, 'tiny_yolo_anchors.txt')
        model_name = 'darknet_tiny'
    elif 'squeeze' in args.backbone:
        anchors_path = os.path.join(CONFIG_PATH, 'darknet53_anchors.txt')
        model_name = 'squeeze'

    if not os.path.isfile(os.path.join(EXPERIMENT_PATH, partition, 'y_' + partition + '_pred.txt')):

        print("Start inference model: {}".format(
            os.path.join(BEST_MODEL_PATH, EXPERIMENT_NAME + '_checkpoint_weights.h5')))
        yolo = y.YOLO(model_name
                      , classes_path=os.path.join(CONFIG_PATH, 'full_classes.txt')
                      , anchors_path=anchors_path
                      , weights_path=os.path.join(BEST_MODEL_PATH, EXPERIMENT_NAME + '_checkpoint_weights.h5'))
        y.detect_img(yolo
                     , model_name=model_name
                     , test_path=os.path.join(PREPROCCESSED_PATH, 'X_' + partition + '.txt')
                     , output_path=os.path.join(EXPERIMENT_PATH, partition, 'y_' + partition + '_pred.txt'))
        print("Finished inference.")
    else:
        print("Predictions already exists in {}".format(
            os.path.join(EXPERIMENT_PATH, partition, 'y_' + partition + '_pred.txt')))
        print("[WARN] Progress with existing file.")


def start_transform_labels(partition):
    transform_prediction(annotation_file=os.path.join(EXPERIMENT_PATH, partition, 'y_' + partition + '_pred.txt')
                         , output_path=os.path.join(EXPERIMENT_PATH, partition, 'predictions')
                         , classes_path=os.path.join(CONFIG_PATH, 'full_classes.txt')
                         , is_ground_truth=False)

    transform_prediction(annotation_file=os.path.join(PREPROCCESSED_PATH, 'X_' + partition + '.txt')
                         , output_path=os.path.join(EXPERIMENT_PATH, partition, 'ground_truth')
                         , classes_path=os.path.join(CONFIG_PATH, 'full_classes.txt')
                         , is_ground_truth=True)

    print("Finished start_transform_labels.")


def start_main_calculation(partition):
    for iou in args.iou_threshold:
        main(GROUND_TRUTH_PATH=os.path.join(EXPERIMENT_PATH, partition, 'ground_truth')
             , PREDICTED_PATH=os.path.join(EXPERIMENT_PATH, partition, 'predictions')
             , results_files_path=os.path.join(EXPERIMENT_PATH, 'results_' + str(iou), partition)
             , MINOVERLAP=iou)


def calculate_partition(partition):
    print('Start for ', partition)

    for path in [os.path.join(EXPERIMENT_PATH, partition, 'predictions'),
                 os.path.join(EXPERIMENT_PATH, partition, 'ground_truth')]:
        print("Create {}".format(path))
        if not os.path.exists(path):
            os.makedirs(path)

    start_extract_labels_from_test(partition)
    start_transform_labels(partition)
    start_main_calculation(partition)


if __name__ == '__main__':
    calculate_partition(partition=args.partition)
    print("Finished start_transform_labels")
