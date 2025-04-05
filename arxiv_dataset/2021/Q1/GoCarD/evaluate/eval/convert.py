#python convert_keras-yolo3.py -o ex_in_compare_pred --dr /home/lstappen/CV/yolo/predict/predict_ex_in_compare_darknet_trained_weights_final.txt
'''
ABOUT THIS SCRIPT:
Converts ground-truth from the annotation files
according to the https://github.com/qqwweee/keras-yolo3
or https://github.com/gustavovaliati/keras-yolo3 format.

And converts the detection-results from the annotation files
according to the https://github.com/gustavovaliati/keras-yolo3 format.
'''

import argparse
import datetime
import os


def transform_prediction(annotation_file, output_path, classes_path, is_ground_truth):

    print('IS_ground_truth: ',is_ground_truth)
    print('annotation_file', annotation_file)

    with open(classes_path, 'r') as class_file:
        class_map = [f.strip('\n') for f in class_file.readlines()]

    with open(annotation_file, 'r') as annot_f:
        for annot in annot_f:
            annot = annot.strip(' \n').split(' ')

            img_path = annot[0].strip()
            # if ARGS.gen_recursive:
            #     annotation_dir_name = os.path.dirname(img_path)
            #     # remove the root path to enable to path.join.
            #     if annotation_dir_name.startswith('/'):
            #         annotation_dir_name = annotation_dir_name.replace('/', '', 1)
            #     destination_dir = os.path.join(ARGS.output_path, annotation_dir_name)
            #     os.makedirs(destination_dir, exist_ok=True)
            #     # replace .jpg with your image format.
            #     file_name = os.path.basename(img_path).replace('.jpg', '.txt')
            #     output_file_path = os.path.join(destination_dir, file_name)
            # else:
            img_path = img_path.split('/')[-1]
            file_name = img_path.replace('.jpg', '.txt').replace('/', '__')
            output_file_path = os.path.join(output_path, file_name)
            #if len(annot[1:]) > 0:
            with open(output_file_path, 'w') as out_f:
                for bbox in annot[1:]:
                    if is_ground_truth:
                        # Here we are dealing with ground-truth annotations
                        # <class_name> <left> <top> <right> <bottom> [<difficult>]
                        # todo: handle difficulty
                        #print(bbox.strip().split(','))
                        x_min, y_min, x_max, y_max, class_id = list(map(float, bbox.split(',')))
                        out_box = '{} {} {} {} {}'.format(
                            class_map[int(class_id)].strip(), x_min, y_min, x_max, y_max)
                    else:
                        # Here we are dealing with detection-results annotations
                        # <class_name> <confidence> <left> <top> <right> <bottom>
                        x_min, y_min, x_max, y_max, class_id, score = list(map(float, bbox.split(',')))
                        out_box = '{} {} {} {} {} {}'.format(
                            class_map[int(class_id)].strip(), score,  x_min, y_min, x_max, y_max)
                    out_f.write(out_box + "\n")
#            else:
#                print("No predictions found in {}".format(output_file_path))