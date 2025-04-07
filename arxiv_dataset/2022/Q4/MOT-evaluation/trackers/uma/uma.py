
import os
import cv2
import glob
import time

import uma.run_public as run_public
import uma.config.config as CONFIG

from Tracker import Tracker_abs



class uma(Tracker_abs):

    def __init__(self, batch_size):

        super().__init__('uma', batch_size)

        self.max_age = 1
        self.min_hits = 3
        self.iou_threshold = 0.3



    def load_data(self, img_path, detections_path, aux_path):

        self.detections_path = detections_path

        pass


    def calculate_tracks(self, img_path, aux_path, track_path, verbose=0):

        # data_dir = '../../MOT-experimental-framework/dataset/MOT20/'
        # det_dir = '../../MOT-experimental-framework/outputs/detections/public/MOT20/'
        # output_dir = '../../MOT-experimental-framework/outputs/tracks/uma/MOT20/'

        sequences_time = {}
        self.max_age = 1


        data_dir = img_path
        det_dir  = self.detections_path
        output_dir = track_path


        os.makedirs(output_dir, exist_ok=True)

        trained_model = os.path.join(os.getcwd(), CONFIG.MODEL_DIR)

        display = CONFIG.DISPLAY
        context_amount = CONFIG.PRAM['context_amount']
        occlusion_thres = CONFIG.PRAM['occlusion_thres']
        association_thres = CONFIG.PRAM['association_thres']
        iou = CONFIG.PRAM['iou']


        sequences = os.listdir(data_dir)
        sequence_speed = []

        for sequence in sequences:

            if verbose: print("Running sequence %s" % sequence)

            sequence_dir = os.path.join(data_dir, sequence)
            output_file = os.path.join(output_dir, "%s.txt" % sequence)

            info_filename = os.path.join(sequence_dir, "seqinfo.ini")

            if os.path.exists(info_filename):

                with open(info_filename, "r") as f:

                    line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
                    info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)
                    frame_count = int(info_dict["seqLength"])
                    max_age = int(CONFIG.PRAM['life_span'] * int(info_dict["frameRate"]))   # max span-life or not

            else:

                img_list = os.listdir(os.path.join(sequence_dir, 'img1'))
                frame_count = int(max(img_list).split('.')[0])
                max_age = int(CONFIG.PRAM['life_span'] * 30)


                img_seq_list = os.listdir(os.path.join(sequence_dir, 'img1'))

                path_img_check = os.path.join(sequence_dir, 'img1', img_seq_list[0])
                img_check = cv2.imread(path_img_check)
                width = img_check.shape[1]
                height = img_check.shape[0]
                _, extension = os.path.splitext(img_seq_list[0])

                text = '[Sequence]\nname=%s\nimDir=img1\nframeRate=30\nseqLength=%d\nimWidth=%d\nimHeight=%d\nimExt=%s' % (sequence, len(img_seq_list), width, height, extension)


                with open(info_filename, "w") as f:

                    f.write(text)

            start = time.time()

            sequence_speed.append(run_public.run(
                sequence_dir, det_dir, trained_model, output_file,
                max_age,  context_amount, iou, occlusion_thres,
                association_thres, display))

            end = time.time()

            sequences_time[sequence] = end - start

        if verbose: print("Runtime: %g ms, %g fps.\n%s" % (sum(sequence_speed)/len(sequence_speed), 1000/(sum(sequence_speed)/len(sequence_speed)),output_dir))


        return sequences_time
