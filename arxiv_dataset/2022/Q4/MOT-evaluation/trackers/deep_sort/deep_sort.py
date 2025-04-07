
import os
import time

from Tracker import Tracker_abs

from deep_sort.tools.generate_detections import *
from deep_sort.deep_sort_app import *


class deep_sort(Tracker_abs):

    def __init__(self, batch_size):

        super().__init__('deep_sort', batch_size)

        # Model for load detections.
        self.model = 'trackers/deep_sort/models/mars-small128.pb'

        # Hyperparameters
        self.min_confidence = 0
        self.max_cosine_distance = 0.2
        self.nms_max_overlap = 1.0
        self.min_detection_height = 0
        self.nn_budget = 0


    def load_data(self, img_path, detections_path, aux_path):

        encoder = create_box_encoder(self.model, batch_size=self.batch_size)
        generate_detections(encoder, img_path, aux_path, detections_path)


    def calculate_tracks(self, img_path, aux_path, track_path, verbose=0):

        # Display
        if verbose:  display = True
        else:        display = False

        sequences_time = {}

        list_sequence = os.listdir(img_path)

        for sequece_name in list_sequence:

            sequence_dir = os.path.join(img_path, sequece_name)
            output_file =  os.path.join(track_path, sequece_name) + '.txt'
            detection_file = os.path.join(aux_path, sequece_name) + '.npy'

            start = time.time()

            run(sequence_dir, detection_file, output_file, self.min_confidence,
                self.nms_max_overlap, self.min_detection_height, self.max_cosine_distance,
                self.nn_budget, display)


            end = time.time()

            sequences_time[sequece_name] = end - start


        return sequences_time