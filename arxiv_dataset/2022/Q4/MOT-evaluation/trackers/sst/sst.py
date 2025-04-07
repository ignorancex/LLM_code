
import os
import cv2
import sys
import time
import numpy as np

sys.path.append('trackers/sst')

from tracker import SSTTracker, TrackerConfig, Track
from data.mot_data_reader import MOTDataReader
from config.config import config
from utils.timer import Timer
# import argparse

from Tracker import Tracker_abs

# parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
# parser.add_argument('--version', default='v1', help='current version')
# parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
# parser.add_argument('--type', default=config['type'], help='train/test')
# parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
# parser.add_argument('--save_video', default=True, help='save video if true')
# parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
# parser.add_argument('--mot_version', default=17, help='mot version')

# args = parser.parse_args()



class sst(Tracker_abs):

    def __init__(self, batch_size):

        super().__init__('sst', batch_size)

        self.save_video = False
        self.show_image = False
        self.log_folder = 'trackers/sst/aux/logs'
        config['cuda'] = True

        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)



    def load_data(self, img_path, detections_path, aux_path):

        self.detections_path = detections_path


    def calculate_tracks(self, img_path, aux_path, track_path, verbose=0):

        self.img_path = img_path
        self.track_path = track_path


        all_choices = TrackerConfig.get_choices_age_node()
        iteration = 3
        # test()

        i = 0
        for age in range(1):
            for node in range(1):
                c = (0, 0, 4, 0, 3, 3)
                choice_str = TrackerConfig.get_configure_str(c)
                TrackerConfig.set_configure(c)
                print('=============================={}.{}=============================='.format(i, choice_str))
                sequences_time = self.test(c)
                i += 1

        return sequences_time



    def test(self, choice=None):

        timer = Timer()
        sequences_time = {}

        for subset in os.listdir(self.img_path):


            saved_file_name     = os.path.join(self.track_path, subset + '.txt')
            image_folder        = os.path.join(self.img_path, subset, 'img1')
            detection_file_name = os.path.join(self.detections_path, subset, 'det/det.txt')

            # print('start processing '+saved_file_name)

            # print('*******************************************************')
            # print('*******************************************************')
            # print('*******************************************************')
            # print(image_folder)
            # print(detection_file_name)
            # print(saved_file_name)
            # print('*******************************************************')

            os.chdir("trackers/sst/")
            tracker = SSTTracker()
            os.chdir("../../")
            reader = MOTDataReader(image_folder = image_folder,
                          detection_file_name =detection_file_name,
                                   min_confidence=0.0)

            result = list()
            result_str = saved_file_name
            first_run = True

            start = time.time()

            for i, item in enumerate(reader, start=1):
                if i > len(reader):
                    break

                if item is None:
                    continue

                img = item[0]
                det = item[1]


                if img is None or det is None or len(det)==0:
                    continue

                if len(det) > config['max_object']:
                    det = det[:config['max_object'], :]

                h, w, _ = img.shape


                if first_run and self.save_video:
                    vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
                    first_run = False


                det = det.astype(float)


                # det[:, [2,4]] /= float(w)
                det[:, [2,4]] = det[:, [2,4]] / float(w)
                # det[:, [3,5]] /= float(h)
                det[:, [3,5]] = det[:, [3,5]] / float(h)
                timer.tic()


                image_org = tracker.update(img, det[:, 2:6], self.show_image, i)
                timer.toc()
                # print('{}:{}, {}, {}\r'.format(os.path.basename(saved_file_name), i, int(i*100/len(reader)), choice_str))
                if self.show_image and not image_org is None:
                    cv2.imshow('res', image_org)
                    cv2.waitKey(1)

                if self.save_video and not image_org is None:
                    vw.write(image_org)

                # save result
                for t in tracker.tracks:

                    n = t.nodes[-1]
                    if t.age == 1:

                        b = n.get_box(tracker.frame_index-1, tracker.recorder)

                        result.append(
                            [i] + [t.id] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h] + [-1, -1, -1, -1]
                        )

            end = time.time()
            # save data
            np.savetxt(saved_file_name, np.array(result).astype(int), fmt='%i', delimiter=',')

            sequences_time[subset] = end - start

        #print(timer.total_time)
        #print(timer.average_time)
        return sequences_time




# if __name__ == '__main__':
#     all_choices = TrackerConfig.get_choices_age_node()
#     iteration = 3
#     # test()

#     i = 0
#     for age in range(1):
#         for node in range(1):
#             c = (0, 0, 4, 0, 3, 3)
#             choice_str = TrackerConfig.get_configure_str(c)
#             TrackerConfig.set_configure(c)
#             print('=============================={}.{}=============================='.format(i, choice_str))
#             test(c)
#             i += 1

    # for i in range(10):
    #     c = all_choices[-i]
    #
    #     choice_str = TrackerConfig.get_configure_str(c)
    #     TrackerConfig.set_configure(c)
    #     print('=============================={}.{}=============================='.format(i, choice_str))
    #     test(c)
