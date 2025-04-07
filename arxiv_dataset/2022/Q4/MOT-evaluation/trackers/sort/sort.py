
import os
import glob
import time

from Tracker import Tracker_abs

from sort.sort_def import *


class sort(Tracker_abs):

    def __init__(self, batch_size):

        super().__init__('sort', batch_size)

        self.max_age = 1
        self.min_hits = 3
        self.iou_threshold = 0.3



    def load_data(self, img_path, detections_path, aux_path):

        self.detections_path = detections_path


    def calculate_tracks(self, img_path, aux_path, track_path, verbose=0):

        # all train
        total_time = 0.0
        total_frames = 0

        path = track_path
        sequences_time = {}

        pattern = os.path.join(self.detections_path, '*', 'det', 'det.txt')


        for seq_dets_fn in glob.glob(pattern):

            mot_tracker = Sort(max_age=self.max_age, 
                               min_hits=self.min_hits,
                               iou_threshold=self.iou_threshold) #create instance of the SORT tracker

            seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
            seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

            start = time.time()
    

            with open(os.path.join(path, '%s.txt'%(seq)),'w') as out_file:

                if verbose == 2: print("Processing %s."%(seq))


                for frame in range(int(seq_dets[:,0].max())):

                    frame += 1 #detection and frame numbers begin at 1
                    dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                    dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                    total_frames += 1

                    start_time = time.time()
                    trackers = mot_tracker.update(dets)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    for d in trackers:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)


            end = time.time()

            sequences_time[seq] = end - start



        if verbose == 1:

            print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))


        return sequences_time

        