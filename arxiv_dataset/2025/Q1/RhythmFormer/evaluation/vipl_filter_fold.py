import cv2
import os

def read_frame_count(video_path):
    capture = cv2.VideoCapture(video_path)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    return frame_count

def read_gt_time(fps_file):
    if not os.path.exists(fps_file):  
        return 0
    value = 0
    with open(fps_file, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        last_line = last_line.strip()  
        value = float(last_line)  
    return value / 1000
  
def read_label_time(hr_file):
    with open(hr_file, 'r') as file:
        for i, line in enumerate(file, 1):
            pass
    return i-2

with open('vipl_filter_fold.txt', 'w') as file:
    for k in range(1, 108):
        for j in range(1, 13):
            if j == 10:
                j = "1-2"
            if j == 11:
                j = "3-2"
            if j == 12:
                j = "9-2"
            for i in range(1, 4):
                data_dirs = "../Dataset/VIPL_v1/VIPL_v1/p" + str(k) + "/v" + str(j) + "/source" + str(i)
                if not os.path.exists(os.path.join(data_dirs,"gt_HR.csv")):  
                    continue

                frame_count = read_frame_count(os.path.join(data_dirs,"video.avi"))
                label_time = read_label_time(os.path.join(data_dirs, "gt_HR.csv"))

                if not os.path.exists(os.path.join(data_dirs,"time.txt")):
                    # source 2
                    if abs(frame_count/label_time - 30) < 1 :
                        file.write('p'+str(k)+'_v'+str(j)+'_s'+str(i)+'\n')
                else:
                    gt_time = read_gt_time(os.path.join(data_dirs, "time.txt"))
                    if abs(gt_time - label_time) < 1:
                        if abs(frame_count/gt_time - 30) < 2:
                            file.write('p'+str(k)+'_v'+str(j)+'_s'+str(i)+'\n')
