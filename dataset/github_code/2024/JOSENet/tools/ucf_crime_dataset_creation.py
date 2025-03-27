
from tqdm import tqdm
import os
import cv2 



def load_video(path):
    vidcap = cv2.VideoCapture(path)
    assert vidcap.isOpened()

    fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    fps_out = 7.5

    index_in = -1
    index_out = -1

    segments_count = 0
    segment  = []
    video_tot = []


    while True:
        success = vidcap.grab()
        if not success: break
        index_in += 1

        if (index_in >= 300):
            out_due = int(index_in / fps_in * fps_out)
            if out_due > index_out:
                success, frame = vidcap.retrieve()
                if not success: break
                index_out += 1
                # do something with `frame`
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dim = (224, 224)
                frame = cv2.resize(frame, dim)
                segment.append(frame)

                if (len(segment) == 16):
                    video_tot.append(segment)
                    segments_count+=1
                    segment = []

                if (segments_count == 62):
                    return video_tot
        
    return video_tot


with open("Anomaly_Train.txt") as f:
    train_list = f.readlines()  

for i,l in enumerate(train_list):
    train_list[i] = train_list[i].split("/")[1]
    train_list[i] = train_list[i].strip("\n")

for root, _, files in os.walk("anomaly_videos", topdown=False):
  for name in tqdm(sorted(files)):
    path_to_file = root+"/"+name
    file_name = name.split(".")[0]

    if (name in train_list):
      video = load_video(path_to_file)
      save_dir = "datasets/ucf_crime_jpg_frames/train/"+ file_name
    else:
      video = load_video(path_to_file)
      save_dir = "datasets/ucf_crime_jpg_frames/val/"+ file_name

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    idx = 0
    for segment in video:
      for frame in segment:
        cv2.imwrite(save_dir+"/"+file_name+"_"+str(idx)+".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        idx+=1
