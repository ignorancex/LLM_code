#
# Copyright (c) [2025] [personReID]
#
#
# Project title: Person detection and re-identification in open-world settings of retail stores and public spaces
#
# Authors: Branko Brkljač [Faculty of Technical Sciences, University of Novi Sad]
#   	   Milan Brkljač  [Faculty of Finance, Banking and Auditing, Alfa BK Univeristy]
#
## 
# Released under the MIT License
# You may obtain a copy of the License at: https://github.com/brkljac/personReID/blob/main/LICENSE
#
##
# personReID: near real-time demonstration of person re-identification (ReID) task  
#             using OAK-D embedded vision platform and OpenVINO™ framework
# 
# Achieves ~10fps on Intel® Movidius™ Myriad™ X vision processor and color camera from OAK-D lite device
#
#
##
# Based on the original project named: "Pedestrian reidentification" (MIT License Copyright (c) 2020 luxonis)
# Original source code: https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification
# You may obtain a copy of the original code License at: https://github.com/luxonis/depthai-experiments/blob/master/LICENSE
#
##
# Modifications: Added new functionalities for visualization and control, parallel encoding of input camera feed and
# output video, recoding of output video with processing results of person re-identification
#
# Requires the following pre-trained models (https://docs.openvino.ai/archives/index.html):
#   1) person-reidentification-retail-0288_openvino_2022.1_6shave
#   2) person-detection-retail-0013_openvino_2022.1_6shave
#
#
##
# Keyboard controls: 
# 'e': end execution (with additional operations after stopping threads execution); 
# 'p': pause execution; 'q': quit execution (without additional operations)
# 


import os
import subprocess
from datetime import datetime
import random

import cv2
import numpy as np
from depthai_sdk import OakCamera
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition


# console program control flags
write_output_video = True               # write processing results as output video file,
                                        # by default, visualization thread writes output video with 30fps
transform_output_video_fps = True       # recode fps rate of the output video to real fps of the system (to better resemble real operation speed)
record_original_camera_feed = True      # save original color camera feed

turn_off_keyboard_controls = False       # set to True for faster execution, 'q' will still work

root_folder_path = "./personReID_results/"      # location of the main output folder


str_message="""# ********************************************************
# personReID: real-time demonstration of person re-identification task
#
# Keyboard controls: 
# 'e': end execution; 'p': pause execution; 'q': quit execution
# 
# (check also program control flags inside the script)
# """

str_note="""# Note:
#
# - for simplicity, output video results are always recorded at 30 fps (regardless of the processing speed of the system)
# - however, in some cases output video fps should be close to the real speed of the system, e.g. 10 fps
# - in order to recode the output video results to the reproduction speed that is close to the real operating speed of the system (e.g. 10 fps)
#   please consider the following ffmpeg command that re-specifies the original presentation timestamps (PTS):
#
# 	ffmpeg -i .\output_30fps_video.mp4 -filter:v "setpts=PTS*30/10" -r 30 output_30fps_video__effective_10fps.mp4
#"""



# current date and time
now = datetime.now()

print("\n\n"+str_message+"\n\n"+now.strftime('%A, %B %d, %Y at %I:%M %p')+"\n\n")

# date and time for the folder name
folder_name = now.strftime("%Y%m%d_%H%M%S")

# output folder path
path = os.path.join(root_folder_path, folder_name)

# create the output folder
os.makedirs(path, exist_ok=True)

print(f"Folder '{folder_name}' created at: {path}")


class PedestrianReId:
    def __init__(self) -> None:
        self.results = []

    def _cosine_dist(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def new_result(self, vector_result) -> int:
        vector_result = np.array(vector_result)
        for i, vector in enumerate(self.results):
            dist = self._cosine_dist(vector, vector_result)
            if dist > 0.7:
                self.results[i] = vector_result
                return i
        else:
            self.results.append(vector_result)
            return len(self.results) - 1


with OakCamera() as oak:
    color = oak.create_camera('color', fps=10)

    person_det = oak.create_nn('person-detection-retail-0013', color)
    person_det.node.setNumInferenceThreads(2)
    person_det.config_nn(resize_mode='crop')

    nn_reid = oak.create_nn('person-reidentification-retail-0288', input=person_det)
    nn_reid.node.setNumInferenceThreads(2)

    reid = PedestrianReId()
    results = []

    id_color_map = {}  # Dictionary to store unique colors for each ID
    id_list = []

    def cb(packet: TwoStagePacket):
        visualizer = packet.visualizer
  
        for det, rec in zip(packet.detections, packet.nnData):
            reid_result = rec.getFirstLayerFp16()
            id = reid.new_result(reid_result)

            # check if the ID already has a color assigned
            if id not in id_list:
                # generate a random color for the new ID
                id_list.append(id)
                id_color_map[id] = [random.randint(0, 255) for _ in range(3)]

            # get the color for the current ID
            color = id_color_map[id]

            # draw the bounding box with the unique color
            visualizer.add_bbox(bbox=(*det.top_left, *det.bottom_right), thickness=4, color=color, label=f"ID: {id}")

            # add white text with the person ID
            visualizer.add_text(f"ID: {id}",
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.MID,
                            color=[255,255,255])

        if turn_off_keyboard_controls == False:
            #key = oak.poll()    # waits only 1ms for keyboard interrupt, check line 404 in oak_camera.py
            key = cv2.waitKey(60)
        else:
            key = 1

        if key == ord('p'):
            print("Pause execution ... press any key to continue")
            #cv2.waitKey(2000)
            # output videos' encodings will continue from the place in timeline where they were paused
            cv2.waitKey(-1)
            print(">>...")
        else:
            frame = visualizer.draw(packet.frame)
            cv2.imshow('Person reidentification', frame)

        if key == ord('e'):
            print("Ending execution ...")
            oak._stop = True
            
    
    if write_output_video == True:
        output_video_filenamePath = path+"/personReID_"+folder_name
        oak.visualize(nn_reid, record_path=output_video_filenamePath+".mp4", callback=cb, fps=True)
        print("\n\n"+str_note)
    else:
        oak.visualize(nn_reid, callback=cb, fps=True)
    
    # oak.show_graph()

    if record_original_camera_feed == True:
        oak.record(outputs=color, path=path, record_type=1)
                
    oak.start(blocking=True)    # stop execution by pressing 'e' on the keyboard

    if(oak._stop==True and transform_output_video_fps==True):
        # create additional output video with more realistic fps, e.g. ~10 fps (should be changed in case that real fps is higher)
        ffmpeg_command_list = ["ffmpeg", "-i", "./"+output_video_filenamePath+".mp4", "-filter:v", "setpts=PTS*30/10", "-r", "30", "./"+output_video_filenamePath+"_real_fps.mp4"]
        print("\n\nffmpeg command for output video recoding: \n\n>> " + ' '.join(ffmpeg_command_list) + "\n\n...\n\n")
        if subprocess.run(ffmpeg_command_list).returncode == 0:
            print ("ffmpeg ran successfully\n")
        else:
            print ("ffmpeg error ...")
    
    # in necessary comment: oak_camera.py, line 96 >> report_crash_dump(self.device)
