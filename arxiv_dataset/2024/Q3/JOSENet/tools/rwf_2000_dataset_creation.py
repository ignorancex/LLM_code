import os
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm



def load_video(in_dir, in_file, max_frames=0):

    # this is the complete path of the video, including name
    in_file = in_file
    #print("received input file: ", in_file)

    # file name is just the suffix of path, indicating the video name
    file_name = in_file.split("/")[-1].replace(".avi", "")

    # create save dir as received input dir + file name
    save_dir = os.path.join(in_dir, file_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print("input file: ", in_file)

    vidcap = cv2.VideoCapture(in_file)
    
    success,image = vidcap.read()
    idx = 0

    # # Change the current directory 
    # # to specified directory 
    # os.chdir(save_dir)

    try:
        while success:
            
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dim = (224, 224)
            frame = cv2.resize(img, dim)
            # print("file name: ", os.path.join(save_dir, file_name+"_"+str(idx)+".jpg"))
            cv2.imwrite( os.path.join(save_dir, "frame"+"_"+str(idx)+".jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


            success,image = vidcap.read()
            idx += 1

    finally:
        vidcap.release()

    return


dataset_path = "2 - хИТхИЖш┐ЗчЪДцХ░цНощЫЖ"

# Save train frames

dict_names = {}

for root, dir, files in os.walk(dataset_path+"/train", topdown=False):
  # print(root)
  # for subroot in root:
  length = len(root.split("/"))
  if(length>3):
    key = ((root.split("/")[3]))
    dict_names[key] = []
    for f in listdir(root):
      if isfile(join(root, f)):
        dict_names[key].append(join(root, f))


base_dir = "datasets/RWF-2000_frames"
save_dir_train = os.path.join(base_dir, "train")
if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)


for key in dict_names:
  savedir = os.path.join(save_dir_train, key)
  # create directory if doesn't exist
  if not os.path.exists(savedir):
        os.makedirs(savedir)
  for elem in tqdm(dict_names[key]):
    load_video(savedir , elem )


# Save validation frames

dict_names_validation = {}

for root, dir, files in os.walk(dataset_path+"/val", topdown=False):
  # print(root)
  # for subroot in root:
  length = len(root.split("/"))
  if(length>3):
    key = ((root.split("/")[3]))
    dict_names_validation[key] = []
    for f in listdir(root):
      if isfile(join(root, f)):
        dict_names_validation[key].append(join(root, f))

base_dir = "datasets/RWF-2000_frames"
save_dir_val = os.path.join(base_dir, "val")
if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)


for key in dict_names_validation:
  savedir = os.path.join(save_dir_val, key)
  # create directory if doesn't exist
  if not os.path.exists(savedir):
        os.makedirs(savedir)
  for elem in tqdm(dict_names_validation[key]):
    load_video(savedir , elem )
