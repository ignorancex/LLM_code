import os
os.system ("bash -c 'mkdir datasets/UCF-101/video_data'")
os.system ("bash -c 'wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate'")
os.system ("bash -c 'unrar x UCF101.rar UCF-101/video_data")
os.system ("bash -c 'rm UCF101.rar")

os.system ("bash -c 'wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate'")
os.system ("bash -c 'unzip UCF101TrainTestSplits-RecognitionTask.zip -d UCF-101'")
os.system ("bash -c 'rm UCF101TrainTestSplits-RecognitionTask.zip'")
