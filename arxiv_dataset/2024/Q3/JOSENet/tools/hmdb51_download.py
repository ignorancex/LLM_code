import os
os.system ("bash -c 'mkdir datasets/hmdb51'")
os.system ("bash -c 'wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'")
os.system ("bash -c 'mkdir -p datasets/hmdb51/video_data  datasets/hmdb51/test_train_splits'")
os.system ("bash -c 'unrar e test_train_splits.rar datasets/hmdb51/test_train_splits'")
os.system ("bash -c 'rm test_train_splits.rar'")

os.system ("bash -c 'wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'")
os.system ("bash -c 'unrar e hmdb51_org.rar'")
os.system ("bash -c 'mv *.rar datasets/hmdb51/video_data'")

import os
for files in os.listdir('datasets/hmdb51'):
    foldername = files.split('.')[0]
    os.system("mkdir -p datasets/hmdb51/video_data/" + foldername)
    os.system("unrar e datasets/hmdb51/video_data/"+ files + " datasets/hmdb51/video_data/"+foldername)

os.system ("bash -c 'rm datasets/hmdb51/video_data/*.rar'")
os.system ("bash -c 'rm -r datasets/hmdb51/hmdb51_org'")

