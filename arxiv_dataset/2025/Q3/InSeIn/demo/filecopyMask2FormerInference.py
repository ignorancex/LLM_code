import shutil
import os



video = '/cluster/work/cvl/shbasu/phyfeaSegformer/data/cityscapes/leftImg8bit/val/'
for filename in os.listdir(video):
    for files in os.listdir(video+filename):
        shutil.copy(video+filename+'/'+ files, '/cluster/work/cvl/shbasu/phyfeaSegformer/results/'+files)
