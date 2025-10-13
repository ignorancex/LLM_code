import os
import shutil

root = "/cluster/work/cvl/shbasu/phyfeaSegformer/data/cityscapes/leftImg8bit/test/"
dest_input = "/cluster/work/cvl/shbasu/phyfeaSegformer/results/"

for folders in os.listdir(root):
    for files in os.listdir(root+folders):
        shutil.copy(root+folders+'/'+files, dest_input+files)

