from PIL import Image
import glob
import os

dir = '/home/shong/data_original_size/test/'
out_dir = '/home/shong/data/test/'
for file in glob.glob(dir+"*_mask.png"):
    print(file)
    only_filename = os.path.basename(file)
    image = Image.open(file)
    image = image.resize((256,256), Image.ANTIALIAS)
    image.save(fp=out_dir+only_filename)

    print(out_dir+only_filename)

