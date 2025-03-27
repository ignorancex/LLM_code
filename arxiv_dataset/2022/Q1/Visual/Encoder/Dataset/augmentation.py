import os
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from utils.json_utils import  fill_annotation, save_json
from utils.annotation_parser import JsonParser
from utils.utils import get_filenames
import random
import shutil
from PIL import Image

random.seed(10)

path_to_annotations = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/fold_2/fold_train_val/annotations_train.json"
path_to_images = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/fold_2/fold_train_val/train"
path_to_dest = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/fold_2/fold_train_val/train_aug"

transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=(0, 180), expand=True),
                T.ColorJitter(brightness=.2, hue=.2,saturation=.2,contrast=.2)
            ])
# transform_flip = T.Compose([
#                 T.RandomHorizontalFlip(),
#                 T.RandomVerticalFlip(),
#                 T.RandomRotation(degrees=(0, 270), expand=True, interpolation=InterpolationMode.NEAREST),
#             ])

# Load annotations
path_to_json = path_to_annotations
annotations = JsonParser()
annotations.load_json(path_to_json)
keys = {"annotations"}
gt_json = {key: [] for key in keys}
images_path = get_filenames(path_to_images, ".png")
images_path.sort()
for i in range(0, len(images_path)):
    # Load image
    image = Image.open(images_path[i])
    # Load annotation
    index = annotations.image_name.index(os.path.basename(images_path[i]))
    src = os.path.join(images_path[i])
    dst = os.path.join(path_to_dest, os.path.basename(images_path[i]))
    shutil.copy(src, dst)
    fill_annotation(gt_json,
                    image_name=annotations.image_name[index],
                    cont_id=annotations.container_id[index],
                    ar_w=annotations.ar_w[index],
                    ar_h=annotations.ar_h[index],
                    avg_d=annotations.avg_d[index],
                    wt=annotations.wt[index],
                    wb=annotations.wb[index],
                    h=annotations.height[index],
                    cap=annotations.capacity[index],
                    m=annotations.mass[index])
    for j in range(2):
        trasformed_image = transform(image)
        image_name = os.path.splitext(os.path.basename(images_path[i]))[0] + "_aug{}.png".format(j)
        trasformed_image.save(os.path.join(path_to_dest, image_name))
        fill_annotation(gt_json,
                        image_name=image_name,
                        cont_id=annotations.container_id[index],
                        ar_w=annotations.ar_w[index],
                        ar_h=annotations.ar_h[index],
                        avg_d=annotations.avg_d[index],
                        wt=annotations.wt[index],
                        wb=annotations.wb[index],
                        h=annotations.height[index],
                        cap=annotations.capacity[index],
                        m=annotations.mass[index])
    print("{}/{}".format(i+1, len(images_path)))

json_dest = os.path.splitext(path_to_annotations)[0] + "_aug_rgb.json"
save_json(json_dest, gt_json)
print("Finished!")
