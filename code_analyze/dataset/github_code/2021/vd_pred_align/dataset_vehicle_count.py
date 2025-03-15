

dataset_dir = "E:/work/dataset/raw/DA_images/NTT_scale0.3/7_19"

import os

anno_files = []

for x in os.listdir(dataset_dir):
    file = os.path.join(dataset_dir,  x)
    if os.path.isfile(file) and os.path.splitext(file)[-1] == ".txt":
        anno_files.append(file)

vehicle_num = 0

for annotation_file in anno_files:
    with open(annotation_file, "r") as annotations:
        line = annotations.readline()
        while (line):
            # xmin, ymin, xmax, ymax = line.split(",")
            # xmin = int(xmin)
            # ymin = int(ymin)
            # xmax = int(xmax)
            # ymax = int(ymax)
            # bbox.append([ymin - 1, xmin - 1, ymax - 1, xmax - 1])  # obey the rule of chainercv
            # label.append(vehicle_classes.index("car"))
            vehicle_num += 1
            line = annotations.readline()

print("vehicle num:{}".format(vehicle_num))