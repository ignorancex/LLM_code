import cv2
import os

boxes = open("./CUB_200_2011/CUB_200_2011/bounding_boxes.txt")
images_id = open("./CUB_200_2011/CUB_200_2011/images.txt")
train_test = open("./CUB_200_2011/CUB_200_2011/train_test_split.txt")
folder = "./CUB_200_2011/CUB_200_2011/images/"
path_test = "./CUB_200_2011/CUB_200_2011/datasets/cub200_cropped/test_cropped/"
path_training = "./CUB_200_2011/CUB_200_2011/datasets/cub200_cropped/train_cropped/"


# Make directory if it does not exist
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


for (line, box, train) in zip(images_id, boxes, train_test):
    # Parse lines from given files e.g. get coordinates box, image path and name and train or test
    coords = list(map(int, map(float, box.split()[1:])))
    image_path = line.split()[1]
    image_folder = image_path.split("/")[0]
    image_name = image_path.split("/")[1]
    is_train = int(train.split()[1])

    # Load and crop image
    image = cv2.imread(os.path.join(folder, image_path))[
            coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2], :]

    # Store image either as train or test image
    if is_train:
        path_folder = os.path.join(path_training, image_folder)
        makedir(os.path.join(path_training, image_folder))
    else:
        path_folder = os.path.join(path_test, image_folder)
        makedir(os.path.join(path_test, image_folder))
    cv2.imwrite(os.path.join(path_folder, image_name).replace('jpg', 'JPEG'), image)

