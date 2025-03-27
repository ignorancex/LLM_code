import csv
import os
import mat4py
import cv2

def show_img_(The_img_dir, The_image_labels, The_PETA_labels, show=True):

    if show:
        indices = [i for i, x in enumerate(The_image_labels) if x ==1]
        labels = [The_PETA_labels[i+1] for i in indices]
        print(indices)
        print(labels)
        img = cv2.imread(The_img_dir)
        the_img_name = The_img_dir.split("/")[-1]
        cv2.imshow(the_img_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


PETA_dir = "../DATA/resized_imgs/"
List_of_images = os.listdir(PETA_dir)
Annot_mat_dir = "../DATA/PETA.mat"
PETA_dictionary = mat4py.loadmat(Annot_mat_dir)

PETA_annotation =  PETA_dictionary["peta"]
PETA_Attributes = PETA_annotation["attribute"]
PETA_Selected_Attributes = PETA_annotation["selected_attribute"]
PETA_Annotations_list = PETA_annotation["data"]
partition_attribute = PETA_annotation["partion_attribute"][0]
train_index = partition_attribute[0]["train"] + partition_attribute[0]["val"]
test_index = partition_attribute[0]["test"]
## Get the name of the images
PETA_name_of_imgs = []

for i, img_name in enumerate(List_of_images):
    nam = img_name.split(".")[0]
    PETA_name_of_imgs.append(nam)
PETA_name_of_imgs.sort()

# Add female label (88) to selected attributes »»» 88 + 4 = 92
PETA_Selected_Attributes.append(92)


# Train set
indx = 0
print("Indices of Labels: ", PETA_Selected_Attributes)
# To catch the right attribute, we need to shift the selected attribute 5 number.
# PETA_Attributes contains the attributes name only (starts from 1 to 105) but annotation data is from 5 to 109 (in MATLAB).
PETA_labels = [PETA_Attributes[eachNumber-5][0] for j,eachNumber in enumerate(PETA_Selected_Attributes)]
PETA_labels_train = ["Train_Filenames"] + PETA_labels
print("Labels: ", PETA_labels_train)
print("Number of labels: ",len(PETA_labels_train))
all_lables = []
for myindex, IMG_index in enumerate(train_index):
    img_name = PETA_name_of_imgs[IMG_index[0]-1]
    img_labels = PETA_Annotations_list[IMG_index[0]-1]
    selected_img_annot = [img_labels[i-1] for i in PETA_Selected_Attributes]
    img_labels_joined = [PETA_dir + str(img_name) + ".png"]
    show_img_(The_img_dir = img_labels_joined[0], The_image_labels = selected_img_annot, The_PETA_labels = PETA_labels_train, show=False)

    ## this if statement is for creating a light example.
    if os.path.isfile(img_labels_joined[0]):
        for index, element in enumerate(selected_img_annot):
            img_labels_joined.append(str(element))
        all_lables.append(img_labels_joined)

    if i % 10000 == 0:
        print("Creating TRAIN dataframe \t {}/{}".format(i, len(List_of_images)))
    # if indx==100:
    #    break
with open("TRAIN_PETA_pandas_frame_data_format_1.csv", "w", newline='') as CSV:
    text = csv.writer(CSV, delimiter=',')
    text.writerow(PETA_labels_train)
    for line in all_lables:
        text.writerow(line)





# Test set
indx = 0
print("Indices of Labels: ", PETA_Selected_Attributes)
# To catch the right attribute, we need to shift the selected attribute 5 number.
# PETA_Attributes contains the attributes name only (starts from 1 to 105) but annotation data is from 5 to 109 (in MATLAB).
PETA_labels = [PETA_Attributes[eachNumber-5][0] for j,eachNumber in enumerate(PETA_Selected_Attributes)]
PETA_labels_test = ["Test_Filenames"] + PETA_labels
print("Labels: ", PETA_labels_test)
print("Number of labels: ",len(PETA_labels_test))
all_lables = []
for myindex, IMG_index in enumerate(test_index):
    img_name = PETA_name_of_imgs[IMG_index[0]-1]
    img_labels = PETA_Annotations_list[IMG_index[0]-1]
    selected_img_annot = [img_labels[i-1] for i in PETA_Selected_Attributes]
    img_labels_joined = [PETA_dir + str(img_name) + ".png"]
    show_img_(The_img_dir = img_labels_joined[0], The_image_labels = selected_img_annot, The_PETA_labels = PETA_labels_test, show=False)

    ## this if statement is for creating a light example.
    if os.path.isfile(img_labels_joined[0]):
        for index, element in enumerate(selected_img_annot):
            img_labels_joined.append(str(element))
        all_lables.append(img_labels_joined)

    if i % 10000 == 0:
        print("Creating TEST dataframe \t {}/{}".format(i, len(List_of_images)))
    # if indx==100:
    #    break
with open("TEST_PETA_pandas_frame_data_format_1.csv", "w", newline='') as CSV:
    text = csv.writer(CSV, delimiter=',')
    text.writerow(PETA_labels_test)
    for line in all_lables:
        text.writerow(line)
