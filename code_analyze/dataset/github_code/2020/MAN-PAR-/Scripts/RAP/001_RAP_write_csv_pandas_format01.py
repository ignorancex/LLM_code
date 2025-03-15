import csv
import os
import mat4py
import cv2

def show_img_(img_dir, img_labels, RAP_labels, show=True):

    if show:
        indices = [i for i, x in enumerate(img_labels) if x ==1]
        labels = [RAP_labels[i+1] for i in indices]
        print(labels)
        img = cv2.imread(img_dir)
        cv2.imshow(img_dir, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


RAP_dir = "../RAP_Data/IMG/"
List_of_images = os.listdir(RAP_dir)
Annot_mat_dir = "../RAP_Data/RAP_annotation.mat"
RAP_dictionary = mat4py.loadmat(Annot_mat_dir)

RAP_annotation =  RAP_dictionary["RAP_annotation"]
RAP_Attributes = RAP_annotation["attribute"]
RAP_Selected_Attributes = RAP_annotation["selected_attribute"]
RAP_name_of_imgs = RAP_annotation["name"]
RAP_Annotations_list = RAP_annotation["data"]
partition_attribute = RAP_annotation["partition_attribute"][0]
train_index = partition_attribute["train_index"] + partition_attribute["val_index"]
test_index = partition_attribute["test_index"]

# Train set
indx = 0
print("Indices of Labels: ", RAP_Selected_Attributes)
RAP_labels = [RAP_Attributes[i-1][0] for i in RAP_Selected_Attributes]
RAP_labels_train = ["Train_Filenames"] + RAP_labels + ["Male"]
print("Labels: ", RAP_labels_train)
print("Number of labels: ",len(RAP_labels_train))
all_lables = []
for i, IMG_index in enumerate(train_index):
    img_name = RAP_name_of_imgs[IMG_index-1]
    img_labels = RAP_Annotations_list[IMG_index-1]
    selected_img_annot = [img_labels[i-1] for i in RAP_Selected_Attributes]
    selected_img_annot.append(1-selected_img_annot[0])
    img_labels_joined = [RAP_dir + img_name[0]]
    show_img_(img_labels_joined[0], selected_img_annot, RAP_labels_train, show=False)

    for index, element in enumerate(selected_img_annot):
        img_labels_joined.append(str(element))
    all_lables.append(img_labels_joined)

    if i % 10000 == 0:
        print("Creating TRAIN dataframe \t {}/{}".format(i, len(List_of_images)))
    # if indx==100:
    #    break
with open("TRAIN_RAP_pandas_frame_data_format_1.csv", "w", newline='') as CSV:
    text = csv.writer(CSV, delimiter=',')
    text.writerow(RAP_labels_train)
    for line in all_lables:
        text.writerow(line)





# Test set
indx = 0
RAP_labels_test = ["Test_Filenames"] + RAP_labels + ["Male"]
all_lables = []
for i, IMG_index in enumerate(test_index):
    img_name = RAP_name_of_imgs[IMG_index-1]
    img_labels = RAP_Annotations_list[IMG_index-1]
    selected_img_annot = [img_labels[i-1] for i in RAP_Selected_Attributes]
    selected_img_annot.append(1-selected_img_annot[0])
    img_labels_joined = [RAP_dir + img_name[0]]

    show_img_(img_labels_joined[0], selected_img_annot, RAP_labels_test, show=False)

    for index, element in enumerate(selected_img_annot):
        img_labels_joined.append(str(element))

    all_lables.append(img_labels_joined)

    if i % 10000 == 0:
        print("Creating TEST dataframe \t {}/{}".format(i, len(List_of_images)))
    # if indx==100:
    #    break
with open("TEST_RAP_pandas_frame_data_format_1.csv", "w", newline='') as CSV:
    text = csv.writer(CSV, delimiter=',')
    text.writerow(RAP_labels_test)
    for line in all_lables:
        text.writerow(line)
