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

def write_csv_RAP(split, index_of_the_selected_attrs, list_of_image_names, dictionary_containing_images, show_image):
    all_images_with_selected_annotations = []
    name_of_the_attrs = []
    image_without_kyepoints = 0
    for index, img_name in enumerate(list_of_image_names):
        if img_name in dictionary_containing_images:
            if len(name_of_the_attrs) == 0:
                name_of_the_attrs = dictionary_containing_images[img_name]["attibute_names"]
                name_of_the_attrs = sorted(name_of_the_attrs, key=lambda k: name_of_the_attrs[k])
            all_annotations_for_this_img = dictionary_containing_images[img_name]["attrs"] # annotations: e.g. [1,1,0,0,0,0,1,0,1,0,...]
            selected_annotations_for_this_img = [all_annotations_for_this_img[i] for i in index_of_the_selected_attrs]
            selected_annotations_for_this_img.insert(0,img_name)
            all_images_with_selected_annotations.append(selected_annotations_for_this_img)
            if show_image:
                # prepare list of existed labels in this image e.g. ["male", "tall"]
                all_appeared_labels_for_this_img = [] # labels: e.g. ["male", "tall", "sport_shoes", ...]
                selected_and_appeared_labels_for_this_img = []
                for index, annotation in enumerate(all_annotations_for_this_img):
                    if annotation==1:
                        all_appeared_labels_for_this_img.append(name_of_the_attrs[index])
                        if index in index_of_the_selected_attrs:
                            selected_and_appeared_labels_for_this_img.append(name_of_the_attrs[index])
                 # read image to show
                img_path = os.path.join(rap_images_dir, img_name)
                img = cv2.imread(img_path)
                cv2.imshow("{}".format(img_name),img)
                print(">>>>\n\tImage name and annotations:", selected_annotations_for_this_img)
                print("\tALL_appeared_labels_for_this_img:", all_appeared_labels_for_this_img)
                print("\tSELECTED_and_appeared_labels_for_this_img:", selected_and_appeared_labels_for_this_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if index % 10000 == 0:
                print("Creating dataframe \t {}/{}".format(index, len(dictionary_containing_images)))
        else:
            image_without_kyepoints += 1
            print("Not have Keypoint: {},\t{} doesn't have keypoint and will not be considered in our experiment".format(image_without_kyepoints,img_name))

    first_row_of_the_pandas_dataframe = [name_of_the_attrs[i] for i in index_of_the_selected_attrs]
    first_row_of_the_pandas_dataframe.insert(0,"{}_filenames".format(split))
    with open("{}_RAP_pandas_frame_data_format_1.csv".format(split), "w", newline='') as CSV:
        text = csv.writer(CSV, delimiter=',')
        text.writerow(first_row_of_the_pandas_dataframe)
        for line in all_images_with_selected_annotations:
            text.writerow(line)


if __name__ == '__main__':
    #TODO: creating two csv files (for train and test data) in the format of panda dataframe as below:
    # Train_file_names,        label1,          label2,          label3,        ...,      labelN
    # path/to/img1,         zero_or_one,     zero_or_one,      zero_or_one,     ...,    zero_or_one
    # path/to/img2,         zero_or_one,     zero_or_one,      zero_or_one,     ...,    zero_or_one
    # path/to/img3,         zero_or_one,     zero_or_one,      zero_or_one,     ...,    zero_or_one
    # path/to/img4,         zero_or_one,     zero_or_one,      zero_or_one,     ...,    zero_or_one
    # path/to/img5,         zero_or_one,     zero_or_one,      zero_or_one,     ...,    zero_or_one

    import rap_data_loading as rap
    # paths
    rap_images_dir = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_images'
    rap_masks_dir = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_masks'
    rap_keypoints_json = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Anchor_level_rap/rap_annotations/RAP_keypoints.json'
    rap_attribute_annotations = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Anchor_level_rap/rap_annotations/RAP_annotation.mat'

    train_test_split_partition = 0 # you can select partition from 0 to 5 for different splits

    train_imgs, test_imgs = rap.train_test_imgs(rap_mat_file=rap_attribute_annotations, partition=train_test_split_partition)

    # select the index of the desired attributes that you like to show up in csv file.
    index_of_the_selected_attrs = [21,22,23,24,25,26,27,28,29,45,46,47,48,49,50,51,52] # upperbody and lowerbody clothes
    rap_data_with_keypoint = rap.load_rap_dataset(rap_attributes_filepath=rap_attribute_annotations, rap_keypoints_json=rap_keypoints_json)

    write_csv_RAP(split="TRAIN",
                  index_of_the_selected_attrs = index_of_the_selected_attrs,
                  list_of_image_names=train_imgs,
                  dictionary_containing_images=rap_data_with_keypoint,
                  show_image=False)
    write_csv_RAP(split="TEST",
                  index_of_the_selected_attrs = index_of_the_selected_attrs,
                  list_of_image_names=train_imgs,
                  dictionary_containing_images=rap_data_with_keypoint,
                  show_image=False)

    # # TODO: write train csv
    # all_imgs_with_selected_labels = []
    # name_of_the_attrs = []
    # image_without_kyepoints = 0
    # for index, img_name in enumerate(train_imgs):
    #     if img_name in rap_data_with_keypoint:
    #         if len(name_of_the_attrs) == 0:
    #             name_of_the_attrs = rap_data_with_keypoint[img_name]["attibute_names"]
    #             name_of_the_attrs = sorted(name_of_the_attrs, key=lambda k: name_of_the_attrs[k])
    #         img_path = os.path.join(rap_images_dir, img_name)
    #         img_labels_all = rap_data_with_keypoint[img_name]["attrs"]
    #         img_labels_selected = [img_labels_all[i] for i in index_of_the_selected_attrs]
    #         img_labels_selected.insert(0,img_name)
    #         all_imgs_with_selected_labels.append(img_labels_selected)
    #         if index % 10000 == 0:
    #             print("Creating TRAIN dataframe \t {}/{}".format(index, len(rap_data_with_keypoint)))
    #     else:
    #         image_without_kyepoints += 1
    #         print("Not have Keypoint: {},\t{} doesn't have keypoint and will not be considered in our experiment".format(image_without_kyepoints,img_name))
    #
    # first_row_of_the_pandas_dataframe = [name_of_the_attrs[i] for i in index_of_the_selected_attrs]
    # first_row_of_the_pandas_dataframe.insert(0,"Train_filenames")
    # with open("TRAIN_RAP_pandas_frame_data_format_1.csv", "w", newline='') as CSV:
    #     text = csv.writer(CSV, delimiter=',')
    #     text.writerow(first_row_of_the_pandas_dataframe)
    #     for line in all_imgs_with_selected_labels:
    #         text.writerow(line)



