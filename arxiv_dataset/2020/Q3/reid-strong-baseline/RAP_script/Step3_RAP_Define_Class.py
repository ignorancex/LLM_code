import os
import mat4py



def DefineClassRAP (traitIndex, IMG_name, RAP_dictionary): # This function returns the class as a list of integers
    class_ = []
    RAP_img_name = IMG_name.split(".")[0]
    img_number = int(RAP_img_name)

    annotation_data_dictionary = RAP_dictionary["RAP"]
    RAP_Annotations_list = annotation_data_dictionary["data"]
    Annotations_of_this_image = RAP_Annotations_list[img_number-1] # Image names start from 00001, so we need -1 here.
    _Gender_class_of_this_image = Annotations_of_this_image[traitIndex]

    return _Gender_class_of_this_image


RAP_Train_dir = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_Images/RAP_175x100/Test/"
List_of_images = os.listdir(RAP_Train_dir)
Annot_mat_dir = "/media/ehsan/48BE4782BE476810/MyPythonCodes/RAP/RAP_Images/Original_RAP_with_keypoints/RAP.mat"
RAP_dictionary = mat4py.loadmat(Annot_mat_dir)

# 17 is male, to understand look at the RAP.mat, data.attribute, data.annotations
# To use in Python we need: 17-1+4 = 20 ## 17 refers to male, -1 is because of in python (in contrast with Matlab)
# index is started from zero, we add by 4 because the first four column of the annotation files are not
# the annotation but the name of the file, name of the dataset, etc.
for i, IMG_name in enumerate(List_of_images):
    myclass = DefineClassRAP (20, IMG_name, RAP_dictionary)  # 17 is male
    print(myclass)
