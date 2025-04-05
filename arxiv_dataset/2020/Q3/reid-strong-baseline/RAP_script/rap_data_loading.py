import os
import cv2
import json
import mat4py
import random
import numpy as np

# paths
Unresized_RAP_dir = "/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Original_RAP_images"
rap_images_dir = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_resized_imgs256x256'
rap_masks_dir = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_resized_masks256x256'
#rap_keypoints_json = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Anchor_level_rap/rap_annotations/RAP_keypoints.json'
rap_attribute_annotations = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Anchor_level_rap/rap_annotations/RAP_annotation.mat'
rap_keypoints_json = "/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/Anchor_level_rap/rap_annotations/RAP_256x256_alphapose-results.json"

input_img_size = (256, 256)

# rap keypoints names
kp_left_ankle = 0
kp_left_knee = 1
kp_left_hip = 2
kp_right_hip = 3
kp_right_knee = 4
kp_right_ankle = 5
kp_pelvis = 6
kp_abdomen = 7
kp_neck = 8
kp_head = 9
kp_left_wrist = 10
kp_left_elbow = 11
kp_left_shoulder = 12
kp_right_shoulder = 13
kp_right_elbow = 14
kp_right_wrist = 15
# end rap keypoint names

# rap attibute names
attr_Female = 0
attr_AgeLess16 = 1
attr_Age17_30 = 2
attr_Age31_45 = 3
attr_Age46_60 = 4
attr_AgeBigger60 = 5
attr_BodyFatter = 6
attr_BodyFat = 7
attr_BodyNormal = 8
attr_BodyThin = 9
attr_BodyThiner = 10
attr_Customer = 11
attr_Employee = 12
attr_hs_BaldHead = 13
attr_hs_LongHair = 14
attr_hs_BlackHair = 15
attr_hs_Hat = 16
attr_hs_Glasses = 17
attr_hs_Sunglasses = 18
attr_hs_Muffler = 19
attr_hs_Mask = 20
attr_ub_Shirt = 21
attr_ub_Sweater = 22
attr_ub_Vest = 23
attr_ub_TShirt = 24
attr_ub_Cotton = 25
attr_ub_Jacket = 26
attr_ub_SuitUp = 27
attr_ub_Tight = 28
attr_ub_ShortSleeve = 29
attr_ub_Others = 30
attr_ub_ColorBlack = 31
attr_ub_ColorWhite = 32
attr_ub_ColorGray = 33
attr_up_ColorRed = 34
attr_ub_ColorGreen = 35
attr_ub_ColorBlue = 36
attr_ub_ColorSilver = 37
attr_ub_ColorYellow = 38
attr_ub_ColorBrown = 39
attr_ub_ColorPurple = 40
attr_ub_ColorPink = 41
attr_ub_ColorOrange = 42
attr_ub_ColorMixture = 43
attr_ub_ColorOther = 44
attr_lb_LongTrousers = 45
attr_lb_Shorts = 46
attr_lb_Skirt = 47
attr_lb_ShortSkirt = 48
attr_lb_LongSkirt = 49
attr_lb_Dress = 50
attr_lb_Jeans = 51
attr_lb_TightTrousers = 52
attr_lb_ColorBlack = 53
attr_lb_ColorWhite = 54
attr_lb_ColorGray = 55
attr_lb_ColorRed = 56
attr_lb_ColorGreen = 57
attr_lb_ColorBlue = 58
attr_lb_ColorSilver = 59
attr_lb_ColorYellow = 60
attr_lb_ColorBrown = 61
attr_lb_ColorPurple = 62
attr_lb_ColorPink = 63
attr_lb_ColorOrange = 64
attr_lb_ColorMixture = 65
attr_lb_ColorOther = 66
attr_shoes_Leather = 67
attr_shoes_Sports = 68
attr_shoes_Boots = 69
attr_shoes_Cloth = 70
attr_shoes_Sandals = 71
attr_shoes_Casual = 72
attr_shoes_Other = 73
attr_shoes_ColorBlack = 74
attr_shoes_ColorWhite = 75
attr_shoes_ColorGray = 76
attr_shoes_ColorRed = 77
attr_shoes_ColorGreen = 78
attr_shoes_ColorBlue = 79
attr_shoes_ColorSilver = 80
attr_shoes_ColorYellow = 81
attr_shoes_ColorBrown = 82
attr_shoes_ColorPurple = 83
attr_shoes_ColorPink = 84
attr_shoes_ColorOrange = 85
attr_shoes_ColorMixture = 86
attr_shoes_ColorOther = 87
attr_attachment_Backpack = 88
attr_attachment_ShoulderBag = 89
attr_attachment_HandBag = 90
attr_attachment_WaistBag = 91
attr_attachment_Box = 92
attr_attachment_PlasticBag = 93
attr_attachment_PaperBag = 94
attr_attachment_HandTrunk = 95
attr_attachment_Baby = 96
attr_attachment_Other = 97
attr_action_Calling = 98
attr_action_StrechOutArm = 99
attr_action_Talking = 100
attr_action_Gathering = 101
attr_action_LyingCounter = 102
attr_action_Squatting = 103
attr_action_Running = 104
attr_action_Holding = 105
attr_action_Pushing = 106
attr_action_Pulling = 107
attr_action_CarryingByArm = 108
attr_action_CarryingByHand = 109
attr_action_Other = 110
attr_viewpoint = 111
attr_OcclusionLeft = 112
attr_OcclusionRight = 113
attr_OcclusionUp = 114
attr_OcclusionDown = 115
attr_occlustion_TypeEnvironment = 116
attr_occlustion_TypeAttachment = 117
attr_occlustion_TypePerson = 118
attr_occlustion_TypeOther = 119
attr_person_position_x = 120
attr_person_position_y = 121
attr_person_position_w = 122
attr_person_position_h = 123
attr_headshoulder_position_x = 124
attr_headshoulder_position_y = 125
attr_headshoulder_position_w = 126
attr_headshoulder_position_h = 127
attr_upperbody_position_x = 128
attr_upperbody_position_y = 129
attr_upperbody_position_w = 130
attr_upperbody_position_h = 131
attr_lowerbody_position_x = 132
attr_lowerbody_position_y = 133
attr_lowerbody_position_w = 134
attr_lowerbody_position_h = 135
attr_attachment1_position_x = 136
attr_attachment1_position_y = 137
attr_attachment1_position_w = 138
attr_attachment1_position_h = 139
attr_attachment2_position_x = 140
attr_attachment2_position_y = 141
attr_attachment2_position_w = 142
attr_attachment2_position_h = 143
attr_attachment3_position_x = 144
attr_attachment3_position_y = 145
attr_attachment3_position_w = 146
attr_attachment3_position_h = 147
attr_attachment4_position_x = 148
attr_attachment4_position_y = 149
attr_attachment4_position_w = 150
attr_attachment4_position_h = 151
# end rap attribute names

rap_attibute_names = {"attr_Female" : 0,
                    "attr_AgeLess16" : 1,
                    "attr_Age17_30" : 2,
                    "attr_Age31_45" : 3,
                    "attr_Age46_60" : 4,
                    "attr_AgeBigger60" : 5,
                    "attr_BodyFatter" : 6,
                    "attr_BodyFat" : 7,
                    "attr_BodyNormal" : 8,
                    "attr_BodyThin" : 9,
                    "attr_BodyThiner" : 10,
                    "attr_Customer" : 11,
                    "attr_Employee" : 12,
                    "attr_hs_BaldHead" : 13,
                    "attr_hs_LongHair" : 14,
                    "attr_hs_BlackHair" : 15,
                    "attr_hs_Hat" : 16,
                    "attr_hs_Glasses" : 17,
                    "attr_hs_Sunglasses" : 18,
                    "attr_hs_Muffler" : 19,
                    "attr_hs_Mask" : 20,
                    "attr_ub_Shirt" : 21,
                    "attr_ub_Sweater" : 22,
                    "attr_ub_Vest" : 23,
                    "attr_ub_TShirt" : 24,
                    "attr_ub_Cotton" : 25,
                    "attr_ub_Jacket" : 26,
                    "attr_ub_SuitUp" : 27,
                    "attr_ub_Tight" : 28,
                    "attr_ub_ShortSleeve" : 29,
                    "attr_ub_Others" : 30,
                    "attr_ub_ColorBlack" : 31,
                    "attr_ub_ColorWhite" : 32,
                    "attr_ub_ColorGray" : 33,
                    "attr_up_ColorRed" : 34,
                    "attr_ub_ColorGreen" : 35,
                    "attr_ub_ColorBlue" : 36,
                    "attr_ub_ColorSilver" : 37,
                    "attr_ub_ColorYellow" : 38,
                    "attr_ub_ColorBrown" : 39,
                    "attr_ub_ColorPurple" : 40,
                    "attr_ub_ColorPink" : 41,
                    "attr_ub_ColorOrange" : 42,
                    "attr_ub_ColorMixture" : 43,
                    "attr_ub_ColorOther" : 44,
                    "attr_lb_LongTrousers" : 45,
                    "attr_lb_Shorts" : 46,
                    "attr_lb_Skirt" : 47,
                    "attr_lb_ShortSkirt" : 48,
                    "attr_lb_LongSkirt" : 49,
                    "attr_lb_Dress" : 50,
                    "attr_lb_Jeans" : 51,
                    "attr_lb_TightTrousers" : 52,
                    "attr_lb_ColorBlack" : 53,
                    "attr_lb_ColorWhite" : 54,
                    "attr_lb_ColorGray" : 55,
                    "attr_lb_ColorRed" : 56,
                    "attr_lb_ColorGreen" : 57,
                    "attr_lb_ColorBlue" : 58,
                    "attr_lb_ColorSilver" : 59,
                    "attr_lb_ColorYellow" : 60,
                    "attr_lb_ColorBrown" : 61,
                    "attr_lb_ColorPurple" : 62,
                    "attr_lb_ColorPink" : 63,
                    "attr_lb_ColorOrange" : 64,
                    "attr_lb_ColorMixture" : 65,
                    "attr_lb_ColorOther" : 66,
                    "attr_shoes_Leather" : 67,
                    "attr_shoes_Sports" : 68,
                    "attr_shoes_Boots" : 69,
                    "attr_shoes_Cloth" : 70,
                    "attr_shoes_Sandals" : 71,
                    "attr_shoes_Casual" : 72,
                    "attr_shoes_Other" : 73,
                    "attr_shoes_ColorBlack" : 74,
                    "attr_shoes_ColorWhite" : 75,
                    "attr_shoes_ColorGray" : 76,
                    "attr_shoes_ColorRed" : 77,
                    "attr_shoes_ColorGreen" : 78,
                    "attr_shoes_ColorBlue" : 79,
                    "attr_shoes_ColorSilver" : 80,
                    "attr_shoes_ColorYellow" : 81,
                    "attr_shoes_ColorBrown" : 82,
                    "attr_shoes_ColorPurple" : 83,
                    "attr_shoes_ColorPink" : 84,
                    "attr_shoes_ColorOrange" : 85,
                    "attr_shoes_ColorMixture" : 86,
                    "attr_shoes_ColorOther" : 87,
                    "attr_attachment_Backpack" : 88,
                    "attr_attachment_ShoulderBag" : 89,
                    "attr_attachment_HandBag" : 90,
                    "attr_attachment_WaistBag" : 91,
                    "attr_attachment_Box" : 92,
                    "attr_attachment_PlasticBag" : 93,
                    "attr_attachment_PaperBag" : 94,
                    "attr_attachment_HandTrunk" : 95,
                    "attr_attachment_Baby" : 96,
                    "attr_attachment_Other" : 97,
                    "attr_action_Calling" : 98,
                    "attr_action_StrechOutArm" : 99,
                    "attr_action_Talking" : 100,
                    "attr_action_Gathering" : 101,
                    "attr_action_LyingCounter" : 102,
                    "attr_action_Squatting" : 103,
                    "attr_action_Running" : 104,
                    "attr_action_Holding" : 105,
                    "attr_action_Pushing" : 106,
                    "attr_action_Pulling" : 107,
                    "attr_action_CarryingByArm" : 108,
                    "attr_action_CarryingByHand" : 109,
                    "attr_action_Other" : 110,
                    "attr_viewpoint" : 111,
                    "attr_OcclusionLeft" : 112,
                    "attr_OcclusionRight" : 113,
                    "attr_OcclusionUp" : 114,
                    "attr_OcclusionDown" : 115,
                    "attr_occlustion_TypeEnvironment" : 116,
                    "attr_occlustion_TypeAttachment" : 117,
                    "attr_occlustion_TypePerson" : 118,
                    "attr_occlustion_TypeOther" : 119,
                    "attr_person_position_x" : 120,
                    "attr_person_position_y" : 121,
                    "attr_person_position_w" : 122,
                    "attr_person_position_h" : 123,
                    "attr_headshoulder_position_x" : 124,
                    "attr_headshoulder_position_y" : 125,
                    "attr_headshoulder_position_w" : 126,
                    "attr_headshoulder_position_h" : 127,
                    "attr_upperbody_position_x" : 128,
                    "attr_upperbody_position_y" : 129,
                    "attr_upperbody_position_w" : 130,
                    "attr_upperbody_position_h" : 131,
                    "attr_lowerbody_position_x" : 132,
                    "attr_lowerbody_position_y" : 133,
                    "attr_lowerbody_position_w" : 134,
                    "attr_lowerbody_position_h" : 135,
                    "attr_attachment1_position_y" : 137,
                    "attr_attachment1_position_w" : 138,
                    "attr_attachment1_position_h" : 139,
                    "attr_attachment2_position_x" : 140,
                    "attr_attachment2_position_y" : 141,
                    "attr_attachment2_position_w" : 142,
                    "attr_attachment2_position_h" : 143,
                    "attr_attachment3_position_x" : 144,
                    "attr_attachment3_position_y" : 145,
                    "attr_attachment3_position_w" : 146,
                    "attr_attachment3_position_h" : 147,
                    "attr_attachment4_position_x" : 148,
                    "attr_attachment4_position_y" : 149,
                    "attr_attachment4_position_w" : 150,
                    "attr_attachment4_position_h" : 151}


def load_crop_rap_mask(mask_image_path):
    top_border = 200
    bottom_border = 200
    left_border = 237
    right_border = 237

    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mh, mw =  mask.shape[0], mask.shape[1]
    mask_cropped = mask[top_border:mh - bottom_border,
                   left_border: mw - right_border]
    return mask_cropped


def load_rap_keypoints(rap_keypoints_json_path, map_keypoints = False, Unresized_RAP_dir = None, map_to_size=None):
    with open(rap_keypoints_json_path) as f:
        data = json.load(f)

        rap_images_keypoints = dict()
        for indx, echfile in enumerate(data):
            img_name = data[indx]['image_id']
            img_keypoints_data = data[indx]['keypoints']
            img_keypoints = []

            if map_keypoints:
                if (map_to_size or Unresized_RAP_dir) is None:
                    raise ValueError(" map_to_size is None while it should be a tuple indicating the size of the target image")
                from RAP_script.Step2_RAP_resize_and_map_kepoints import X_Y_coordinate_evaluation
                for pt_idx in range(0, len(img_keypoints_data), 3):
                    x = int(img_keypoints_data[pt_idx])
                    y = int(img_keypoints_data[pt_idx + 1])
                    image_array = cv2.imread(os.path.join(Unresized_RAP_dir,img_name), cv2.IMREAD_GRAYSCALE)
                    if image_array is None:
                        print("Warning: Image is not found")
                        continue

                    flag, x_prime, y_prime = X_Y_coordinate_evaluation (X_coordinate= x, Y_coordinate= y, image_name= img_name, img= image_array, resize_scale= input_img_size)
                    img_keypoints.append((int(x_prime), int(y_prime)))
                    # oldX, oldY = image_array.shape
                    # newX, newY = map_to_size
                    # Rx = newX/oldX
                    # Ry = newY/oldY
                    # mapped_X = round(Rx * x)
                    # mapped_Y = round(Ry * y)
                    # mapped_pt = (int(mapped_X), int(mapped_Y))
                    # img_keypoints.append(mapped_pt)

                rap_images_keypoints[img_name] = img_keypoints
                if indx == 100:
                    #print("mapping keypoints from {} to {} image size: \t{}/{}".format((oldX,oldY), (newX,newY), indx, len(data)))
                    break
            else:
                for pt_idx in range(0, len(img_keypoints_data), 3):
                    pt = (int(img_keypoints_data[pt_idx]), int(img_keypoints_data[pt_idx + 1]))
                    img_keypoints.append(pt)

                rap_images_keypoints[img_name] = img_keypoints

    return rap_images_keypoints

def load_rap_keypoints_v2(keypoints_json_path, load_previouse_processed_keypoints_from_disk):
    # TODO: Some images have several persons. I have selected the target person as one with the most score.

    if load_previouse_processed_keypoints_from_disk:
        cd = os.path.dirname(os.path.realpath(__file__))
        is_the_file_exists = os.path.isfile(os.path.join(cd,"RAP_processed_keypoints.json"))
        if is_the_file_exists:
            with open(os.path.join(cd,"RAP_processed_keypoints.json"), "r") as file:
                images_keypoints = json.load(file)
            return images_keypoints
        else:
            print(">> The keypoints file needs to be processed first ...")

    with open(keypoints_json_path) as f:
        data = json.load(f)
        images_keypoints = dict()
        all_names = []
        all_scores = []
        for index, keypoint in enumerate(data):
            image_name = keypoint["image_id"]
            score = keypoint["score"]
            all_names.append(image_name)
            all_scores.append(score)
        unique_img_name = set(all_names)
        for index, name in enumerate(unique_img_name):
            indices_with_this_name = [i for i, x in enumerate(all_names) if x == name]
            scores_for_this_img = [all_scores[i] for i in indices_with_this_name]
            index_of_max_score = np.argmax(scores_for_this_img)
            index_of_the_person_with_max_score_in_this_image = indices_with_this_name [int(index_of_max_score)]
            target_person_keypoints = data[index_of_the_person_with_max_score_in_this_image]["keypoints"]

            if index % 1000 == 0:
                print(">> Keypoints pre-processing:\t{}/{}".format(index,len(unique_img_name)))

            img_keypoints = []
            for pt_idx in range(3, len(target_person_keypoints), 3):
                pt = (round(target_person_keypoints[pt_idx]), round(target_person_keypoints[pt_idx + 1]))
                img_keypoints.append(pt)
            images_keypoints[name] = img_keypoints

    with open("RAP_processed_keypoints.json", "w") as file:
        json.dump(images_keypoints, file)
        cd = os.path.dirname(os.path.realpath(__file__))
        print(">> RAP keypoints are processed and saved as *RAP_processed_keypoints* on the disk at: {} ".format(os.path.join(cd,"RAP_processed_keypoints.json")))
    return images_keypoints

def load_rap_attributes(rap_mat_file):
    rap_data = mat4py.loadmat(rap_mat_file)
    # attributes = rap_data['RAP_annotation']['attribute']
    # attributes = np.asarray(attributes)
    # attributes = list(np.squeeze(attributes))

    #names = list(np.squeeze(np.asarray(rap_data['RAP_annotation']['name'])))
    names = list(np.squeeze(np.asarray(rap_data['RAP_annotation']['name'])))
    data = rap_data['RAP_annotation']['data']
    return names, data


def get_images_with_attib(rap_data, attrib_index, attrib_value):
    names_with_attrib = [image_name for image_name in rap_data if rap_data[image_name]['attrs'][attrib_index] == attrib_value]
    return names_with_attrib

def load_rap_dataset(rap_attributes_filepath, rap_keypoints_json, edit_position_attributs= True, load_from_file=True):

    if load_from_file:
        print(" >>> It is declared to load the rap_data.json form disk ...")
        cd = os.path.dirname(os.path.realpath(__file__))
        is_the_file_exists = os.path.isfile(os.path.join(cd,"rap_data.json"))
        if is_the_file_exists:
            with open(os.path.join(cd,"rap_data.json"), "r") as file:
                rap_data = json.load(file)
                print(">> *rap_data* is loaded from disk successfully")
                return rap_data
        else:
            print(">> Processed data is not found. Data needs to be processed first.")

    print(">> A dictionary will be created as: \n{\"name_of_the_image.png\":\n {\"keypoints\": [[188, 84], ..., [148, 94]],\n \"attrs\": [0, ..., 1], \n \"attibute_names\": {\"attr_OcclusionRight\": 113, ...}}}")
    image_names, attributes = load_rap_attributes(rap_attributes_filepath)
    rap_keypoints_data = load_rap_keypoints_v2(keypoints_json_path = rap_keypoints_json, load_previouse_processed_keypoints_from_disk = False)
    #rap_keypoints_data = load_rap_keypoints(rap_keypoints_json_path=rap_keypoints_json, map_keypoints=True, Unresized_RAP_dir=Unresized_RAP_dir, map_to_size=input_img_size)

    rap_data = dict()
    print(">> Creating the dictionary ...")
    for idx, image_name in enumerate(image_names):
        if image_name not in rap_keypoints_data:
            continue
        if edit_position_attributs:
            img_attribute_vector = attributes[idx].copy()
            img_attribute_vector = map_points_to_a_new_squared_size(img_attribute_vector, input_img_size)
        else:
            img_attribute_vector = attributes[idx]

        rap_data[image_name] = dict()
        rap_data[image_name]['attrs'] = img_attribute_vector
        rap_data[image_name]['keypoints'] = rap_keypoints_data[image_name]
        rap_data[image_name]["attibute_names"] = rap_attibute_names

    # writing the processed data to the disk
    cd = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cd,"rap_data.json")
    with open(path, 'w') as outfile:
        json.dump(rap_data, outfile)
        print(">> *rap_data* is processed and saved on the disk at: {} ".format(path))
        print(">> The form of the saved dictionary is as: \n"
              "{\"name_of_the_image.png\":\n "
              "{\"keypoints\": [[188, 84], ..., [148, 94]],\n "
              "\"attrs\": [0, ..., 1], \n "
              "\"attibute_names\": {\"attr_OcclusionRight\": 113, ...}}}")

    return rap_data

def get_head_bbox(attrs):

    head_brect = (int(attrs[attr_headshoulder_position_x] - attrs[attr_person_position_x]),
                  int(attrs[attr_headshoulder_position_y] - attrs[attr_person_position_y]),
                  int(attrs[attr_headshoulder_position_w]),
                  int(attrs[attr_headshoulder_position_h]))
    return head_brect

def get_head_torso_bbox(attrs):

    head_torso_area_tl = (int(attrs[attr_person_position_x]), int(attrs[attr_headshoulder_position_y]))
    head_torso_area_br = (int(attrs[attr_upperbody_position_x] + int(attrs[attr_upperbody_position_w])),
                          int(attrs[attr_upperbody_position_y] + int(attrs[attr_upperbody_position_h])))
    head_torso_w = int(head_torso_area_br[0] - head_torso_area_tl[0])
    head_torso_h = int(head_torso_area_br[1] - head_torso_area_tl[1])

    head_torso_brect = (int(head_torso_area_tl[0] - attrs[attr_person_position_x]),
                        int(head_torso_area_tl[1] - attrs[attr_person_position_y]),
                        head_torso_w,
                        head_torso_h)

    # head_torso_brect = (min(int(attrs[attr_headshoulder_position_x]),int(attrs[attr_upperbody_position_x])),
    #                     max(int(attrs[attr_headshoulder_position_y]),int(attrs[attr_upperbody_position_y])),
    #                     (int(attrs[attr_headshoulder_position_w])+int(attrs[attr_upperbody_position_w])),
    #                     (int(attrs[attr_headshoulder_position_h])+int(attrs[attr_upperbody_position_h])))

    return head_torso_brect

def get_whole_body_bbox(attrs):
    whole_body_rect = (int(attrs[attr_person_position_x]),
                       int(attrs[attr_person_position_y]),
                       int(attrs[attr_person_position_w]),
                       int(attrs[attr_person_position_h]))
    return whole_body_rect


def train_test_imgs(rap_mat_file, partition):
    RAP_dictionary = mat4py.loadmat(rap_mat_file)
    partition_attribute = RAP_dictionary["RAP_annotation"]["partition_attribute"][partition]
    RAP_name_of_imgs = RAP_dictionary["RAP_annotation"]["name"]
    train_index = partition_attribute["train_index"] + partition_attribute["val_index"]
    test_index = partition_attribute["test_index"]
    print(">> Splitting test and train ...")
    train_imgs = []
    for i, IMG_index in enumerate(train_index):
        img_name = RAP_name_of_imgs[IMG_index-1]
        train_imgs.append(img_name[0])
    test_imgs = []
    for i, IMG_index in enumerate(test_index):
        img_name = RAP_name_of_imgs[IMG_index-1]
        test_imgs.append(img_name[0])
    return train_imgs, test_imgs

def load_reid_data(rap_mat_file):
    """
    For each person in RAP dataset, we have sampled one image from one camera at one day as the query,
    so one person may have several queries from one camera in different days. To change this setting,
    we refer you to https://github.com/dangweili/RAP/blob/master/person-reid/evaluation/rap2_evaluation_features.m
    """
    RAP_dictionary = mat4py.loadmat(rap_mat_file)
    name_of_imgs = RAP_dictionary["RAP_annotation"]["name"]
    all_person_ids = RAP_dictionary["RAP_annotation"]["person_identity"][:41585]
    train_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["train_index"]
    gallery_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["gallery_index"]
    query_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["query_index"]

    name_of_imgs = [subsub for sub in name_of_imgs for subsub in sub]
    all_person_ids = [subsub for sub in all_person_ids for subsub in sub]
    train_ids_indices = [subsub for sub in train_ids_indices for subsub in sub]
    gallery_ids_indices = [subsub for sub in gallery_ids_indices for subsub in sub]
    #query_ids_indices = [subsub for sub in query_ids_indices for subsub in sub]

    return name_of_imgs, all_person_ids, train_ids_indices, gallery_ids_indices, query_ids_indices

def load_rap_dataset_from_disk(data_dir):
    _data = open(data_dir,"r")
    if _data.mode == "r":
        rap_data_from_disk = _data.read()
    else:
        rap_data_from_disk = None
    return rap_data_from_disk

def map_points_to_a_new_squared_size(attrs, desired_size):
    # get original image size
    w_img_old, h_img_old = attrs[122], attrs[123]
    try:
        assert h_img_old > w_img_old
    except AssertionError:
        print("Warning: image_weight > image_height") # to elaviate these cases, first check the output of resizeAndpad(), then add an if statement here to add 'compensate_squared' variable to the right side of the image
        print("h_img_old: {}, w_img_old: {} ".format(h_img_old, w_img_old))
    # Regardless of adding borders to make the image squared, if height becomes let's say 256, what is the new value for the weight?
    factor = h_img_old/desired_size[1]
    # what is new size
    h_img_new = desired_size[1] # Maybe it should be like: (h_img_new = h_img_old/factor) in which (factor=w_img_old/desired_size[2])
    w_img_new = w_img_old/factor
    # calculate the amount of addition value in x-axis when the image become squared
    compensate_squared = abs((w_img_old - h_img_old) / 2)

    new_atribute_vector = attrs.copy()
    vlaue1 = 124
    vlaue2 = 125
    vlaue3 = 126
    vlaue4 = 127
    while (vlaue4<136):
        x =  abs(attrs[vlaue1] - attrs[120])
        y =  abs(attrs[vlaue2] - attrs[121])
        w = attrs[vlaue3]
        h = attrs[vlaue4]

        # map the points
        x_prime = (x + compensate_squared) * (w_img_new / w_img_old)
        y_prime = (y * (h_img_new / h_img_old))
        # map the width and height of the bbox to new size.
        # calculate the bottom-right coordinates and map them to the new size
        x_b = w + x
        y_b = h + y
        x_b_prime = (x_b + compensate_squared) * (w_img_new / w_img_old)
        y_b_prime = (y_b * (h_img_new / h_img_old))

        # calculate width and height of the bbox in new size
        w_prime = abs(x_b_prime - x_prime)
        h_prime = abs(y_b_prime - y_prime)
        # update the positional values in annotation file
        new_atribute_vector[vlaue1] = int(round(x_prime))
        new_atribute_vector[vlaue2] = int(round(y_prime))
        new_atribute_vector[vlaue3] = int(round(w_prime))
        new_atribute_vector[vlaue4] = int(round(h_prime))

        vlaue1 += 4
        vlaue2 += 4
        vlaue3 += 4
        vlaue4 += 4

    # change the person_position to (0,0) and person bbox to desired_size
    new_atribute_vector[120] = 0 # maybe it should be compensate_squared
    new_atribute_vector[121] = 0
    new_atribute_vector[122] = w_img_new
    new_atribute_vector[123] = h_img_new

    return new_atribute_vector

if __name__ == '__main__':

    rap_data = load_rap_dataset(rap_attributes_filepath = rap_attribute_annotations, rap_keypoints_json= rap_keypoints_json, load_from_file=False)

    for img_name in rap_data:
        img_path = os.path.join(rap_images_dir, img_name)
        mask_path = os.path.join(rap_masks_dir, img_name)
        keypoints = rap_data[img_name]['keypoints']
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #mask = load_crop_rap_mask(mask_path)

        if img is None or mask is None:
            print('Error! Could not find image or mask for ', img_name)
            continue

        assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]

        for pt in keypoints:
            cv2.circle(img, tuple(pt), 3, (0, 255, 0))
        #cv2.destroyWindow('keypoints')
        cv2.imshow('keypoints', img)
        cv2.imshow('mask', mask)
        cv2.waitKey()

    # img_name = get_images_with_attib(rap_data, attrib_index=1, attrib_value=1)
    # for im in img_name:
    #     img_path = os.path.join(rap_images_dir, im)
    #     img = cv2.imread(img_path)
    #     cv2.imshow('img', img)
    #     cv2.waitKey()
    #
    # for img_name in rap_data:
    #
    #     img_path = os.path.join(rap_images_dir, img_name)
    #     mask_path = os.path.join(rap_masks_dir, img_name)
    #     keypoints = rap_data[img_name]['keypoints']
    #     img = cv2.imread(img_path)
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     #mask = load_crop_rap_mask(mask_path)
    #
    #     if img is None or mask is None:
    #         print('Error! Could nou find image or mask for ', img_name)
    #         continue
    #
    #     assert mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]
    #
    #
    #     for pt in keypoints:
    #         cv2.circle(img, pt, 3, (0, 255, 0))
    #     cv2.destroyWindow('keypoints')
    #     cv2.imshow('keypoints', img)
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey()
