import cv2
import os
import json
import numpy as np
def load_rap_keypoints_v2(keypoints_json_path, load_previouse_processed_from_disk):
    # TODO: Some images have several persons. I have selected the target person as one with the most score.

    if load_previouse_processed_from_disk:
        cd = os.path.dirname(os.path.realpath(__file__))
        is_the_file_exists = os.path.isfile(os.path.join(cd,"saved_file.json"))
        if is_the_file_exists:
            with open(os.path.join(cd,"saved_file.json"), "r") as file:
                images_keypoints = json.load(file)
            return images_keypoints
        else:
            print(">> The original keypoints file needs to be processed first ...")

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

    with open("saved_file.json", "w") as file:
        json.dump(images_keypoints, file)
    return images_keypoints

if __name__ == "__main__":
    keypoints_json_path = "/media/ehsan/HDD2TB/AlphaPose/examples/res/test.json"
    img_path = "/media/ehsan/HDD2TB/AlphaPose/examples/demo/test/1.jpg"
    image_name = img_path.split("/")[-1]
    KEYPOINTS = load_rap_keypoints_v2(keypoints_json_path = keypoints_json_path, load_previouse_processed_from_disk = False)
    img = cv2.imread(img_path)

    for pt in KEYPOINTS[image_name]:
        cv2.circle(img, tuple(pt), 3, (0, 255, 0))
        cv2.imshow('keypoints', img)
        cv2.waitKey()

