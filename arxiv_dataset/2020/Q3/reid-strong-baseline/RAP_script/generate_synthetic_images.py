import os
import cv2
import utils
import random
import numpy as np
from RAP_script import utils
from RAP_script import rap_data_loading as rap
import time


def get_head_area(img, mask, keypoints):
    assert img is not None and mask is not None

    head_point = keypoints[rap.kp_head]
    neck_point = keypoints[rap.kp_neck]
    left_elbow = keypoints[rap.kp_left_elbow]
    right_elbow = keypoints[rap.kp_right_elbow]

    # @todo: treat case when persons are viewed from the back; the left and right side are reversed
    if left_elbow[0] > right_elbow[0]:
        left_elbow, right_elbow = right_elbow, left_elbow

    # head bounding rect in format: [x, y, w, h]
    approx_head_brect = [left_elbow[0],
                         0,
                         right_elbow[0] - left_elbow[0],
                         neck_point[1]]

    head_area_mask = mask[approx_head_brect[1]: approx_head_brect[1] + approx_head_brect[3],
                          approx_head_brect[0]: approx_head_brect[0] + approx_head_brect[2]]
    contours,  _ = cv2.findContours(head_area_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # In recent version (4.1.2.30), this fcn returns two values

    # no head contours found
    if len(contours) == 0:
        return None, None, None

    head_contour = contours[0]
    head_contour = utils.translate_points(points = head_contour,
                                          translate_factor=(approx_head_brect[0], approx_head_brect[1]))

    # Find bounding rectangle for HEAD coordinates
    head_brect = cv2.boundingRect(head_contour)
    head_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(head_mask, [head_contour], -1, (255, 255, 255), -1)
    # cv2.destroyWindow('head contour')
    # cv2.imshow('head contour', head_mask)
    # cv2.waitKey(0)
    return head_brect, head_contour, head_mask


def remove_area(img, head_brect, head_mask):
    kernel = np.ones((7, 7), np.uint8)
    mask_area_enlarged = cv2.dilate(head_mask, kernel, iterations=5)
    img_area_removed = cv2.inpaint(img, mask_area_enlarged, head_brect[2]*0.1 , cv2.INPAINT_TELEA)

    return img_area_removed
    #return mask_area_enlarged


def replace_head_rect(img1, mask1, keypoints1, attrs1,
                     img2, mask2, keypoints2, attrs2):

    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        #print("Warning: Bounding Head rectangles of this set of images are NONE")
        return None

    # head_brect1 = (int(attrs1[rap.attr_headshoulder_position_x] - attrs1[rap.attr_person_position_x]), int(attrs1[rap.attr_headshoulder_position_y] - attrs1[rap.attr_person_position_y]),
    #                int(attrs1[rap.attr_headshoulder_position_w]), int(attrs1[rap.attr_headshoulder_position_h]))
    #
    # head_brect2 = (int(attrs2[rap.attr_headshoulder_position_x]  - attrs2[rap.attr_person_position_x]), int(attrs2[rap.attr_headshoulder_position_y] - attrs2[rap.attr_person_position_y]),
    #                int(attrs2[rap.attr_headshoulder_position_w]), int(attrs2[rap.attr_headshoulder_position_h]))

    # @todo - test
    # result_image = remove_area(img1, head_brect1, head_mask1)
    result_image = np.copy(img1)
    r = remove_area(result_image, head_brect1, head_mask1)
    result_image = utils.copy_roi(r, result_image, head_brect1)

    head_img2 = img2[head_brect2[1]: head_brect2[1]+head_brect2[3],
                    head_brect2[0]: head_brect2[0]+head_brect2[2]]
    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))

    result_image[head_brect1[1]:head_brect1[1]+head_brect1[3],
                    head_brect1[0]:head_brect1[0]+head_brect1[2]] = head_img2

    # cv2.rectangle(result_image, (head_brect1[0], head_brect1[1]),
    #               (head_brect1[0] + head_brect1[2], head_brect1[1] + head_brect1[3]), (0, 255, 0), 2)

    return result_image


def replace_head_area(img1, mask1, keypoints1,
                     img2, mask2, keypoints2):

    head_brect1, head_contour1, head_mask1 = get_head_area(img1, mask1, keypoints1)
    head_brect2, head_contour2, head_mask2 = get_head_area(img2, mask2, keypoints2)

    if head_brect1 is None or head_brect2 is None:
        return None

    # @todo - test
    # result_image = remove_area(img1, head_brect1, head_mask1)

    result_image = np.copy(img1)
    r = remove_area(img = result_image, head_brect = head_brect1, head_mask = head_mask1)
    result_image = utils.copy_roi(src = r, dst = result_image, roi = head_brect1)

    head_img2 = img2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]
    mask_img2 = mask2[head_brect2[1]: head_brect2[1] + head_brect2[3],
                head_brect2[0]: head_brect2[0] + head_brect2[2]]

    head_img2 = cv2.resize(head_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.resize(mask_img2, (head_brect1[2], head_brect1[3]))
    mask_img2 = cv2.cvtColor(mask_img2, cv2.COLOR_GRAY2BGR)
    mask_img2 = mask_img2.astype(np.bool)

    np.copyto(dst = result_image[head_brect1[1]: head_brect1[1] + head_brect1[3], head_brect1[0]: head_brect1[0] + head_brect1[2]],
              src = head_img2,
              where=mask_img2,
              casting='unsafe')

    return result_image


def replace_head_torso_area(img1, mask1, keypoints1, attrs1,
                            img2, mask2, keypoints2, attrs2):
    head_torso_rect1 = rap.get_head_torso_bbox(attrs1)
    head_torso_rect2 = rap.get_head_torso_bbox(attrs2)

    # head_torso_img1 = utils.get_roi(img1, head_torso_rect1)
    # head_torso_img2 = utils.get_roi(img2, head_torso_rect2)

    head_torso_mask1 = utils.get_mask_for_roi(img = mask1, rect = head_torso_rect1)
    head_torso_mask2 = utils.get_mask_for_roi(img = mask2, rect = head_torso_rect2)

    #if head_torso_mask1 is None or head_torso_mask2 is None or head_torso_rect1 is None or head_torso_rect2 is None or head_torso_img1 is None or head_torso_img2 is None:
        #return None

    # @todo:  can add here multiple constraint functions on masks (like orientation, center etc).
    enable_constraints_ht = False
    if enable_constraints_ht:
        if not is_compatible_iou(None, None, head_torso_mask1, head_torso_mask2, None, None, None, None):
            print('Incompatible head torso masks by IOU')
            return None

        if not is_compatible_area(None, None, head_torso_mask1, head_torso_mask2, None, None, None, None, th_area=0.5):
            print('Incompatible head torso masks by area')
            return None
    # end

    # cv2.imshow('ht', head_torso_img1)
    # cv2.imshow('htm', head_torso_mask1)
    # cv2.waitKey()

    result_image = np.copy(img2)
    r = remove_area(result_image, head_torso_rect1, head_torso_mask1)
    result_image = utils.copy_roi(src=r, dst=result_image, roi=head_torso_rect1)

    ht_img1 = img1[head_torso_rect1[1]: head_torso_rect1[1] + head_torso_rect1[3],
                   head_torso_rect1[0]: head_torso_rect1[0] + head_torso_rect1[2]]

    mask_img1 = mask1[head_torso_rect1[1]: head_torso_rect1[1] + head_torso_rect1[3],
                      head_torso_rect1[0]: head_torso_rect1[0] + head_torso_rect1[2]]


    #if ht_img2 is None or mask_img2 is None or result_image is None:
        #return None

    ht_img1 = cv2.resize(src=ht_img1, dsize=(head_torso_rect2[2], head_torso_rect2[3]))
    mask_img1 = cv2.resize(src=mask_img1, dsize=(head_torso_rect2[2], head_torso_rect2[3]))
    mask_img1 = cv2.cvtColor(src=mask_img1, code=cv2.COLOR_GRAY2BGR)
    mask_img1 = mask_img1.astype(np.bool)

    np.copyto(dst=result_image[head_torso_rect2[1]: head_torso_rect2[1] + head_torso_rect2[3],
                               head_torso_rect2[0]: head_torso_rect2[0] + head_torso_rect2[2]],
              src=ht_img1,
              where=mask_img1,
              casting='unsafe')

    return result_image


def replace_background(img1, mask1,
                       img2, mask2):

    # @todo:  can add here multiple constraint functions on masks (like orientation, center etc).
    enable_constraints_ht = False
    if enable_constraints_ht:
        if not is_compatible_iou(None, None, mask1, mask2, None, None, None, None):
            print('Incompatible head torso masks by IOU')
            return None

        if not is_compatible_area(None, None, mask1, mask2, None, None, None, None, th_area=0.5):
            print('Incompatible head torso masks by area')
            return None

    # x1, y1, w1, h1 = cv2.boundingRect(mask1)
    x2, y2, w2, h2 = cv2.boundingRect(mask2)
    # whole_body_rect1 = [x1, y1, w1, h1]
    whole_body_rect2 = [x2, y2, w2, h2]

    result_image = np.copy(img2)
    result_image = remove_area(result_image, whole_body_rect2, mask2)
    # cv2.imshow('result_image_2', result_image)
    mask_img1 = cv2.cvtColor(src=mask1, code=cv2.COLOR_GRAY2BGR)
    mask_img1 = mask_img1.astype(np.bool)

    np.copyto(dst=result_image,
              src=img1,
              where=mask_img1,
              casting='unsafe')

    # cv2.imshow('result_image_final', result_image)
    # cv2.imshow('img2', img2)
    # cv2.imshow('mask2', mask2)
    # cv2.waitKey()
    return result_image
# ---------------------------------------------------
# constrain functions
# ---------------------------------------------------
def is_compatible_area(img1, img2, mask1, mask2,
                       kp1, kp2, attr1, attr2, th_area = 0.8):
    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)

    if min(area1, area2)/max(area1, area2) < th_area:
        return False
    return True


def is_compatible_iou(img1, img2, mask1, mask2,
                      kp1, kp2, attr1, attr2, iou_threshold = 0.5):
    mask1_aligned, mask2_aligned, _, _ = utils.align_images_width(mask1, mask2)
    iou = utils.compute_mask_iou(mask1_aligned, mask2_aligned)
    # print('iou is ', iou)
    if iou < iou_threshold:
        return False

    return True

# ---------------------------------------------------
# end constraint functions
# ---------------------------------------------------

def generate_syntethic_images(rap_data, num_images_to_generate, viewpoint = 1, other_attrs = None,
                              constraint_functions = []):

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data, rap.attr_OcclusionUp, 0))
    target_images = set(rap.get_images_with_attib(rap_data, rap.attr_viewpoint, viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)

    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = rap.get_images_with_attib(rap_data, attr, other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)

    for idx in range(0, num_images_to_generate):
        cv2.destroyAllWindows()

        img_name1 = random.choice(target_images)
        img_name2 = random.choice(target_images)

        img_path1 = os.path.join(rap.rap_images_dir, img_name1)
        mask_path1 = os.path.join(rap.rap_masks_dir, img_name1)
        keypoints1 = rap_data[img_name1]['keypoints']
        attr1 = rap_data[img_name1]['attrs']
        img1 = cv2.imread(img_path1)
        mask1 = cv2.imread(mask_path1,cv2.IMREAD_GRAYSCALE)
        #mask1 = rap.load_crop_rap_mask(mask_path1)

        img_path2 = os.path.join(rap.rap_images_dir, img_name2)
        mask_path2 = os.path.join(rap.rap_masks_dir, img_name2)
        keypoints2 = rap_data[img_name2]['keypoints']
        attr2 = rap_data[img_name2]['attrs']
        img2 = cv2.imread(img_path2)
        mask2 = cv2.imread(mask_path2,cv2.IMREAD_GRAYSCALE)
        #mask2 = rap.load_crop_rap_mask(mask_path2)

        assert mask1.shape[0] == img1.shape[0] and mask2.shape[1] == img2.shape[1]

        for constraint_function in constraint_functions:
            if not constraint_function(img1, img2, mask1, mask2, keypoints1, keypoints2,
                                       attr1, attr2):
                idx -= 1
                continue

        generated_replaced_area = replace_head_area(img1, mask1, keypoints1,
                                                    img2, mask2, keypoints2)
        generated_replaced_rect = replace_head_rect(img1, mask1, keypoints1, attr1,
                                                    img2, mask2, keypoints2, attr2)

        generated_replaced_ht = replace_head_torso_area(img1, mask1, keypoints1, attr1,
                                                        img2, mask2, keypoints2, attr2)

        # # display
        # img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # concat_images = [img1, img2_display]
        # if generated_replaced_area is not None:
        #     concat_images.append(generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     concat_images.append(generated_replaced_rect)
        # if generated_replaced_ht is not None:
        #     concat_images.append(generated_replaced_ht)
        #
        # display_img = cv2.hconcat(concat_images)
        # cv2.imshow('morphing', display_img)
        # cv2.waitKey()
    return

def generate_images_from_this_image (image_name, rap_data___= None, constraint_functions= None, other_attrs=None):
    #tic = time.time()
    num_images_to_generate = 1
    image_attrs = rap_data___[image_name]["attrs"] # get the attributes of this image
    viewpoint = image_attrs[rap.attr_viewpoint] # get the viewpoint attribute of this image

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data___, rap.attr_OcclusionUp, 0))
    target_images = set(rap.get_images_with_attib(rap_data=rap_data___, attrib_index=rap.attr_viewpoint, attrib_value=viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)
    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = rap.get_images_with_attib(rap_data___, attr, other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)
    generated_replaced_ht = None
    _generated_replaced_background = None
    for idx in range(0, num_images_to_generate):
        #cv2.destroyAllWindows()
        img_name1 = image_name
        img_path1 = os.path.join(rap.rap_images_dir, img_name1)
        mask_path1 = os.path.join(rap.rap_masks_dir, img_name1)
        keypoints1 = rap_data___[img_name1]['keypoints']
        attr1 = rap_data___[img_name1]['attrs']
        img1 = cv2.imread(img_path1)
        mask1 = cv2.imread(mask_path1,cv2.IMREAD_GRAYSCALE)
        #mask1 = rap.load_crop_rap_mask(mask_path1)

        img_name2 = random.choice(target_images)
        img_path2 = os.path.join(rap.rap_images_dir, img_name2)
        mask_path2 = os.path.join(rap.rap_masks_dir, img_name2)
        keypoints2 = rap_data___[img_name2]['keypoints']
        attr2 = rap_data___[img_name2]['attrs']
        img2 = cv2.imread(img_path2)
        mask2 = cv2.imread(mask_path2,cv2.IMREAD_GRAYSCALE)
        #mask2 = rap.load_crop_rap_mask(mask_path2)
        if mask1 is None or mask2 is None:
            num_images_to_generate += 1
            continue
        else:
            assert mask1.shape[0] == img1.shape[0] and mask2.shape[1] == img2.shape[1]

        if constraint_functions is not None:
            for constraint_function in constraint_functions:
                if not constraint_function(img1, img2, mask1, mask2, keypoints1, keypoints2,
                                           attr1, attr2):
                    idx -= 1
                    continue
        #
        # generated_replaced_area = replace_head_area(img1, mask1, keypoints1,
        #                                             img2, mask2, keypoints2)
        # generated_replaced_rect = replace_head_rect(img1, mask1, keypoints1, attr1,
        #                                             img2, mask2, keypoints2, attr2)

        #generated_replaced_ht = replace_head_torso_area(img1, mask1, keypoints1, attr1, img2, mask2, keypoints2, attr2)

        _generated_replaced_background = replace_background(img1, mask1, img2, mask2)

        if generated_replaced_ht is None:
            num_images_to_generate += 1
            continue
        # ## display
        # img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # concat_images = [img1, img2_display]
        # if generated_replaced_area is not None:
        #     concat_images.append(generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     concat_images.append(generated_replaced_rect)
        # if generated_replaced_ht is not None:
        #     concat_images.append(generated_replaced_ht)
        # if generated_replaced_background is not None:
        #     concat_images.append(generated_replaced_background)
        #
        # display_img = cv2.hconcat(concat_images)
        # cv2.imshow('morphing', display_img)
        # cv2.waitKey()
    #toc = time.time()
    #print("elapsed = ", toc - tic)
    return _generated_replaced_background

def generate_images_from_this_image_v2 (image_name, rap_data___= None, constraint_functions= None, other_attrs=None):
    num_images_to_generate = 1
    image_attrs = rap_data___[image_name]["attrs"] # get the attributes of this image
    viewpoint = image_attrs[rap.attr_viewpoint] # get the viewpoint attribute of this image

    images_no_head_occlusions = set(rap.get_images_with_attib(rap_data___, rap.attr_OcclusionUp, 0))
    target_images = set(rap.get_images_with_attib(rap_data=rap_data___, attrib_index=rap.attr_viewpoint, attrib_value=viewpoint))
    target_images = target_images.intersection(images_no_head_occlusions)
    if other_attrs is not None:
        for attr in other_attrs:
            images_with_attr = rap.get_images_with_attib(rap_data___, attr, other_attrs[attr])
            if len(images_with_attr) > 0:
                target_images = target_images.intersection(set(images_with_attr))

    target_images = list(target_images)
    _generated_replaced_background = None
    for idx in range(0, num_images_to_generate):
        cv2.destroyAllWindows()
        img_name1 = image_name
        img_path1 = os.path.join(rap.rap_images_dir, img_name1)
        mask_path1 = os.path.join(rap.rap_masks_dir, img_name1)
        keypoints1 = rap_data___[img_name1]['keypoints']
        attr1 = rap_data___[img_name1]['attrs']
        img1 = cv2.imread(img_path1)
        mask1 = cv2.imread(mask_path1,cv2.IMREAD_GRAYSCALE)
        #mask1 = rap.load_crop_rap_mask(mask_path1)

        img_name2 = random.choice(target_images)
        img_path2 = os.path.join(rap.rap_images_dir, img_name2)
        mask_path2 = os.path.join(rap.rap_masks_dir, img_name2)
        keypoints2 = rap_data___[img_name2]['keypoints']
        attr2 = rap_data___[img_name2]['attrs']
        img2 = cv2.imread(img_path2)
        mask2 = cv2.imread(mask_path2,cv2.IMREAD_GRAYSCALE)
        #mask2 = rap.load_crop_rap_mask(mask_path2)
        if mask1 is None or mask2 is None:
            num_images_to_generate += 1
            continue
        else:
            assert mask1.shape[0] == img1.shape[0] and mask2.shape[1] == img2.shape[1]

        if constraint_functions is not None:
            for constraint_function in constraint_functions:
                if not constraint_function(img1, img2, mask1, mask2, keypoints1, keypoints2,
                                           attr1, attr2):
                    idx -= 1
                    continue

        # generated_replaced_area = replace_head_area(img1, mask1, keypoints1,
        #                                             img2, mask2, keypoints2)
        # generated_replaced_rect = replace_head_rect(img1, mask1, keypoints1, attr1,
        #                                             img2, mask2, keypoints2, attr2)
        #
        # generated_replaced_ht = replace_head_torso_area(img1, mask1, keypoints1, attr1,
        #                                                 img2, mask2, keypoints2, attr2)

        _generated_replaced_background = replace_background(img1, mask1, img2, mask2)

        # cv2.imshow('morphing1', generated_replaced_ht)
        # cv2.waitKey()
        #path = "/home/eshan/replication"
        #cv2.imwrite(os.path.join(path,"{}_{}.jpg".format(img_name1,img_name2)), _generated_replaced_background)

        # if generated_replaced_background is None:
        #     num_images_to_generate += 1
        #     continue
        # ## display
        # img2_display = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        # concat_images = [img1, img2_display]
        # if generated_replaced_area is not None:
        #     concat_images.append(generated_replaced_area)
        # if generated_replaced_rect is not None:
        #     concat_images.append(generated_replaced_rect)
        # if generated_replaced_ht is not None:
        #     concat_images.append(generated_replaced_ht)
        # if generated_replaced_background is not None:
        #     concat_images.append(generated_replaced_background)
        #
        # display_img = cv2.hconcat(concat_images)
        # cv2.imshow('morphing2', display_img)
        # cv2.waitKey()

    return _generated_replaced_background


if __name__ == '__main__':
    rap_dataset = rap.load_rap_dataset(rap_attributes_filepath = rap.rap_attribute_annotations, rap_keypoints_json = rap.rap_keypoints_json, load_from_file= True)
    additional_attrs = {rap.attr_viewpoint : 111}
    constraint_funcs = [is_compatible_area]
    num_images_to_generate = 100
    #generate_syntethic_images(rap_dataset, num_images_to_generate, other_attrs=additional_attrs, constraint_functions=constraint_funcs)

    # rap_img_name = os.listdir("/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_resized_masks256x256")
    # for i, each_img_name in enumerate(rap_img_name):

    ##################### todo: generate fake images from one image
    Person1 = "CAM16-2014-02-25-20140225151636-20140225152224-tarid149-frame2192-line1.png"
    Person2 = "CAM25-2014-04-22-20140422123345-20140422123909-tarid186-frame1483-line2.png"
    Person3 = "CAM30-2014-04-22-20140422120203-20140422120739-tarid73-frame3423-line2.png"
    Person4 = "CAM08-2014-03-26-20140326144149-20140326144741-tarid105-frame870-line1.png"
    Person5 = "CAM01-2013-12-23-20131223122515-20131223123103-tarid7-frame1602-line1.png"
    Person6 = "CAM22-2014-04-15-20140415112937-20140415113525-tarid63-frame1148-line1.png"
    Person7 = "CAM22-2014-04-17-20140417132119-20140417132715-tarid92-frame1706-line2.png"
    Person8 = "CAM22-2014-03-21-20140321153041-20140321153521-tarid381-frame1881-line1.png"
    for i in range(50):
        try:
            try:
                try:
                    img = generate_images_from_this_image_v2(Person6, rap_data___=rap_dataset, constraint_functions=None, other_attrs=additional_attrs)
                    path = "../RESULTS/PERSON6"
                    os.makedirs(path,exist_ok=True)
                    name = path + "/{}.jpg".format(i)
                    cv2.imwrite(filename=name, img=img)
                    #generate_images_from_this_image_v2(image_name=each_img_name, rap_data___=rap_dataset, other_attrs=additional_attrs)
                    print(i)
                except KeyError:
                    #print("keyerror: ", each_img_name)
                    pass
            except cv2.error:
                #print("cv2.error: ", each_img_name)
                pass
        except ValueError:
            #print("ValueError: ", each_img_name)
            pass




