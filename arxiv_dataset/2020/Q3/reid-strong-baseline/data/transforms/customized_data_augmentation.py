from torchvision.transforms.functional import to_tensor
import random
from RAP_script.generate_synthetic_images import generate_images_from_this_image
from PIL import Image
import os
import cv2

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class part_substitution(object):

    def __init__(self, probability, rap_data__,constraint_funcs=None, other_attrs=None):
        self.probability = probability
        self.rap_data__ = rap_data__
        self.constraint_funcs=constraint_funcs
        self.other_attrs=other_attrs

    def __call__(self, current_image_path):
        img = read_image(current_image_path)
        if random.uniform(0, 1) >= self.probability:
            return img

        try:
            try:
                try:
                    img_name = current_image_path.split("/")[-1]
                    img = generate_images_from_this_image(image_name=img_name,
                                                          rap_data___=self.rap_data__,
                                                          constraint_functions=self.constraint_funcs,
                                                          other_attrs=self.other_attrs)
                except KeyError:
                    #print("keyerror: ", img_name)
                    return None
            except cv2.error:
                #print("cv2.error: ", img_name)
                return None
        except ValueError:
            #print("ValueError: ", img_name)
            return None
        return img
