import numpy as np
import voxelmorph as vxm
import tensorflow as tf

from utils import probHeterogeneous

def gen_data_generator(
    img_list,
    seg_list,
    attributes,
    batch_size,
    max_conds
):

    prob = probHeterogeneous(attributes)

    while True:
        registration_idx = np.random.choice(len(img_list), batch_size, p=prob)
 
        registration_images = [vxm.py.utils.load_volfile(img_list[i], add_batch_axis=True, add_feat_axis=True) for i in registration_idx]
        registration_images = np.concatenate(registration_images, axis=0)   
        segmentation_images = [vxm.py.utils.load_volfile(seg_list[i], add_batch_axis=True, add_feat_axis=False) for i in registration_idx]
        segmentation_images = np.concatenate(segmentation_images, axis=0)
        registration_images_condns = np.stack([attributes[i] for i in registration_idx], axis=0)
        registration_images_condns = registration_images_condns / max_conds

        yield (
        tf.convert_to_tensor(registration_images, dtype=tf.float32),
        tf.convert_to_tensor(registration_images_condns, dtype=tf.float32),
        tf.convert_to_tensor(segmentation_images, dtype=tf.float32)
        )

def disc_data_generator(
    img_list,
    attributes,
    batch_size,
    max_conds
):
    
    prob = probHeterogeneous(attributes)

    while True:

        registration_idx = np.random.choice(len(img_list), batch_size, p=prob)
        adversarial_idx = np.random.choice(len(img_list), batch_size, p=prob)
        
        registration_images = [vxm.py.utils.load_volfile(img_list[i], add_batch_axis=True, add_feat_axis=True) for i in registration_idx]
        registration_images = np.concatenate(registration_images, axis=0)   
        registration_images_condns = np.stack([attributes[i] for i in registration_idx], axis=0)
        registration_images_condns = registration_images_condns / max_conds

        adversarial_images = [vxm.py.utils.load_volfile(img_list[i], add_batch_axis=True, add_feat_axis=True) for i in  adversarial_idx]
        adversarial_images = np.concatenate(adversarial_images, axis=0)
        adversarial_images_condns = np.stack([attributes[i] for i in adversarial_idx], axis=0)
        adversarial_images_condns = adversarial_images_condns / max_conds

        yield (
            tf.convert_to_tensor(registration_images, dtype=tf.float32),
            tf.convert_to_tensor(adversarial_images, dtype=tf.float32),
            tf.convert_to_tensor(registration_images_condns, dtype=tf.float32), 
            tf.convert_to_tensor(adversarial_images_condns, dtype=tf.float32)
        )