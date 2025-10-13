"""
Example inference script without any metrics. 
"""

import os
import sys
# reduce tf output 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
import voxelmorph as vxm
import json 
import time
import pandas as pd

#local import
from networks import AtlasGenerationModel
from utils import save_nifti_image

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    device, _ = vxm.tf.utils.setup_device('0')

    if len(sys.argv) != 2:
        print("Usage: python inference_CAL.py /path/to/config.json")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    

    print("Config loaded successfully:")
    print(json.dumps(config, indent=2))  

    input_cfg = config["input"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    out_cfg = config["out"]
    
    os.makedirs(out_cfg['output_inference_path'], exist_ok=True)
    model_path = os.path.join(out_cfg['output_model_path'], 'best_model_weights.h5')

    # load data
    test_metadata = pd.read_csv(os.path.join(input_cfg['data_path'],'test', 'metadata.csv'), delimiter=';')
    test_images_path = [os.path.join(input_cfg['data_path'], 'test', 'images', subject + input_cfg['img_suffix']) for subject in test_metadata['subject']]
    test_labels_path = [os.path.join(input_cfg['data_path'], 'test', 'labels', subject + input_cfg['seg_suffix']) for subject in test_metadata['subject']]
    test_conds = [int(cond) for cond in test_metadata['condition']]

    #load model
    model = AtlasGenerationModel(
        input_shape=input_cfg['img_res'],
        n_condns=input_cfg['cond_dim'],
        nb_unet_features=[model_cfg['vxm_unet_arch_enc'], model_cfg['vxm_unet_arch_dec']], 
        bool_bidir=training_cfg['bidir']
    )  
    model.load_weights(model_path)

    full_model = Model(inputs=model.input, outputs=model.output)

    seg_layer = model.get_layer(name='seg_gen')
    seg_model = Model(inputs=model.input, outputs=seg_layer.output)

    with tf.device(device):

        for img_path, label_path, cond in zip(test_images_path, test_labels_path, test_conds):
            
            cond_pred = np.array([cond / input_cfg['cond_num']]) 

            # fixed images 
            start = time.clock()
            fixed_img = vxm.py.utils.load_volfile(img_path, add_batch_axis=True, add_feat_axis=True)
            fixed_label = vxm.py.utils.load_volfile(label_path, add_batch_axis=True)
            end = time.clock()

            print(f"PROCESSING TIME: {end - start}s")

            # predictions (moving images)
            [moved_template_forward, moved_label, moved_template_backward, template_img, disp] = full_model([fixed_img, cond_pred])          
            template_seg = seg_model.predict([fixed_img, cond_pred])

            # save output
            save_nifti_image(image=fixed_img, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_fixed_img.nii.gz'))
            save_nifti_image(image=fixed_label, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_fixed_label.nii.gz'))
            save_nifti_image(image=template_img, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_template_img.nii.gz'))
            save_nifti_image(image=template_seg, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_template_seg.nii.gz'))
            save_nifti_image(image=moved_template_forward, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_moved_img_forw.nii.gz'))
            save_nifti_image(image=moved_template_backward, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_moved_img_backw.nii.gz'))
            save_nifti_image(image=moved_label, target_shape=input_cfg['img_res'], output_path=os.path.join(out_cfg['output_inference_path'], f'{str(cond)}_moved_label.nii.gz'))
            

            # Continue here with further evaluation, laplacian filter, ... 


    print('DONE')

if __name__ == "__main__":
    main()