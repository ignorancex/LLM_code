"""
Example training script 

Script makes use of the following GIT repositories: 
https://github.com/voxelmorph/voxelmorph
https://github.com/neel-dey/Atlas-GAN/tree/main
"""

import sys
import os
import tensorflow as tf
import pandas as pd
import json

from voxelmorph.tf.losses import Grad, NCC
# local imports 
from networks import AtlasGenerationModel, ProjectionDiscriminator
from data_generator import gen_data_generator, disc_data_generator
from AtlasGAN.discriminator_augmentations import disc_augment

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config_path):
    """
    ARGS:
        config_path: path to the config file
    """

    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
    grad_loss_fn = Grad('l2')
    ncc_loss_fn = NCC()

    config = load_config(config_path)

    print("Config loaded successfully:")
    print(json.dumps(config, indent=2))  

    input_cfg = config["input"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    out_cfg = config["out"]

    map_loss_fn = {
        'MSE': mse_loss_fn,
        'BCE': bce_loss_fn
    }

    try:
        disc_loss_fn = map_loss_fn[training_cfg['disc_loss']]
    except KeyError:
        raise ValueError("disc_loss has to be 'MSE' or 'BCE'")


    os.makedirs(out_cfg['output_model_path'], exist_ok=True)

    # LOAD DATA 
    train_metadata = pd.read_csv(os.path.join(input_cfg['data_path'],'train', 'metadata.csv'), delimiter=';')
    train_images_path = [os.path.join(input_cfg['data_path'], 'train', 'images', subject + input_cfg['img_suffix']) for subject in train_metadata['subject']]
    train_labels_path = [os.path.join(input_cfg['data_path'], 'train', 'labels', subject + input_cfg['seg_suffix']) for subject in train_metadata['subject']]
    train_conds = [int(cond) for cond in train_metadata['condition']]

    gen_train_data = gen_data_generator(
        img_list=train_images_path,
        seg_list=train_labels_path,
        attributes=train_conds,
        batch_size=training_cfg['batch_size'], 
        max_conds=input_cfg['cond_num']
    )

    disc_train_data = disc_data_generator(
        img_list=train_images_path,
        attributes=train_conds,
        batch_size=training_cfg['batch_size'],
        max_conds=input_cfg['cond_num']
    )

    # LOAD MODELS
    gen_model = AtlasGenerationModel(
        input_shape=input_cfg['img_res'],
        n_condns=input_cfg['cond_dim'],
        nb_unet_features=[model_cfg['vxm_unet_arch_enc'], model_cfg['vxm_unet_arch_dec']], 
        bool_bidir=training_cfg['bidir']
    )  

    disc_model = ProjectionDiscriminator(
        input_shape=(128, 128, 96), 
        n_condns=input_cfg['cond_dim']
    )

    # SET UP OPTIMIZERS
    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_cfg['lr_gen'],
        beta_1=training_cfg['beta_1'],
        beta_2=training_cfg['beta_2'],
        epsilon=training_cfg['epsilon'],
    )
    gen_optimizer.iterations

    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_cfg['lr_disc'],
        beta_1=training_cfg['beta_1'],
        beta_2=training_cfg['beta_2'],
        epsilon=training_cfg['epsilon'],
    )
    disc_optimizer.iterations

    @tf.function
    def train_disc_step(target_image, real_image, target_condn, real_cond):
        """
        Discriminator training step

        ARGS INPUT: 
            target_image: real samples used for generating templates
            real_image: real samples  
            target_condn: conditions used for generting templates
            real_condn: conditions of the real samples 

        ARGS OUTPUT: 
            total_disc_loss: total discriminator loss
        """
        with tf.GradientTape() as disc_tape:

            moved_atlas, _, _, _, _ = gen_model((target_image, target_condn), training=False) 

            # DATA AUGMENTATION
            aug_choice_real = tf.random.uniform((1,), 0, 4, dtype=tf.int32)
            aug_choice_fake = tf.random.uniform((1,), 0, 4, dtype=tf.int32)

            moved_atlases = disc_augment(moved_atlas, aug_choice_fake, intensity_mods=False)
            real_image = disc_augment(real_image, aug_choice_real, intensity_mods=False)

            disc_logits_real_local = disc_model((real_image, real_cond), training=True)
            disc_logits_fake_local = disc_model((moved_atlases, target_condn), training=True)

            disc_real_loss = disc_loss_fn(tf.ones_like(disc_logits_real_local), disc_logits_real_local)
            disc_fake_loss = disc_loss_fn(tf.zeros_like(disc_logits_fake_local), disc_logits_fake_local)
            
            total_disc_loss = 0.5*(disc_fake_loss + disc_real_loss)

        disc_gradients = disc_tape.gradient(total_disc_loss, disc_model.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_gradients, disc_model.trainable_variables))

        return total_disc_loss
    
    @tf.function
    def train_gen_step(real_image, real_cond, real_labels):
        """
        Generator training step: 

        ARGS INPUT:
            real_image: real samples
            real_cond: conditions of the real samples
            real_labels: segmentation labels (one-hot encoded)

        ARGS OUPUT: 
             gen_loss_total: total generator loss
             sim_loss_fwd: image similarity (NCC) loss between (forward) warped atlas and real image
             sim_loss_bwd: image similarity (NCC) loss between (backward) warped real image and atlas 
             seg_loss_total: MSE loss of segmentation labels
             smoothness_loss: smoothness of the deformation field / displacement
             magnitude_loss: magnitude of the deformation field / displacement 
        
        """
        with tf.GradientTape() as gen_tape:

            moved_atlas, moved_labels, moved_subject, atlas, disp = gen_model((real_image, real_cond), training=True) 

            # IMAGE SIMILARITY LOSS FORWARD
            similarity_loss = 1 + (tf.reduce_mean(ncc_loss_fn.loss(moved_atlas, real_image)))

            # IMAGE SIMILARITY LOSS BACKWARD
            if training_cfg['bidir']:
                similarity_loss_2 = 1 + (tf.reduce_mean(ncc_loss_fn.loss(moved_subject, atlas)))
            else:
                similarity_loss_2 = tf.constant(0.0)

            # MAGNITUDE DEFORMATION FIELD
            magnitude_loss = tf.reduce_mean(mse_loss_fn(tf.zeros_like(disp), disp))

            # SMOOTHNESS DEFORMATION FIELD
            smoothness_loss = tf.reduce_mean(grad_loss_fn.loss(tf.zeros_like(disp), disp))

            # SEGMENTATION LOSS
            mse_loss = []
            for channel in range(real_labels.shape[-1]):
                loss = mse_loss_fn(moved_labels[0, ..., channel], real_labels[0, ..., channel])
                mse_loss.append(tf.reduce_mean(loss))
            segmentation_loss_total = tf.reduce_mean(mse_loss)
            
            # ADVERSARIAL LOSS
            disc_logits_fake = disc_model((moved_atlas, real_cond), training=True)
            adv_loss = disc_loss_fn(tf.ones_like(disc_logits_fake), disc_logits_fake)

            # TOTAL GENERATOR LOSS 
            total_gen_loss = (
                (training_cfg['weight_loss_reg_def_smoothness'] * smoothness_loss) +
                (training_cfg['weight_loss_reg_def_magnitude'] * magnitude_loss) +
                (training_cfg['weight_loss_img'] * (training_cfg['weight_bidir_forward'] * similarity_loss + training_cfg['weight_bidir_backward'] * similarity_loss_2)) +
                (training_cfg['weight_loss_seg'] * segmentation_loss_total) +
                (training_cfg['weight_loss_adverserial'] * adv_loss)
                )

        gen_gradients = gen_tape.gradient(total_gen_loss, gen_model.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))

        return total_gen_loss, similarity_loss, similarity_loss_2, segmentation_loss_total, smoothness_loss, magnitude_loss

    # TRAINING LOOP
    steps_per_epoch = int(len(train_images_path) / training_cfg['batch_size'])
    best_gen_loss = float('inf')

    for epoch in range(training_cfg['num_epoch']):
        print(f"Epoch {epoch+1}/{training_cfg['num_epoch']}")        
        
        epoch_gen_loss = 0
        epoch_disc_loss = 0

        disc_iter = iter(disc_train_data)
        gen_iter = iter(gen_train_data)

        for n in range(steps_per_epoch):

            # DISCRIMINATOR TRAINING LOOP
            for _ in range(training_cfg['train_step_disc']):

                target_image, real_image, target_condn, real_cond = next(disc_iter)

                disc_loss = train_disc_step(target_image, real_image, target_condn, real_cond)
                epoch_disc_loss += float(disc_loss)

            epoch_disc_loss /= training_cfg['train_step_disc']

            # GENERATOR TRAINING LOOP
            for _ in range(training_cfg['train_step_gen']):

                real_image, real_cond, real_labels = next(gen_iter)

                (gen_loss, sim_fwd, sim_bwd, seg_loss,
                smooth_loss, mag_loss) = train_gen_step(real_image, real_cond, real_labels)

                epoch_gen_loss += float(gen_loss)

            epoch_gen_loss /= training_cfg['train_step_gen']

            # print(
            #     f"{n}/{steps_per_epoch}  "
            #     f"GEN: {float(gen_loss):.4f}, DISC: {float(disc_loss):.4f}  "
            #     f"SIM(F): {float(sim_fwd):.4f}, SIM(B): {float(sim_bwd):.4f}  "
            #     f"SEG: {float(seg_loss):.4f}  "
            #     f"MAG: {float(mag_loss):.4f}, SMOOTH: {float(smooth_loss):.4f} "
            # )

        mean_epoch_gen_loss = epoch_gen_loss / steps_per_epoch
        mean_epoch_disc_loss = epoch_disc_loss / steps_per_epoch

        print(
            f"GENERATOR LOSS: {mean_epoch_gen_loss:.4f} \n"
            f"DISCRIMINATOR LOSS: {mean_epoch_disc_loss:.4f} \n"
        )

        gen_model.save_weights(os.path.join(out_cfg['output_model_path'], 'latest_model_weights.h5'))
        
        if mean_epoch_gen_loss < best_gen_loss:
            best_gen_loss = mean_epoch_gen_loss
            gen_model.save_weights(os.path.join(out_cfg['output_model_path'], out_cfg['name_best_model_weights'] + '.h5'))
            print(f"New best model saved \n")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python train_CAL.py /path/to/config.json")
        sys.exit(1)

    main(config_path=sys.argv[1])