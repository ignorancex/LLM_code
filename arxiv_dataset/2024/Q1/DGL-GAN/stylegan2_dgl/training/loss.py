# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014

def G_logistic(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -tf.nn.softplus(fake_scores_out) # log(1-sigmoid(fake_scores_out)) # pylint: disable=invalid-unary-operand-type
    return loss, None

def G_logistic_ns(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('Loss/G/fake_scores', fake_scores_out)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    loss = autosummary('Loss/G_bar/loss', loss)
    return loss, None

def G_logistic_ns_smooth(G, D, D_bar, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('Loss/G/fake_scores', fake_scores_out)
    p_fake = tf.math.sigmoid(fake_scores_out_bar)
    p_fake = autosummary('Loss/G/p_fake', p_fake)
    loss = (1 - p_fake) * tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    loss = autosummary('Loss/G/loss', loss)
    return loss, None

def G_logistic_ns_distill_smooth(G, D, D_bar, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('Loss/G_bar/fake_scores', fake_scores_out)
    p_fake = tf.math.sigmoid(fake_scores_out_bar)
    loss = p_fake * tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    loss = autosummary('Loss/G_bar/loss', loss)
    return loss, None

#----------------------------------------------------------------------------
# min M(G, D) + \alpha * M(G, D_bar).
# Duality gap
# D_bar is a pretrained discriminator.

def G_dg_logistic_ns(G, D, D_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.1):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar)
    dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))

    return loss, None

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/D_bar/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/D_bar/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = autosummary('Loss/D_bar/loss', loss)
    return loss, None

# ---------------------------------------------------------------------------
# max M(G, D) + \alpja * M(G_bar, D)
# G_bar is a pretrained generator.
def D_logistic_r1_dg(G, G_bar, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, alpha=0.1, name='D'):
    _ = opt, training_set
    print(f'loss function, reals is {reals}, D is {D}, G_bar is {G_bar}')
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out_bar = G_bar.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out_bar = D.get_output_for(fake_images_out_bar, labels, is_training=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    fake_scores_out_bar = autosummary(f'Loss/{name}/scores/D_Gbar', fake_scores_out_bar)
    dg_term = tf.nn.softplus(fake_scores_out_bar)
    dg_term = autosummary('Loss/D_loss/DG_term', dg_term)
    ns_loss = tf.nn.softplus(fake_scores_out) + tf.nn.softplus(-real_scores_out)
    ns_loss = autosummary('Loss/D_loss/ns_term', ns_loss)
    loss = ns_loss + alpha * dg_term # -log(1-sigmoid(fake_scores_out))
    #loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        print(f'real_grads is {real_grads}, real_scores_out is {real_scores_out}')
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary(f'Loss/{name}/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

# ---------------------------------------------------------------------------
# Monitor D(x), D(G(z)), Dbar(x), and Dbar(G(z)).
def D_logistic_r1_monitor(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0, alpha=0.1, name='D'):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out_bar = D_bar.get_output_for(reals, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    real_scores_out_bar = autosummary(f'Loss/{name}/scores/real_bar', real_scores_out_bar)
    fake_scores_out_bar = autosummary(f'Loss/{name}/scores/fake_bar', fake_scores_out_bar)
    ns_loss = tf.nn.softplus(fake_scores_out) + tf.nn.softplus(-real_scores_out)
    ns_loss = autosummary('Loss/D_loss/ns_term' ,ns_loss)
    loss = ns_loss  # -log(1-sigmoid(fake_scores_out))
    #loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        print(f'real_grads is {real_grads}, real_scores_out is {real_scores_out}')
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return real_scores_out_bar, fake_scores_out_bar, loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# Smooth labels
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_smooth(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    fake_images_out  = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out_bar = D_bar.get_output_for(reals, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    p_fake, p_real = tf.math.sigmoid(fake_scores_out_bar), tf.math.sigmoid(real_scores_out_bar)
    p_fake, p_real = autosummary('Loss/D/p_fake', p_fake), autosummary('Loss/D/p_real', p_real)
    loss = (1-p_fake) * tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += p_real * tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss =autosummary('Loss/D/D_loss', loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# Smooth labels
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_smooth_internal(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0, name='D'):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    #fake_images_out  = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out_bar = D_bar.get_output_for(reals, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    p_fake, p_real = tf.math.sigmoid(fake_scores_out_bar), tf.math.sigmoid(real_scores_out_bar)
    p_fake, p_real = autosummary('Loss/D/p_fake', p_fake), autosummary('Loss/D/p_real', p_real)
    loss = (1-p_fake) * tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += p_real * tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss =autosummary('Loss/D/D_loss', loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, name=''):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = autosummary(f'Loss/{name}/D_loss', loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary(f'Loss/{name}/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

def D_logistic_r2(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type

    with tf.name_scope('GradientPenalty'):
        fake_grads = tf.gradients(tf.reduce_sum(fake_scores_out), [fake_images_out])[0]
        gradient_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_internal(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0, name=''):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    #fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = autosummary(f'Loss/{name}/D_loss', loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# L1(D_student, D_teacher)
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_align(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0, lambbda=0.1, name=''):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_0, real_1, real_2 = D.get_output_for(reals, labels, is_training=True, return_internal_features=True)
    fake_scores_out, fake_0, fake_1, fake_2 = D.get_output_for(fake_images_out, labels, is_training=True, return_internal_features=True)
    _, real_0_bar, real_1_bar, real_2_bar = D_bar.get_output_for(reals, labels, is_training=False, return_internal_features=True)
    _, fake_0_bar, fake_1_bar, fake_2_bar = D_bar.get_output_for(fake_images_out, labels, is_training=False, return_internal_features=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    adv_loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    adv_loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    l1_loss_real = l1_loss([real_0, real_1, real_2], [real_0_bar, real_1_bar, real_2_bar])
    l1_loss_fake = l1_loss([fake_0, fake_1, fake_2], [fake_0_bar, fake_1_bar, fake_2_bar])
    adv_loss = autosummary(f'Loss/{name}/D_Adv_loss', adv_loss)
    l1_loss_real = autosummary(f'Loss/{name}/L1_real', l1_loss_real)
    l1_loss_fake = autosummary(f'Loss/{name}/L1_fake', l1_loss_fake)
    loss = adv_loss + lambbda * (l1_loss_real + l1_loss_fake)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary(f'Loss/{name}/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# cross_entropy(logits(D_student), logits(D_teacher))
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_align_logits(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0, lambbda=0.1, name=''):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_0, real_1, real_2 = D.get_output_for(reals, labels, is_training=True, return_internal_features=True)
    fake_scores_out, fake_0, fake_1, fake_2 = D.get_output_for(fake_images_out, labels, is_training=True, return_internal_features=True)
    real_scores_out_bar, real_0_bar, real_1_bar, real_2_bar = D_bar.get_output_for(reals, labels, is_training=False, return_internal_features=True)
    fake_scores_out_bar, fake_0_bar, fake_1_bar, fake_2_bar = D_bar.get_output_for(fake_images_out, labels, is_training=False, return_internal_features=True)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    adv_loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    adv_loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    KL_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.math.sigmoid(real_scores_out_bar), logits=real_scores_out)
    KL_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.math.sigmoid(fake_scores_out_bar), logits=real_scores_out)
    adv_loss = autosummary(f'Loss/{name}/D_Adv_loss', adv_loss)
    KL_real = autosummary(f'Loss/{name}/KL_real', KL_real)
    KL_fake = autosummary(f'Loss/{name}/KL_fake', KL_fake)
    
    loss = adv_loss + lambbda * (KL_real + KL_fake)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary(f'Loss/{name}/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018
# Adv(G, D) + \lambda * KL(D_bar([real, fake]), D([real, fake]))

def D_logistic_r1_kl(G, D, D_bar, opt, training_set, minibatch_size, reals, labels, gamma=10.0, lamda=1.0, name=''):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    #fake_images_out, _, _, _ = G.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out_bar = D_bar.get_output_for(reals, labels, is_training=False)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=False)
    real_labels = tf.concat([tf.math.sigmoid(real_scores_out_bar), 1-tf.math.sigmoid(real_scores_out_bar)], axis=1)
    real_distri = tf.concat([tf.math.sigmoid(real_scores_out), 1-tf.math.sigmoid(real_scores_out)], axis=1)
    fake_labels = tf.concat([tf.math.sigmoid(fake_scores_out_bar), 1-tf.math.sigmoid(fake_scores_out_bar)], axis=1)
    fake_distri = tf.concat([tf.math.sigmoid(fake_scores_out), 1-tf.math.sigmoid(fake_scores_out)], axis=1)
    kl_real = tf.nn.softmax_cross_entropy_with_logits(labels=real_labels, logits=real_distri)
    kl_fake = tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_distri)
    real_scores_out = autosummary(f'Loss/{name}/scores/real', real_scores_out)
    fake_scores_out = autosummary(f'Loss/{name}/scores/fake', fake_scores_out)
    kl_real = autosummary(f'Loss/{name}/kl_real', kl_real)
    kl_fake = autosummary(f'Loss/{name}/kl_fake', kl_fake)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = autosummary(f'Loss/{name}/Adv_loss', loss)
    loss += (lamda * kl_real + lamda * kl_fake)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary(f'Loss/{name}/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# Paired real samples and fake samples. "Towards a Better Global Loss Landscape of GANs", Sun et al. 2020
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018

def D_logistic_r1_pair(G, D, opt, training_set, minibatch_size, reals, labels, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/r1_bar/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/r1_bar/scores/fake', fake_scores_out)
    gap = real_scores_out - fake_scores_out
    #loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    #loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
    loss = tf.nn.softplus(gap)
    loss = autosummary('Loss/r1_bar/D_loss', loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017

def G_wgan(G, D, opt, training_set, minibatch_size):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    loss = -fake_scores_out
    return loss, None

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, wgan_epsilon=0.001):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon
    return loss, None

#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, wgan_lambda=10.0, wgan_epsilon=0.001, wgan_target=1.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = D.get_output_for(mixed_images_out, labels, is_training=True)
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_scores_out), [mixed_images_out])[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        reg = gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss, reg

#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

def G_logistic_ns_pathreg(G, D, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = autosummary('G/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    loss= autosummary('G/loss', loss)

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
# min_{G} lambbda*|L_{Do}-L_{Dc}|/|L_{Do}| + |L_{Go} - L_{Gc}|/|L_{Go}|
# "Self-Supervised GAN Compression". https://arxiv.org/pdf/2007.01491.pdf

def self_supervised_pathreg(G, G_bar, D_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, lambbda=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out_bar = G_bar.get_output_for(latents, labels, is_training=False)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    # |L_{Do} - L_{Dc}| / |L_{Do}|
    fake_scores_out = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out_bar, labels, is_training=True)
    fake_scores_out = autosummary('Loss/scores/Dbar_G', fake_scores_out)
    fake_scores_out_bar = autosummary('Loss/scores/Dbar_Gbar', fake_scores_out_bar)
    L_Dc = tf.nn.softplus(fake_scores_out)
    L_Do = tf.nn.softplus(fake_scores_out_bar)
    L_Do = tf.Print(L_Do, [tf.shape(L_Do)], message='shape of L_Do is ')
    L_D = (L_Do-L_Dc)**2 / L_Do**2
    L_D = tf.Print(L_D, [tf.shape(L_D)], message='shape of L_D is ')
    L_D = autosummary('Loss/D_loss/L_D', L_D)

    # |L_{Go} - L_{Gc}| / |L_{Go}|
    L_Gc = tf.nn.softplus(-fake_scores_out)
    L_Go = tf.nn.softplus(-fake_scores_out_bar)
    L_G = (L_Go-L_Gc)**2 / L_Go**2
    L_G = autosummary('Loss/G_loss/L_G', L_G)

    # Total loss
    L = L_G + lambbda * L_D
    
    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return L, reg


#----------------------------------------------------------------------------
# min M(G, D) + \alpha * M(G, D_bar).
# Duality gap
# D_bar is a pretrained discriminator.

def G_dg(G, D, D_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar)
    dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))

    return loss, None

#----------------------------------------------------------------------------
# min M(G, D) + \alpha * M(G, D_bar).
# Duality gap
# D_bar is a pretrained discriminator.

def G_dg_pathreg(G, D, D_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar)
    dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
# min_{G} M(G, D) + \alpha * (M(G, D_bar1) + M(G, D_bar2) + .. + M(G, D_barn)).
# Or min_{G} M(G, D) + \alpha * (1 / (-D_bar1(G(z))) * M(G, D_bar1) + 1 / (-D_bar2(G(z))) * M(G, D_bar2) + .. + 1 / (-D_barn(G(z))) * M(G, D_barn)).
# Ensemble of Dbar, with different channels.
# D_bars is a set of pretrained discriminator, with different channels, same architecture.

def G_pathreg_ens(G, D, D_bars, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    loss = g_loss
    for i in range(len(D_bars)):
        D_bar = D_bars[i]
        fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
        fake_scores_out_bar = autosummary(f'Loss/scores/fakes_scores_Dbar{i}', fake_scores_out_bar)
        #dg_term = tf.nn.softplus(-fake_scores_out_bar)
        coeff = tf.stop_gradient((-fake_scores_out_bar))
        coeff = autosummary(f'Loss/G_loss/DG_term_{i}_coeff', coeff)
        ignore = tf.stop_gradient(tf.cast((fake_scores_out_bar < 0.), tf.float32))
        dg_term = coeff * tf.nn.softplus(-fake_scores_out_bar) * ignore
        dg_term = autosummary(f'Loss/G_loss/DG_term_{i}', dg_term)
        loss = loss + alpha * dg_term

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)
            fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
# Get gradients of D(G(z)) and Dbar(G(z)).
# Get fake images during training.

def get_gradients(G, D, D_bar, D_update_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    #fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out_update_bar = D_update_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar)
    g_loss = tf.nn.softplus(-fake_scores_out)
    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))
    dD_dFake = tf.gradients(fake_scores_out, [fake_images_out])[0]
    dDbar_dFake = tf.gradients(fake_scores_out_bar, [fake_images_out])[0]
    dUpdateDbar_dFake = tf.gradients(fake_scores_out_update_bar, [fake_images_out])[0]
    dAdv_dFake = tf.gradients(g_loss, [fake_images_out])[0]
    dDGL_dFake = tf.gradients(dg_term, [fake_images_out])[0]

    return fake_images_out, dD_dFake, dDbar_dFake, dUpdateDbar_dFake, dAdv_dFake, dDGL_dFake


#----------------------------------------------------------------------------
# Get gradients of D(G(z)) and Dbar(G(z)), smooth gradient.
# Get fake images during training.

def get_gradients_smooth(G, D, D_bar, D_update_bar, latent_read, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5, sigma=0.1, N=50.0, std=1.0):
    _ = opt
    labels = training_set.get_random_labels_tf(minibatch_size)
    latent_read = tf.Print(latent_read, [latent_read], 'latent_read is ')
    labels = tf.Print(labels, [labels], 'labels is ')
    fake_images_out, fake_dlatents_out = G.get_output_for(latent_read, labels, is_training=True, return_dlatents=True)
    fake_dlatents_out = tf.Print(fake_dlatents_out, [fake_dlatents_out], 'fake_dlatents_out is ')
    noise = tf.random_normal([minibatch_size] + G.output_shapes[0][1:], stddev=1.0)
    fake_images_out = tf.Print(fake_images_out, [fake_images_out], 'fake_images_out is ')
    fake_images_out_n = fake_images_out + noise
    fake_scores_out_bar_n = D_bar.get_output_for(fake_images_out_n, labels, is_training=True)
    fake_scores_out_update_bar_n = D_update_bar.get_output_for(fake_images_out_n, labels, is_training=True)
    fake_scores_out_n = D.get_output_for(fake_images_out_n, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar_n)
    g_loss = tf.nn.softplus(-fake_scores_out_n)
    dD_dFake_n = tf.gradients(fake_scores_out_n, [fake_images_out_n])[0]
    dDbar_dFake_n = tf.gradients(fake_scores_out_bar_n, [fake_images_out_n])[0]
    dUpdateDbar_dFake_n = tf.gradients(fake_scores_out_update_bar_n, [fake_images_out_n])[0]
    dAdv_dFake_n = tf.gradients(g_loss, [fake_images_out_n])[0]
    dDGL_dFake_n = tf.gradients(dg_term, [fake_images_out_n])[0]

    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))
    dD_dFake = dD_dFake_n
    dDbar_dFake = dDbar_dFake_n
    dUpdateDbar_dFake = dUpdateDbar_dFake_n
    dAdv_dFake = dAdv_dFake_n
    dDGL_dFake = dDGL_dFake_n
    return fake_images_out, fake_dlatents_out, dD_dFake, dDbar_dFake, dUpdateDbar_dFake, dAdv_dFake, dDGL_dFake

#----------------------------------------------------------------------------
# min M(G, D) + \alpha * M(G, D_bar).
# Duality gap
# D_bar is a pretrained discriminator.

def G_dg_pathreg_internal(G, D, D_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    #fake_images_out, fake_dlatents_out = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    dg_term = tf.nn.softplus(-fake_scores_out_bar)
    dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    loss = g_loss + alpha * dg_term   # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)
            #fake_images_out, fake_dlatents_out = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
# min M(G, D) + \gama * L1(student, teacher).
# knowledge distillation

def G_pathreg_l1(G, D, G_bar, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.5, gama=1.0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out, features_0, features_1, features_2 = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_images_out_bar, features_bar_0, features_bar_1, features_bar_2 = G_bar.get_output_for(latents, labels, is_training=True, return_dlatents=False)
    #fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    #dg_term = tf.nn.softplus(-fake_scores_out_bar)
    #dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    l1 = l1_loss([features_0, features_1, features_2, fake_images_out], [features_bar_0, features_bar_1, features_bar_2, fake_images_out_bar])
    l1 = autosummary('loss/G_loss/L1_loss', l1)
    loss = g_loss + gama * l1  # -log(sigmoid(fake_scores_out))

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg

#----------------------------------------------------------------------------
# min M(G, D) + \alpha * M(G, D_bar) + \gama * L1(student, teacher).
# Duality gap and knowledge distillation
# D_bar is a pretrained discriminator.

def G_pathreg_l1_Dbar(G, D, G_bar, D_bars, opt, training_set, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0, alpha=0.1, gama=0.1):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out, fake_dlatents_out, features_0, features_1, features_2 = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    fake_images_out_bar, features_bar_0, features_bar_1, features_bar_2 = G_bar.get_output_for(latents, labels, is_training=True, return_dlatents=False)
    #fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, is_training=True)
    #dg_term = tf.nn.softplus(-fake_scores_out_bar)
    #dg_term = autosummary('Loss/G_loss/DG_term', dg_term)
    g_loss = tf.nn.softplus(-fake_scores_out)
    g_loss = autosummary('Loss/G_loss/ns', g_loss)
    l1 = l1_loss([features_0, features_1, features_2, fake_images_out], [features_bar_0, features_bar_1, features_bar_2, fake_images_out_bar])
    l1 = autosummary('loss/G_loss/L1_loss', l1)
    loss = g_loss + gama * l1  # -log(sigmoid(fake_scores_out))
    for i in range(len(D_bars)):
        D_bar = D_bars[i]
        fake_scores_out_bar = D_bar.get_output_for(fake_images_out, labels, is_training=True)
        fake_scores_out_bar = autosummary(f'Loss/scores/fakes_scores_Dbar{i}', fake_scores_out_bar)
        #dg_term = tf.nn.softplus(-fake_scores_out_bar)
        coeff = tf.stop_gradient((-fake_scores_out_bar))
        coeff = autosummary(f'Loss/G_loss/DG_term_{i}_coeff', coeff)
        ignore = tf.stop_gradient(tf.cast((fake_scores_out_bar < 0.), tf.float32))
        dg_term = coeff * tf.nn.softplus(-fake_scores_out_bar) * ignore
        dg_term = autosummary(f'Loss/G_loss/DG_term_{i}', dg_term)
        loss = loss + alpha * dg_term

    # Path length regularization.
    with tf.name_scope('PathReg'):

        # Evaluate the regularization term using a smaller minibatch to conserve memory.
        if pl_minibatch_shrink > 1:
            pl_minibatch = minibatch_size // pl_minibatch_shrink
            pl_latents = tf.random_normal([pl_minibatch] + G.input_shapes[0][1:])
            pl_labels = training_set.get_random_labels_tf(pl_minibatch)
            fake_images_out, fake_dlatents_out, _, _, _ = G.get_output_for(pl_latents, pl_labels, is_training=True, return_dlatents=True)

        # Compute |J*y|.
        pl_noise = tf.random_normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(G.output_shape[2:]))
        pl_grads = tf.gradients(tf.reduce_sum(fake_images_out * pl_noise), [fake_dlatents_out])[0]
        pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
        pl_lengths = autosummary('Loss/pl_lengths', pl_lengths)

        # Track exponential moving average of |J*y|.
        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
        pl_update = tf.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            pl_penalty = tf.square(pl_lengths - pl_mean)
            pl_penalty = autosummary('Loss/pl_penalty', pl_penalty)

        # Apply weight.
        #
        # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
        # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
        #
        # gamma_pl = pl_weight / num_pixels / num_affine_layers
        # = 2 / (r^2) / (log2(r) * 2 - 2)
        # = 1 / (r^2 * (log2(r) - 1))
        # = ln(2) / (r^2 * (ln(r) - ln(2))
        #
        reg = pl_penalty * pl_weight

    return loss, reg


#----------------------------------------------------------------------------

# The L1 distance between teacher model and the student model

def l1_loss(G_features, G_bar_features):

    loss = tf.reduce_mean(tf.abs(G_features[0] - G_bar_features[0]))
    for s, t in zip(G_features[1:], G_bar_features[1:]):
        loss += tf.reduce_mean(tf.abs(s - t))

    return loss
