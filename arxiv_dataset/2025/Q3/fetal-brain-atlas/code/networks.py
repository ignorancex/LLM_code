"""
Atlas generation model (incl. registration model) and projection GAN model. 

Both model implementations make use of the following GIT repositories: 
https://github.com/voxelmorph/voxelmorph
https://github.com/neel-dey/Atlas-GAN/tree/main
"""

import numpy as np
import neurite as ne
import tensorflow as tf

from tensorflow.keras import layers as KL
from tensorflow_addons.layers import SpectralNormalization

import voxelmorph as vxm
from AtlasGAN.networks import conv_block, const_inp, conv_dec_film
from AtlasGAN.film import FiLM

class AtlasGenerationModel(ne.modelio.LoadableModel):
    """
    Atlas Generation Model to generate (fetal brain) templates based on a condition. 

    Template (Atlas) is generated with corresponding label maps. After that the template is warped (by voxelmorph)
    into the subject space (Forward). In addition, the subject is warped by the inverse deformation field into 
    the template space (Backward). 

    No inital template is needed. 

    ARGS: 
        input_shape: Input image shape  (H, W, D)
        n_condns:  dimension of the condition 
        nb_unet_features: VoxelMorph UNet architecture 
        bool_bdir: make bidirectional use of the deformation field
        src_feats: channels of the template 
        seg_feats: channels of the anatomical labels
        kernel_size: kernel size 
        conv_nb_features: number of convolutional features 
        ch: channels 
        init: initialization 

    MODEL INPUT: 
        image_inputs: Input images (H, W, D)
        condn: Input conditions (B, )

    MODEL OUTPUT:
        moved_atlas: generated template warped into the subject space
        moved_labels: associated anatomical labels of the template warped into the subject space (using the same deformation field)
        moved_subject: backward warp. Meaning the subject is warped by the inverse deformation field into the atlas space
        new_atlas: generated template 
        diff_field: deformation field

    """

    @ne.modelio.store_config_args    
    def __init__(self,
                input_shape,
                n_condns,
                nb_unet_features,
                bool_bidir,
                src_feats=1, 
                seg_feats=6, 
                kernel_size = [3, 3, 3],
                conv_nb_features = 8,
                ch=32, 
                init=None
                ):

        input_resolution = tuple(int(ti/2) for ti in input_shape)

        conv_image_shape = (*input_resolution, conv_nb_features)
        image_inputs = tf.keras.layers.Input((*input_shape, src_feats), name='src_input')
        condn = tf.keras.layers.Input(shape=(n_condns,))

        condn_vec = 1.0 * condn
        for _ in range(4):
            condn_vec = KL.Dense(64, kernel_initializer=init,)(condn_vec)
            condn_vec = KL.LeakyReLU(0.2)(condn_vec)

        # Input layer of decoder: learned parameter vector
        const_vec = KL.Lambda(const_inp)(condn)  
        condn_dense= KL.Dense(np.prod(conv_image_shape),
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,),
                                 bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02,),
                                 )(const_vec)

        # Reshape 
        condn_reshape = KL.Reshape(conv_image_shape)(condn_dense)
        condn_film = FiLM(init=init)([condn_reshape, condn_vec])

        # INTENSITY IMAGE DECODER BLOCK
        s1 = conv_block(condn_film, ch, condn_emb=condn_vec, sn=True, init=init,)
        sres = 1.0 * s1

        for _ in range(5):
            sip = sres
            sres = conv_block(sres, ch, condn_emb=condn_vec, sn=True, init=init)
            sres = conv_block(sres, ch, condn_emb=condn_vec, sn=True, init=init)
            sres = sres + sip

        dec_out = conv_dec_film(
            condn_emb=sres,
            condn_vec=condn_vec,
            nb_features=conv_nb_features,
            input_shape=conv_image_shape,
            nb_levels=2,
            conv_size=kernel_size,
            nb_labels=seg_feats,
            sn=True,
            init=init,
        )
        last_tensor = dec_out

        pout = KL.Conv3D(
            1, kernel_size=3, padding='same', name='atlasmodel_c',
            kernel_initializer=init,
        )(last_tensor)

        new_atlas = KL.Activation('tanh')(pout)
        
        # SEGMENTATION DECODER BLOCK
        s1_seg = conv_block(condn_film, ch, condn_emb=condn_vec, sn=True, init=init,)
        sres_seg = 1.0 * s1_seg

        for _ in range(5):
            sip = sres_seg
            sres_seg = conv_block(sres_seg, ch, condn_emb=condn_vec, sn=True, init=init)
            sres_seg = conv_block(sres_seg, ch, condn_emb=condn_vec, sn=True, init=init)
            sres_seg = sres_seg + sip

        dec_out_seg = conv_dec_film(
            condn_emb=sres_seg,
            condn_vec=condn_vec,
            nb_features=conv_nb_features,
            input_shape=conv_image_shape,
            nb_levels=2,
            conv_size=kernel_size,
            nb_labels=seg_feats,
            sn=True,
            init=init,
            prefix='seg'
        )

        last_tensor_seg = dec_out_seg

        new_seg = KL.Conv3D(
            seg_feats, kernel_size=kernel_size, padding='same', name='seg_gen',
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-7),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-7),
        )(last_tensor_seg)

        # VOXELMORPH REGISTRATION MODEL
        atlas_input = tf.keras.Input((*input_shape, src_feats), name='atlas_input')
        warp_input_model = tf.keras.Model(inputs=[condn, atlas_input, image_inputs], outputs=[new_atlas, image_inputs])
        ################################################################################################

        vxm_model = vxm.networks.VxmDense(input_shape, nb_unet_features=nb_unet_features,
                             bidir=bool_bidir, input_model=warp_input_model)

        #OUTPUT
        moved_atlas = vxm_model.references.y_source
        moved_subject = vxm_model.references.y_target

        diff_field = vxm_model.references.pos_flow
        moved_labels = vxm.tf.layers.SpatialTransformer()([new_seg, diff_field])

        ops = [moved_atlas, moved_labels, moved_subject, new_atlas, diff_field]
        
        super().__init__(inputs=[image_inputs, condn], outputs=ops)

class ProjectionDiscriminator(ne.modelio.LoadableModel):
    """
    Projection Discriminator adapted from 
    https://github.com/neel-dey/Atlas-GAN/blob/4f30012d742432c9f565c1676a2fce1342a61e1f/src/networks.py#L461
    
    Added additional downsampling.  

    ARGS: 
        input_shape: Input image shape  (H, W, D)
        n_condns: dimension of the condition 
        ch: channels
        init: initialization 
        src_feats: channels of the template 

    MODEL INPUT: 
        image_inputs: Input images (B, H, W, D, C). Default channel size is 1
        condn: Input conditions (B, )  

    MODEL OUTPUT: 
        op: projection discriminator logits 
    """
    @ne.modelio.store_config_args    
    def __init__(self, 
                 input_shape,
                 n_condns,
                 ch=32,
                 init='orthogonal', 
                 src_feats = 1
                 ):

        inp = tf.keras.layers.Input(shape=(*input_shape, src_feats), name='input_image')

        condn = tf.keras.layers.Input(shape=(n_condns,))
        condn_age_emb = KL.Dense(ch, kernel_initializer=init)(condn)

        # Convolutional sequence:
        down1 = conv_block(inp, ch, sn=True, stride=2, init=init)
        down2 = conv_block(down1, ch*2, sn=True, stride=2, init=init)
        down3 = conv_block(down2, ch*4, sn=True, stride=2, init=init)
        down4 = conv_block(down3, ch*8, sn=True, stride=2, init=init)
        down5 = conv_block(down4, ch*16, sn=True, stride=2, init=init)
        #down6 = conv_block(down5, ch*32, sn=True, stride=2, init=init)

        # If using SN on discriminator output:
        fin1 = conv_block(down5, ch, sn=True, activation=False, init=init)

        # Local projection discriminator feedback:
        op_age = SpectralNormalization(KL.Conv3D(
            1, 1, padding='valid',
            use_bias=True, kernel_initializer=init,
        ))(fin1)

        condn_emb_spatial_age = tf.tile(
            tf.reshape(
                condn_age_emb,
                (tf.shape(condn_age_emb)[0], 1, 1,
                    1, tf.shape(condn_age_emb)[-1]),
            ),
            (1, tf.shape(op_age)[1], tf.shape(op_age)[2],
                tf.shape(op_age)[3], 1),
        )

        op = op_age + tf.reduce_sum(
            condn_emb_spatial_age * fin1, axis=-1, keepdims=True,
        )

        super().__init__(inputs=[inp, condn], outputs=[op])
