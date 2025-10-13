from os import makedirs
from os.path import join, splitext, basename, dirname, isfile, abspath
import glob
import pickle
import pathlib

import openslide
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
# from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.backend import bias_add, constant    


from yottixel_code.utils import RGB2HSD, clean_thumbnail, save_mosaic


def prepare_slide(query_slide_path, save_dir, tissue_threshold, kmeans_clusters, percentage_selected):
    patch_save_dir = join(save_dir, splitext(basename(query_slide_path))[0], 'patches')
    makedirs(patch_save_dir, exist_ok=True)

    
    # Check if the mosaics have already been created
    if isfile(join(dirname(patch_save_dir), 'mosaic.h5')):
        print(f"\nGenerated mosaics for {query_slide_path} is found at {join(dirname(patch_save_dir), 'mosaic.h5')}. Skipping to feature extraction.\n")
        return

    slide = openslide.open_slide(query_slide_path)
    print(f"{query_slide_path} loaded to be processed...")

    thumbnail = slide.get_thumbnail((500, 500))
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255) * 1.0


    #TODO: add the logic for screen shots' `pathc_sizw` here.
    try:
        objective_power = int(slide.properties['openslide.objective-power'])
    except KeyError:
        objective_power = 20

    w, h = slide.dimensions
    # at 20x its 1000x1000
    # at 40x its 2000x2000
    patch_size = int((objective_power/20.)*1000)
    
    mask_hratio = (tissue_mask.shape[0]/h)*patch_size
    mask_wratio = (tissue_mask.shape[1]/w)*patch_size
    
    # iterating over patches
    patches = []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        _patches = []
        for j, wi in enumerate(range(0, w, int(patch_size))):
            # check if patch contains 70% tissue area
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)
            patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
            tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size
            _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})
        patches.append(_patches)

    #Next step in the pipeline is to calculate the RGB histogram for each patch (but at 5x).
    flat_patches = np.ravel(patches)
    for patch in tqdm(flat_patches):
        # ignore patches with less tissue coverage
        if patch['tissue_coverage'] < tissue_threshold:
            continue
        # this loc is at the objective power
        h, w = patch['wsi_loc']
        # we will go one level lower, i.e. (objective power / 4)
        # we still need patches at 5x of size 250x250
        # this logic can be modified and may not work properly for images of lower objective power < 20 or greater than 40
        patch_size_5x = int(((objective_power / 4) / 5) * 250.)
        if slide.level_count > 1:
            patch_region = slide.read_region((w, h), 1, (patch_size_5x, patch_size_5x)).convert('RGB')
        else: # just to handle UCLA/slide2.svs and UCLA/slide3.svs
            patch_region = slide.read_region((w, h), 0, (4 * patch_size_5x, 4 * patch_size_5x)).convert('RGB')
        if patch_region.size[0] != 250:
            patch_region = patch_region.resize((250, 250))
        histogram = (np.array(patch_region)/255.).reshape((250*250, 3)).mean(axis=0)
        patch['rgb_histogram'] = histogram

    # Now, run k-means on the RGB histogram features for all selected patches
    selected_patches_flags = [patch['tissue_coverage'] >= tissue_threshold for patch in flat_patches]
    selected_patches = flat_patches[selected_patches_flags]

    kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=0)
    features = np.array([entry['rgb_histogram'] for entry in selected_patches])
    
    try:
        kmeans.fit(features)
    except ValueError:
        print(f"{tcga_slide} was NOT processed successfully. Moving to the next slide.")
        return

    mosaic = []
    for i in range(kmeans_clusters):
        cluster_patches = selected_patches[kmeans.labels_ == i]
        n_selected = max(1, int(len(cluster_patches)*percentage_selected/100.))
        km = KMeans(n_clusters=n_selected, n_init=10, random_state=0)
        loc_features = [patch['wsi_loc'] for patch in cluster_patches]
        ds = km.fit_transform(loc_features)
        c_selected_idx = []
        for idx in range(n_selected):
            sorted_idx = np.argsort(ds[:, idx])
            for sidx in sorted_idx:
                if sidx not in c_selected_idx:
                    c_selected_idx.append(sidx)
                    mosaic.append(cluster_patches[sidx])
                    break
    save_mosaic(mosaic, dirname(patch_save_dir))
    
    for patch in tqdm(mosaic):
        # this loc is at the objective power
        h, w = patch['wsi_loc']
        patch_size_20x = int((objective_power/20.)*1000)
        patch_region = slide.read_region((w, h), 0, (patch_size_20x, patch_size_20x)).convert('RGB')

        if objective_power == 40:
            new_size = (patch_size_20x // 2, patch_size_20x // 2)
            patch_region = patch_region.resize(new_size)

        # Save the patch_region as a PNG file
        output_file = join(patch_save_dir, f"patch_{patch['loc'][0]}_{patch['loc'][1]}.png") 
        patch_region.save(output_file)
    
    print(f"{patch_save_dir} process successfully finished...")
    slide.close()


# feature extractor preprocessing function
def preprocessing_fn(input_batch, network_input_patch_width):
    org_input_size = tf.shape(input_batch)[1]
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') / 255.
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width), 
                                  lambda: scaled_input_batch, 
                                  lambda: tf.image.resize(scaled_input_batch, (network_input_patch_width, network_input_patch_width)))
    # normalization, this is equal to tf.keras.applications.densenet.preprocess_input()---------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = constant(-np.array(mean))
    standardized_input_batch = bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    return standardized_input_batch


# feature extractor initialization function
def kimianet_feature_extractor(network_input_patch_width, weights_address):
    dnx = DenseNet121(include_top=False, weights=weights_address, input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')
    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, arguments={'network_input_patch_width': network_input_patch_width}, input_shape=(None, None, 3), dtype=tf.uint8)])
    kn_feature_extractor_seq.add(kn_feature_extractor)
    return kn_feature_extractor_seq


# feature extraction function
def extract_features(query_slide_path, save_dir, network_weights_address, network_input_patch_width, batch_size, img_format):
    patch_save_dir = join(save_dir, splitext(basename(query_slide_path))[0], 'patches')
    extracted_features_save_path = join(save_dir, splitext(basename(query_slide_path))[0], 'features.pkl')
    # Check if the extracted_features have already been calculated
    if isfile(extracted_features_save_path):
        print(f"\nExtracted features for {query_slide_path} is found at {extracted_features_save_path}. Skipping to indexing and retrieval.\n")
        return
    all_patches = join(patch_save_dir, '*.' + img_format)
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(all_patches)]
    feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address)
    feature_dict = {}
    for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
        batch_end_ind = min(batch_st_ind + batch_size, len(patch_adr_list))
        batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        target_shape = (network_input_patch_width, network_input_patch_width)
        patch_batch = np.array([np.array(Image.open(x).resize(target_shape)) for x in batch_patch_adr_list])
        batch_features = feature_extractor.predict(patch_batch)
        feature_dict.update(dict(zip([x.parents[1].name + "_" + x.name for x in batch_patch_adr_list], list(batch_features))))
        with open(extracted_features_save_path, 'wb') as output_file:
            pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)
            