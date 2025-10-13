import os

from yottixel_code.processing import prepare_slide, extract_features
from yottixel_code.search import main_search

def run(query_slide_path_input):
    query_slide_path = query_slide_path_input
    # metadata_path = "/home/data/yottixel/DATABASE/sampled_metadata.csv"
    metadata_path = "/Users/jaiprakashveerla/Documents/Jai/Search_Database/yottixel/DATABASE/sampled_metadata.csv"
    # database_dir = "/home/data/nejm_ai/DATABASE/"
    database_dir = "/Users/jaiprakashveerla/Documents/Jai/Data/DATABASE/"
    database_features_path = "/Users/jaiprakashveerla/Documents/Jai/Search_Database/yottixel/DATABASE/features.pkl"
    query_extension = "svs"
    k = 5
    subtype_search = False
    query_site = None
    save_dir = "./outputs"
    tissue_threshold = 0.7
    kmeans_clusters = 9
    percentage_selected = 15.0
    use_gpu = True
    network_weights_address = './yottixel_code/checkpoints/KimiaNetKerasWeights.h5'
    network_input_patch_width = 1000
    batch_size = 1024
    img_format = 'png'

    if use_gpu:
        os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    query_slide_path = database_dir + query_slide_path
    prepare_slide(query_slide_path, save_dir, tissue_threshold, kmeans_clusters, percentage_selected)
    extract_features(query_slide_path, save_dir, network_weights_address, network_input_patch_width, batch_size, img_format)
    res = main_search(query_slide_path, save_dir, database_dir, database_features_path, metadata_path, query_extension, k, subtype_search, query_site)
    return res