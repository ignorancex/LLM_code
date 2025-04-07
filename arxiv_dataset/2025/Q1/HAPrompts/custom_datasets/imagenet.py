"""
sample from imagenet-1k's val set to create 
our validation and test set
"""
import os
import sys
import json
import shutil
import xmltodict
import numpy as np
from tqdm import tqdm


sys.path.append('../')
from utils.directory import get_filename_list, get_sub_dir_list, load_config

# fix RNG seed 
np.random.seed(10086)


"""
#############################################################
manually download ImageNet-1k at 'dataset-home-dir' specified
by the data_paths.yml configuration file
#############################################################
"""


def xml_info(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    data = xmltodict.parse(xml_data)
    
    imname = data['annotation']['filename']+'.JPEG'
    clsname_list = data['annotation']['object']
    if isinstance(clsname_list, list):
        tp = clsname_list[0]['name']
        for obj in clsname_list:
            cl = obj['name']
            if tp != cl:
                print(f'imname {imname} has multiple classes: ({tp}, {cl})')
        clsname = tp
    else:
        clsname = clsname_list['name']
    
    return imname, clsname


def imagenet_val_set_wn_offset_name(src_anno_dir, src_img_dir, dest_dir):
    """
        organize validation images into class folders named by worndet offsets
        1) validation set
        2) test set
        where all class folder names are wordnet offsets

        note: remove 2 synsets/classes: 
        1) sunglass.n.01, n04355933, definition: 'a convex lens that focuses the rays of the sun; used to start a fire' (sun lense?)
        2) projectile.n.01, n04008634, definition: 'a weapon that is forcibly thrown or projected at a target but is not self-propelled'

        the above 2 synsets/classes are confusing with the following 2:
        1) sunglasses.n.01, n04356056, defintiion: 'spectacles that are darkened or polarized to protect the eyes from the glare of the sun'
        2) missile.n.01, n03773504, 'a rocket carrying a warhead of conventional or nuclear explosives; may be ballistic or directed by remote control'

        the sunglass is incorrect annotation, all its images should belong to sunglasses instead
        we just remove the 2 incorrect/confusing classes (and their validation images to keep each class having 50 examples)
    """

    ignore_list = ['n04355933', 'n04008634']

    filenames = get_filename_list(src_anno_dir, '*.xml')
    for fname in tqdm(filenames, total=len(filenames)):
        fpath = os.path.join(src_anno_dir, fname)
        imname, clsname = xml_info(fpath)
        if clsname in ignore_list: continue # ignore 2 confusing/incorrect annotation classes
        src_im_path = os.path.join(src_img_dir, imname)
        dest_class_dir = os.path.join(dest_dir, clsname)
        if not os.path.exists(dest_class_dir):
            os.makedirs(dest_class_dir)
        dest_img_path = os.path.join(dest_class_dir, imname)
        shutil.copy(src_im_path, dest_img_path)

    return


def wn_offset_to_cupl_and_vcd_classname(cupl_imagenet_classes, offset_dict):
    wn_to_cupl_name = dict()
    for k, v in offset_dict.items():
        k = int(k)
        offset = v['id'].split('-')[0] # e.g., '01440764'
        pos = v['id'].split('-')[1] # e.g., 'n'
        posoffset = pos + offset

        cupl_name = cupl_imagenet_classes[k].replace('/', 'or').lower()

        # we first adopt ImageNet-1k class names from CuPL
        # and then adopt the following changes made by VCD
        # the last change (soap bubble) is made by us
        # all methods share the same class names in our experiments
        if cupl_name == 'tights': cupl_name = 'maillot'
        if cupl_name == 'newt': cupl_name = 'eft'
        if cupl_name == 'bubble': cupl_name = 'soap bubble'

        
        wn_to_cupl_name[posoffset] = cupl_name
    
    return wn_to_cupl_name


def imagenet_val_test_split_cupl_vcd_name(src_val_img_dir, dest_dir, imnet_cupl_names_path):
    """
        split original imagenet val set into my
        1) validation set
        2) test set
    """
    
    def cupl_name_preprocess(name):
        name = name.replace(' ','_').lower()
        return name

    with open(imnet_cupl_names_path, 'r') as f:
        tp_dict = json.load(f)
        print(f'loading from {imnet_cupl_names_path}')
    imnet_classes_cupl, imnet_offset_cupl = tp_dict['imnet_classes_cupl'], tp_dict['imnet_offset_cupl']

    # get dictionary {wn_offset: cupl_name}
    cupl_name = wn_offset_to_cupl_and_vcd_classname(imnet_classes_cupl, imnet_offset_cupl)
    offset_names = get_sub_dir_list(src_val_img_dir)
    
    # class folder name: cupl_name + '@' + wordnet_offset
    clsfolder_names = {offset: cupl_name_preprocess(cupl_name[offset])+'@'+offset for offset in offset_names}
    
    for offset in tqdm(offset_names):
        src_offset_dir = os.path.join(src_val_img_dir, offset)
        dest_wn_name_val_dir = os.path.join(dest_dir, 'val', clsfolder_names[offset])
        dest_wn_name_test_dir = os.path.join(dest_dir, 'test', clsfolder_names[offset])

        if not os.path.exists(dest_wn_name_val_dir): os.makedirs(dest_wn_name_val_dir)
        if not os.path.exists(dest_wn_name_test_dir): os.makedirs(dest_wn_name_test_dir)

        # get src image names of a given offset & shuffle
        src_imname = get_filename_list(src_offset_dir, '*.JPEG')
        assert len(src_imname)>0, f'error, no match found by get_filename_list()'
        np.random.shuffle(src_imname)

        # my val/test split
        src_val_names = src_imname[:len(src_imname)//2]
        src_test_names = src_imname[len(src_imname)//2:]

        # copy my val
        for src_name in src_val_names:
            src_path = os.path.join(src_offset_dir, src_name)
            dest_path = os.path.join(dest_wn_name_val_dir, src_name)
            shutil.copy(src_path, dest_path)

        # copy my test
        for src_name in src_test_names:
            src_path = os.path.join(src_offset_dir, src_name)
            dest_path = os.path.join(dest_wn_name_test_dir, src_name)
            shutil.copy(src_path, dest_path)

    return



if __name__ == '__main__':
    config_path = '../data_paths.yml'
    config = load_config(config_path)
    IN1k_download_dir = config['dataset-home-dir']


    # process of original ImageNet-1k validation set
    src_anno_dir = os.path.join(
        IN1k_download_dir,'imagenet-object-localization-challenge',
        'ILSVRC','Annotations','CLS-LOC','val')
    src_img_dir = os.path.join(
        IN1k_download_dir,'imagenet-object-localization-challenge',
        'ILSVRC','Data','CLS-LOC','val')
    dest_wn_offset_dir = config['imagenet'] + '-val-wn-offset'

    # create imagenet-1k original val set with wordnet synset offset 
    # as folder names for respective classes
    imagenet_val_set_wn_offset_name(src_anno_dir, src_img_dir, dest_wn_offset_dir)


    # split original ImageNet-1k validation set into our val/test splits
    src_wn_offset_img_dir = dest_wn_offset_dir
    dest_nl_name_dir = config['imagenet']
    # initialize natural language class names with CuPL names
    imnet_nl_names_path = './imagenet_cupl_names.json'
    imagenet_val_test_split_cupl_vcd_name(
        src_wn_offset_img_dir, 
        dest_nl_name_dir,
        imnet_nl_names_path
    )

    print('done')