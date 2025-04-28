from echogains.CONST import DEFAULT_CONFIG
import echogains.CONST as CONST
import os
import echogains.RePaint.conf_mgt as conf_mgt
from echogains.RePaint.test import main

def load_default_config(model_name,keep_path,masks_path,buffer_path,output_path):
    '''
    Load the default config and fill in the necessary fields for running RePaint inference with the specified model.
    Note that the default config assumes a certain architecture for the model.
    See the RePaint documentation for more information if you want to run a model with a different architecture.
    :param model_name: str
        The name of the model to use for inference. This model should be loaded using echogains.set_up_model or
        echogains.download_and_set_up_model
    :param keep_path: str
        Path to the directory containing the transformed images with the pixels that are kept.
    :param masks_path: str
        Path to the directory containing the RePaint masks. This tells repaint which pixels to keep.
    :param buffer_path: str
        Path to the directory used for temporary storage during inference
    :return: dict
        The default config with the necessary fields filled in
    '''
    model_file_name = f"{model_name}.pt"
    DEFAULT_CONFIG['model_path'] = os.path.join(CONST.MODEL_DIR, model_file_name)
    srs = output_path
    lrs = os.path.join(buffer_path, "lrs")
    gts = os.path.join(buffer_path, "gts")
    gt_keep_masks = os.path.join(buffer_path, "gt_keep_masks")

    nb_keep_files = len(os.listdir(keep_path))
    nb_mask_files = len(os.listdir(masks_path))
    assert nb_keep_files == nb_mask_files,\
        f"Number of files in keep_path ({nb_keep_files}) does not match number of files in masks_path ({nb_mask_files})"
    nb_files_in_dataset = nb_keep_files
    DEFAULT_CONFIG['data']['eval']['inference']['gt_path'] = keep_path
    DEFAULT_CONFIG['data']['eval']['inference']['mask_path'] = masks_path
    DEFAULT_CONFIG['data']['eval']['inference']['paths']['srs'] = srs
    DEFAULT_CONFIG['data']['eval']['inference']['paths']['lrs'] = lrs
    DEFAULT_CONFIG['data']['eval']['inference']['paths']['gts'] = gts
    DEFAULT_CONFIG['data']['eval']['inference']['paths']['gt_keep_masks'] = gt_keep_masks
    DEFAULT_CONFIG['data']['eval']['inference']['max_len'] = nb_files_in_dataset
    return DEFAULT_CONFIG


def run_repaint(conf):
    '''
    Wrapper around the main inference function in RePaint.
    :param conf: dict
        The config dictionary to use for inference. See RePaint documentation for more information on the config format
    '''
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(conf)
    main(conf_arg)

