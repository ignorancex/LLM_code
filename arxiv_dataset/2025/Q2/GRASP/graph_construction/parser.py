import os
import argparse

def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        try:
            os.makedirs(s, exist_ok=True)
            return s
        except:
            raise argparse.ArgumentTypeError(f"readable_dir:{s} is not a valid path")

def file_path(s):
    if os.path.isfile(s):
        return s
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{s} is not a valid file")

def create_parser():
    parser = argparse.ArgumentParser(description='Graph Construction')
    parser.add_argument("--mags", nargs='+', type=int, required=True,
        help="The list of magnifications in which patches were extracted. For example, '5 10 20' if you have extracted patches from 5x, 10x, and 20x magnifications"
        )
    
    parser.add_argument('--feat_location', type=dir_path, required=True,
        help="The path to the directory where your extracted features have been stored. " 
        "Features must have been stored with this format: /path_to_your_folder/subtype/slide/magnification/slide_id.pt "
        "If you have stored features like above, you should give path_to_your_folder to the module: "
        )

    parser.add_argument('--graph_location', type=dir_path, required=True,
        help="The path to the directory where you intend to store your graphs. " 
        "Graphs are stored as DGL graphs, following this structure: /path_to_your_folder/subtype/slide_id.bin "
        )
    
    parser.add_argument('--manifest_location', type=file_path, required=True,
        help="The path to the manifest file of your dataset: YES, IT HAS TO BE GIVEN TO DO MODULE!!! " 
        "If you are using a subset of the dataset, make sure to modify the "
        "the manifest file so that it contains only info regarding the patches that you are working with"
        )
    return parser
