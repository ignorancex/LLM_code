import os, shutil
from pathlib import Path



def flatten_KAIST_labels(sequences, root_kaist, target_directory):
    
    labels_direcotry = "annotations-xml-new-sanitized"
    # extract all directories for the given sequences, and image_type combination
    directories = []

    for sequence in sequences:
        set_path = os.path.join(root_kaist, labels_direcotry, f"set{sequence}")
        Vs = sorted(os.listdir(set_path))

        for V in Vs:
            image_dir_path = os.path.join(set_path, V)
            directories.append(image_dir_path)
            if not os.path.exists(image_dir_path):
                raise Exception(f"the path: {image_dir_path} ,does not exist, make sure your arguments are correct")
    
    # create and validate target path
    target_path = os.path.join(root_kaist, target_directory, "labels")
    Path(target_path).mkdir(parents=True, exist_ok=True)
    print(target_path)
    
    _consolidate_and_rename_with_order(target_dir=target_path, source_dirs=directories)
    
    return target_path



def flatten_KAIST_images(sequences, root_kaist, target_directory, image_type):
    

    # extract all directories for the given sequences, and image_type combination
    directories = []

    for sequence in sequences:
        set_path = os.path.join(root_kaist, "images", f"set{sequence}")
        Vs = sorted(os.listdir(set_path))

        for V in Vs:
            image_dir_path = os.path.join(set_path, V, image_type)
            directories.append(image_dir_path)
            if not os.path.exists(image_dir_path):
                raise Exception(f"the path: {image_dir_path} ,does not exist, make sure your arguments are correct")
    
    # create and validate target path
    target_path = os.path.join(root_kaist, target_directory, "images", image_type)
    Path(target_path).mkdir(parents=True, exist_ok=True)
    print(target_path)


    _consolidate_and_rename_with_order(target_dir=target_path, source_dirs=directories)
    return target_path

def _consolidate_and_rename_with_order(target_dir, source_dirs):
    """
    Consolidates images from multiple directories into a single target directory,
    preserving order based on sorted directory and file names, and renaming them sequentially.

    Args:
        target_dir (str): Path to the directory where all images will be stored.
        source_dirs (list): List of directories containing images to consolidate.

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Sort the source directories
    source_dirs = sorted(source_dirs)
    image_id = 1  # Start ID
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist. Skipping.")
            continue
        
        # Sort the images within each directory
        for filename in sorted(os.listdir(source_dir)):
            # Check for common image extensions
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.txt', '.xml')):
                source_path = os.path.join(source_dir, filename)
                new_filename = f"{image_id:07d}" + os.path.splitext(filename)[1]
                target_path = os.path.join(target_dir, new_filename)
                
                shutil.copy(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")
                image_id += 1


def organize_dataset(root_path, dataset_dict):
    """
    Organizes dataset into YOLO format under a root directory.

    Args:
        root_path (str): Root directory where the dataset will be structured.
        dataset_dict (dict): Dictionary with structure:
            {
                "train": {
                    "images": path_to_images,
                    "labels": path_to_labels
                },
                "test": {
                    "images": path_to_images,
                    "labels": path_to_labels
                }
            }

    Raises:
        FileNotFoundError: If any source directory does not exist.
    """
    for split in ['train', 'test']:
        for data_type in ['images', 'labels']:
            src_path = dataset_dict[split][data_type]
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Source path does not exist: {src_path}")

            dest_dir = os.path.join(root_path, split, data_type)
            os.makedirs(dest_dir, exist_ok=True)

            for file_name in os.listdir(src_path):
                full_src = os.path.join(src_path, file_name)
                full_dest = os.path.join(dest_dir, file_name)
                if os.path.isfile(full_src):
                    shutil.copy(full_src, full_dest)

            print(f"Copied {data_type} files for {split} to {dest_dir}")












