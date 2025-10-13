import os, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import glob, random, json
import yaml
import cv2


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


def generate_txt_config_by_components(label_dir, strict = True, include_person=False, include_people=False, include_cyclist=False, include_empty=False, output_file="config.txt"):
    """
    Generates a plain text file with file IDs based on desired components in XML labels.

    Args:
        label_dir (str): Directory containing XML label files.
        include_person (bool): Include files with "person".
        include_people (bool): Include files with "people".
        include_cyclist (bool): Include files with "cyclist".
        include_empty (bool): Include files with no objects.
        output_file (str): Path to save the generated config file.

    Returns:
        None
    """
    
    valid_file_ids = []

    for xml_file in os.listdir(label_dir):
        if not xml_file.endswith(".xml"):
            continue

        # Parse the XML file
        xml_path = os.path.join(label_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Track the presence of each component
        has_person = False
        has_people = False
        has_cyclist = False

        # Check objects in the XML
        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower()
            if class_name == "person":
                has_person = True
            elif class_name == "people":
                has_people = True
            elif class_name == "cyclist":
                has_cyclist = True
        
        # print(f"person {has_person} inc {include_person}, cyclist {has_cyclist} {include_cyclist}, people {has_people} {include_people}")
        
        def check_criteria_non_strict(has_cyclist, has_people, has_person, include_person, include_people, include_cyclist):
            print(f"person {has_person} inc {include_person}, cyclist {has_cyclist} {include_cyclist}, people {has_people} {include_people}")

            if(include_cyclist == True and has_cyclist == True ):
                return True
            elif(include_person == True and has_person == True):
                return True
            elif(include_people == True and has_people == True):
                return True
            elif include_empty == True:
                return True
            
            return False       

        def check_criteria_strict(has_cyclist, has_people, has_person, include_person, include_people, include_cyclist):
            """
            if any objects that are not wanted are present returns false
            """

            if(include_cyclist == False and has_cyclist == True ):
                return False
            elif(include_person == False and has_person == True):
                return False
            elif(include_people == False and has_people == True):
                return False
            elif include_empty == False:
                return False
            
            return True

        # Determine if the file matches the criteria
        if strict == True:
            matches_criteria = check_criteria_strict(has_cyclist, has_people, has_person, include_person, include_people, include_cyclist)
        else:
            matches_criteria = check_criteria_non_strict(has_cyclist, has_people, has_person, include_person, include_people, include_cyclist)
        
        if matches_criteria:
            file_id = os.path.splitext(xml_file)[0]  # Get the file ID without extension
            valid_file_ids.append(file_id)

    # sort IDs
    valid_file_ids.sort()

    # Save the valid file IDs to a plain text file
    with open(output_file, "w") as f:
        f.write("\n".join(valid_file_ids))

    print(f"Config file saved to {output_file} with {len(valid_file_ids)} valid IDs.")


def copy_files_based_on_config(input_dir, output_dir, config_file):
    """
    Copies files from the input directory to the output directory based on the provided config file.
    Automatically detects the common file extension.

    Args:
        input_dir (str): Directory containing the source files.
        output_dir (str): Directory to save the copied files.
        config_file (str): Path to the config file containing file IDs.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect the common file extension
    all_files = glob.glob(os.path.join(input_dir, "*.*"))
    if not all_files:
        print("No files found in input directory.")
        return

    # Assume the first file's extension is the correct one
    _, detected_ext = os.path.splitext(all_files[0])
    print(f"Detected extension: {detected_ext}")

    # Read file IDs
    with open(config_file, "r") as f:
        file_ids = [line.strip() for line in f if line.strip()]

    # Copy files
    for file_id in file_ids:
        source_file = os.path.join(input_dir, f"{file_id}{detected_ext}")
        target_file = os.path.join(output_dir, f"{file_id}{detected_ext}")

        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            # print(f"Copied: {source_file} -> {target_file}")
        else:
            print(f"File not found: {source_file}")


def generate_split_config(reference_dir, output_file, train_ratio=0.8):
    """
    Generates a train-test split based on the reference directory, storing file IDs only (without extensions).

    Args:
        reference_dir (str): Path to the reference directory containing files.
        output_file (str): Path to save the split configuration file.
        train_ratio (float): Proportion of files to include in the training set.

    Returns:
        None
    """
    #     raise Exception("comment this exception out if you are sure you want to do this")

    print("ARE YOU SURE YOU WANT TO GENERATE A NEW CONFIGURATION?")
    X = input("")
    # raise Exception 

    # List all files in the reference directory and strip their extensions
    file_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(reference_dir)])
    
    # Shuffle and split
    random.shuffle(file_ids)
    split_index = int(len(file_ids) * train_ratio)
    train_ids = file_ids[:split_index]
    test_ids = file_ids[split_index:]
    
    # Save to a config file
    split_config = {'train': train_ids, 'test': test_ids}
    with open(output_file, 'w') as f:
        json.dump(split_config, f, indent=4)
    
    print(f"Split configuration saved to {output_file}")


def apply_split_to_directories(config_file, directories, output_base_dir):
    """
    Applies the train-test split to YOLO-style datasets where input is a dictionary 
    with 'images' and 'labels' keys. The output structure will follow:
    
    output_base_dir/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

    Args:
        config_file (str): Path to the split configuration file (with "train" and "test" ID lists).
        directories (dict): Dictionary with keys 'images' and 'labels', each mapping to a directory path.
        output_base_dir (str): Base directory to save split files.
    """
    # Load the split configuration
    with open(config_file, 'r') as f:
        split_config = json.load(f)

    train_ids = set(split_config['train'])
    test_ids = set(split_config['test'])

    for key in ['images', 'labels']:
        dir_path = directories.get(key)
        if not dir_path or not os.path.exists(dir_path):
            print(f"Directory for '{key}' not found or does not exist. Skipping.")
            continue

        # Output paths follow YOLO format
        train_output_dir = os.path.join(output_base_dir, 'train', key)
        test_output_dir = os.path.join(output_base_dir, 'test', key)

        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        for file_name in os.listdir(dir_path):
            file_id, file_ext = os.path.splitext(file_name)

            src_path = os.path.join(dir_path, file_name)

            if file_id in train_ids:
                shutil.copy(src_path, os.path.join(train_output_dir, file_name))
            elif file_id in test_ids:
                shutil.copy(src_path, os.path.join(test_output_dir, file_name))

        print(f"Processed '{key}' from {dir_path} into train/test folders.")


def convert_kaist_to_yolo_labels(source_directory):
    """
    Converts KAIST XML annotations to YOLO-compatible TXT labels with normalization.

    Args:
        source_dir (str): Path to the directory containing XML files.
        output_dir (str): Path to the directory to save YOLO-compatible TXT files.

    Returns:
        None
    """
    output_dir = source_directory + "_yolo_format"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    people = 0
    person = 0
    cyclist = 0

    for xml_file in os.listdir(source_directory):
        if not xml_file.endswith(".xml"):
            continue

        # Parse the XML file
        xml_path = os.path.join(source_directory, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        yolo_annotations = []

        # Iterate over objects in the XML file
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name == "person":
                class_id = 0  # You can customize class IDs here
                person += 1
            elif class_name == "people":
                class_id = 1  # Assign a class ID for "people" if needed
                people += 1
            elif class_name == "cyclist":
                class_id = 2  # Skip objects with other class names
                cyclist += 1
            else:
                continue

            bndbox = obj.find("bndbox")
            x = float(bndbox.find("x").text)  # Top-left x-coordinate
            y = float(bndbox.find("y").text)  # Top-left y-coordinate
            w = float(bndbox.find("w").text)  # Width of the bounding box
            h = float(bndbox.find("h").text)  # Height of the bounding box

            # Normalize bounding box dimensions relative to image size
            x_center = (x + w / 2) / img_width  # Normalize x_center
            y_center = (y + h / 2) / img_height  # Normalize y_center
            width = w / img_width  # Normalize width
            height = h / img_height  # Normalize height

            # Append the YOLO annotation line
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Write YOLO annotations to a .txt file
        txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
        with open(txt_file, "w") as f:
            f.writelines(yolo_annotations)

        print(f"Converted {xml_file} to {txt_file}")
        print(f"person instances {person} cyclist {cyclist} people {people}")
    
    return output_dir


def write_yolo_yaml(config_dict, output_path):
    """
    Writes a YOLO training configuration YAML file.

    Args:
        config_dict (dict): Dictionary with keys 'train', 'val', 'nc', and 'names'.
        output_path (str): Path to save the YAML file.

    Returns:
        None
    """
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


class detection_values:
    def __init__(self, object_class_id, x_center, y_center, bbox_width, bbox_height):
        self.object_class_id = object_class_id 
        self.x_center = x_center 
        self.y_center = y_center
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height
        
        
def get_box_values(box,h,w):
    bb = box.xyxy[0].tolist()
    obj_class = box.cls
    obj_class_id = int(obj_class)
    
    # Convert to YOLO format
    x_center = ((bb[0] + bb[2]) / 2) / w
    y_center = ((bb[1] + bb[3]) / 2) / h
    bbox_width = (bb[2] - bb[0]) / w
    bbox_height = (bb[3] - bb[1]) / h
    
    return detection_values(object_class_id=obj_class_id, x_center=x_center, y_center=y_center, bbox_width=bbox_width, bbox_height=bbox_height)


def generate_yolo_labels(image_dir, model, confidence=0.5):
    """
    This is the standard annotaion format for yolo ".txt"
    image_type is either lwir or visible
    """
    label_dir = image_dir + "_generated_yolo_labels"
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    def filter_images(image_file):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Filter for image files            return True
            return False

    for image_file in os.listdir(image_dir):

        if(filter_images(image_file=image_file)):   
            print(f"faultry image {image_file}")
            continue

        # laod image
        image_path = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # predict
        results = model(image_path, conf=confidence, verbose=False)
        annotation_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

        with open(annotation_file, 'w') as f:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    values = get_box_values(box=box, h=h, w=w)
                    
                    # ensure person
                    if(values.object_class_id == 0):
                        f.write(f"{values.object_class_id} {values.x_center} {values.y_center} {values.bbox_width} {values.bbox_height}\n")

    print(f"Annotations saved to {label_dir}")
    return label_dir


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





