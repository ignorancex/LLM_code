import os, shutil
import xml.etree.ElementTree as ET
import glob, random, json



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
