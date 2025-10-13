import os
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


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

