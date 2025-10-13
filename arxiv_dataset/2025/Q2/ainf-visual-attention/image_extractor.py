import os
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np


def save_compressed_image_msgs_as_png(bag_file, output_folder, ds_size):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    bridge = CvBridge()
    centr = open(os.path.join(output_folder, "centroids.csv"), "w")
    
    centroid = np.zeros(8)

    # Create reader instance and open for reading
    with Reader(bag_file) as reader:
        i = 0
        gray_counter = 0
        gc = 0

        # Iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/cam/camera1/image_raw/compressed':
                print(i * 100.0 / ds_size, "%")
  
                msg = deserialize_cdr(rawdata, connection.msgtype)
                
                # Decode CompressedImage data
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if centroid.size == 0:
                    continue
                
                centroid = np.array(centroid)
                if np.bitwise_or(centroid < -5, centroid > 32).all():
                    gray_counter += 1
                    if gray_counter >= 0.05 * ds_size:
                        gc += 1
                        continue
                
                number = (6 - len(str(i))) * "0" + str(i)
                filename = f"img_{number}.jpg"

                # Save the image as JPG
                cv2.imwrite(os.path.join(output_folder, filename), cv_image)
                centr.write(",".join(map(str, centroid)) + "\n")
                
                i += 1
                if i >= ds_size:
                    break
                
            if connection.topic == '/object_projections':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                centroid = np.array(list(msg.data)[:2])
    
        print(gray_counter - gc)
    centr.close()

if __name__ == '__main__':
    # Specify the path to the ROS 2 bag file
    bag_file_path = 'datasets/red/'

    # Specify the output folder where images will be saved
    output_folder_path = bag_file_path + "output/"

    # Call the function to save image messages as PNGs
    save_compressed_image_msgs_as_png(bag_file_path, output_folder_path, 120000)
