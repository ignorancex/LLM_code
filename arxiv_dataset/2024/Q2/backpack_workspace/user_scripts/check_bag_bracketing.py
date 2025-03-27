import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
import argparse

TOPIC_CAM1 = "/stereo/camera1/metadata"
TOPIC_CAM2 = "/stereo/camera2/metadata"
EXPOSURES = [25, 100, 400, 1600, 6400, 25600]
EXPECTED_FPS = 22


def extract_metadata_from_rosbag(bag_file, topic_name):

    data = []
    with AnyReader([Path(bag_file)]) as reader:
        # iterate over messages
        connections = [x for x in reader.connections if x.topic == topic_name]
        for connection, ros_time, rawdata in tqdm(reader.messages(connections=connections)):

            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            exposure_time = msg.ExposureTime
            temp = msg.DeviceTemperature
            data.append([timestamp, exposure_time, temp])
            
    return pd.DataFrame(data, columns=["timestamp", "exposure_time", "temperature"])


def closest_exposure(exposure, sequence):
    closest = min(enumerate(sequence), key=lambda x: abs(x[1] - exposure))
    return closest[1], closest[0]


def verify_exposure_sequence(df, sequence):
    
    # Initialize the break counter
    break_nbr = []

    prev_exposure, prev_index = closest_exposure(df['exposure_time'].iloc[0], sequence)
    for i in range(1, len(df)):
        current_exposure, current_index = closest_exposure(df['exposure_time'].iloc[i], sequence)
        timestamp = df['timestamp'].iloc[i]
        
        # Check if the current exposure is within the buffer range of the previous and next exposures
        if (((prev_index + 1) % len(sequence)) != current_index):
            break_nbr.append(timestamp)

        prev_index = current_index
        
    print("------------------------------------------")
    print(f"Breaks at indexes: {break_nbr}")
    print(f"Number of break sequences: {len(break_nbr)}")
    print(f"Maximum temperature: {df['temperature'].max()} C")


def main():
    parser = argparse.ArgumentParser(description="Extract and verify exposure sequence from ROS bag.")
    parser.add_argument("bag_file", type=str, help="Path to the ROS bag file.")
    args = parser.parse_args()

    bag_file = args.bag_file

    print("---------------- CAMERA 1 -----------------")
    df = extract_metadata_from_rosbag(bag_file, TOPIC_CAM1)
    verify_exposure_sequence(df, EXPOSURES)

    print("---------------- CAMERA 2 -----------------")
    df = extract_metadata_from_rosbag(bag_file, TOPIC_CAM2)
    verify_exposure_sequence(df, EXPOSURES)


if __name__ == "__main__":
    main()
