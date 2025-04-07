#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import json
from time import time
import argparse

class MinimalSubscriber(Node):
    def __init__(self, topic, obj_name):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(TransformStamped, topic, self.callback, 10)
        self.topic = topic
        self.obj_name = obj_name

    def callback(self, data):
        j_msg = {
            'object': self.topic,
            'translation': [data.transform.translation.x, data.transform.translation.y, data.transform.translation.z],
            'rotation': [data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w],
            'time': time()
        }
        with open("vicon_data_" + str(self.obj_name) + ".txt", "a+") as test_data:
            test_data.write(json.dumps(j_msg) + '\n')
        test_data.close()


def listener():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-topic', type=str, help="ROS topic name ex: /rb1_base_d/vicon/pose ")
    parser.add_argument('-name', type=str, help="object name ex: robot_1")
    args = parser.parse_args()

    rclpy.init()

    minimal_publisher = MinimalSubscriber(args.topic, args.name)

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    listener()
