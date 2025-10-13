#!/usr/bin/env python

import rospy
import tf
import math
import threading

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetModelState, SetModelStateRequest

class PosePublisherPlugin:
    def __init__(self):
        self.model_name = "your_model_name"  # Change this to the name of your model in Gazebo
        self.pose_topic = "/your_pose_topic"  # Change this to the desired ROS topic name for publishing poses
        self.rate = 10  # Publishing rate in Hz

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)
        self.tf_listener = tf.TransformListener()

        self.set_model_state_service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.thread = threading.Thread(target=self.publish_pose)
        self.thread.daemon = True
        self.thread.start()

    def model_states_callback(self, msg):
        try:
            index = msg.name.index(self.model_name)
            self.model_pose = msg.pose[index]
        except ValueError:
            pass

    def publish_pose(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform("/world", self.model_name, rospy.Time(0))
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "/world"
                pose_msg.pose.position.x = trans[0]
                pose_msg.pose.position.y = trans[1]
                pose_msg.pose.position.z = trans[2]
                pose_msg.pose.orientation.x = rot[0]
                pose_msg.pose.orientation.y = rot[1]
                pose_msg.pose.orientation.z = rot[2]
                pose_msg.pose.orientation.w = rot[3]
                self.pose_pub.publish(pose_msg)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            rate.sleep()

    def set_model_pose(self, x, y, z, roll, pitch, yaw):
        req = SetModelStateRequest()
        req.model_state.model_name = self.model_name
        req.model_state.pose.position.x = x
        req.model_state.pose.position.y = y
        req.model_state.pose.position.z = z
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        req.model_state.pose.orientation.x = quaternion[0]
        req.model_state.pose.orientation.y = quaternion[1]
        req.model_state.pose.orientation.z = quaternion[2]
        req.model_state.pose.orientation.w = quaternion[3]
        try:
            resp = self.set_model_state_service(req)
            if not resp.success:
                rospy.logwarn("Failed to set model state: {}".format(resp.status_message))
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

if __name__ == "__main__":
    rospy.init_node("pose_publisher_plugin")
    pose_publisher_plugin = PosePublisherPlugin()

    # Example to set the model pose
    pose_publisher_plugin.set_model_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rospy.spin()

