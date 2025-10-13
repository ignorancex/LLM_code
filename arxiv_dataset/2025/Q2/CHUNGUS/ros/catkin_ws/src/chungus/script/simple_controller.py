#!/usr/bin/python3

#
# Very simple following controller
# This controller does NOT make use of CHUNGUS predictions and is simply a basic point following controller, meant to show
# how to use controller_paused and controller_active.
# 
# Based off of:
# https://github.com/leggedrobotics/wild_visual_navigation/blob/main/wild_visual_navigation_jackal/scripts/carrot_follower.py
#

import math
import rospy
import numpy as np
import tf2_ros
import tf.transformations as tr
import threading
from geometry_msgs.msg import Twist, PoseStamped
from tf2_geometry_msgs import PointStamped


class SimpleController:
    def __init__(self):

        # tf settings
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        # Parameters for controller being paused / active (used to control when to move)
        self.controller_paused_param = rospy.get_param('~controller_paused_param', '/controller_paused')
        self.controller_active_param = rospy.get_param('~controller_active_param', '/controller_active')

        # Initialize paused and active parameters
        self.set_controller_active(False)
        self.controller_paused = self.get_is_paused()

        # Threshold to determine when goal is reached
        self.goal_reached_threshold = rospy.get_param('~goal_reached_threshold', 0.1)

        # Subscribe to the goal message
        self.goal_sub = rospy.Subscriber(rospy.get_param('~goal_topic', '/goal'), PoseStamped, self.goal_callback)
        self.goal_lock = threading.Lock()
        self.goal = None

        # Controller
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        # Gains
        self.gain_linear = rospy.get_param('~gain_linear', 1.0)
        self.gain_angular = rospy.get_param('~gain_angular', 1.5)
        self.max_linear = rospy.get_param('~max_linear', 1.0)
        self.max_angular = rospy.get_param('~max_angular', 1.0)

        # Start node
        rate = rospy.Rate(rospy.get_param('~controller_frequency', 10))
        while not rospy.is_shutdown():
            self.run()
            rate.sleep()
        
        # Cleanup here.
        self.send_twist(0.0, 0.0)

    def set_controller_active(self, active=True):
        """ Set the controller active state """
        self.controller_active = active
        rospy.set_param(self.controller_active_param, active)
    
    def get_is_paused(self):
        """ Return whether the controller should be paused """
        return rospy.get_param(self.controller_paused_param, False)

    def run(self):
        """ Run the controller tick """
        goal = None

        # Get goal (should be in world frame)
        with self.goal_lock:
            if self.goal is not None:
                goal = np.array([float(self.goal[0]), float(self.goal[1]), float(self.goal[2])])

        if goal is None:
            # Goal is invalid - send motionless command
            self.set_controller_active(False)
            self.send_twist(0.0, 0.0)
        else:
            # Set the controller to be active
            self.set_controller_active(True)

            # Get robot pose (in world frame)
            transform = self.lookup_transform("world", "base_link")
            if transform is not None:
                # Pose was retrieved successfully
                position = transform[0]
                orientation = transform[1]

                # Get position difference
                position_difference = np.linalg.norm(goal[:2] - position[:2])
        
                if position_difference < self.goal_reached_threshold:
                    # Goal reached - clear goal and send motionless command
                    self.send_twist(0.0, 0.0)
                    with self.goal_lock:
                        self.goal = None
                else:
                    if self.get_is_paused():
                        # Controller is paused
                        self.send_twist(0.0, 0.0)
                    else:
                        # Controller not paused. Pursue to goal

                        # Get angle difference
                        yaw = tr.euler_from_quaternion(orientation)[2]
                        angle_difference = math.atan2(goal[1] - position[1], goal[0] - position[0]) - yaw
                        # Wrap angle
                        angle_difference = np.fmod(angle_difference + np.pi, 2 * np.pi)
                        if angle_difference < 0:
                            angle_difference = angle_difference + 2 * np.pi
                        angle_difference = angle_difference - np.pi

                        # Send twist
                        self.send_twist(
                            np.clip(self.gain_linear * position_difference, -self.max_linear,  self.max_linear),
                            np.clip(self.gain_angular * angle_difference,   -self.max_angular, self.max_angular),
                        )
            else:
                # Cancel
                self.send_twist(0.0, 0.0)

    def send_twist(self, linear_x, angular_z):
        """ Send a twist command on cmd_vel_pub """
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def lookup_transform(self, target_frame, source_frame, timestamp=None, timeout=1.0):
        """ Lookup a tf transform """
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame,
                                                        rospy.Time(0) if timestamp is None else timestamp,
                                                        rospy.Duration(timeout))
            return (
                (transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z),
                (transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Lookup error: ", e)
            return None
        
    def transform_point(self, target_frame, source_frame, point, timestamp=None, timeout=1.0):
        """ Transform a point using tf """
        if target_frame == source_frame:
            # quickly return point in trivial case
            return point
        try:
            # otherwise use tf to transform it
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            if self.tf_buffer.can_transform(target_frame, source_frame, rospy.Time(0) if timestamp is None else timestamp, rospy.Duration(timeout)):
                transformed = self.tf_buffer.transform(object_stamped=point_stamped, target_frame=target_frame, timeout=rospy.Duration(timeout))
                return (
                    transformed.point.x,
                    transformed.point.y,
                    transformed.point.z
                )
            else:
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Transform point error: ", e)
            return None

    def goal_callback(self, msg):
        """ Callback for goal topic """
        frame_id = msg.header.frame_id
        position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        position_transformed = self.transform_point("world", frame_id, position)

        if position_transformed is None:
            rospy.logerr("Failed to transform goal point to world frame")
        
        with self.goal_lock:
            self.goal = position_transformed


if __name__ == "__main__":
    rospy.init_node('simple_controller')
    SimpleController()
