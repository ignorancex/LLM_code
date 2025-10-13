#!/usr/bin/python3

#
# Simple node that publishes a world->odom transform.
#

import time
import rospy
import tf2_ros
import tf.transformations as tr
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import TransformStamped


class GazeboWorldPublisher:
    def __init__(self):
        # tf settings
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)
        
        # Subscribe to the link state message
        self.link_sub = rospy.Subscriber("/gazebo/link_states/", LinkStates, self.link_state_callback)
        self.last_link_tf_time = float('-inf')

        # Spin
        rospy.spin()

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

    def link_state_callback(self, msg):
        """ Callback for link state topic """

        index = -1
        for i, n in enumerate(msg.name):
            if n == "jackal::base_link":
                index = i
        
        # Found the base_link
        if index >= 0:
            # Publish a tf transform from world->odom to allow for world->base_link transformations
            world_position = (msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z)
            world_orientation = (msg.pose[index].orientation.x, msg.pose[index].orientation.y, msg.pose[index].orientation.z, msg.pose[index].orientation.w)

            # Lookup the odom base_link transform
            pose_odom_to_base = self.lookup_transform("odom", "base_link")
            if pose_odom_to_base is None:
                rospy.logerr("Could not get odom base_link transform")
                return
            
            M_odom_to_base = tr.concatenate_matrices(tr.translation_matrix(pose_odom_to_base[0]), tr.quaternion_matrix(pose_odom_to_base[1]))
            M_world_to_base = tr.concatenate_matrices(tr.translation_matrix(world_position), tr.quaternion_matrix(world_orientation))
            inversed_transform = tr.concatenate_matrices(M_world_to_base, tr.inverse_matrix(M_odom_to_base)) # M_world_to_base * M_base_to_odom => M_world_to_odom
            
            current_time = time.time()
            if abs(current_time - self.last_link_tf_time) > 1.0 / 50.0:
                # Publish transform from world->odom
                position = tr.translation_from_matrix(inversed_transform)
                orientation = tr.quaternion_from_matrix(inversed_transform)

                tf_broadcaster = tf2_ros.TransformBroadcaster()
                transform = TransformStamped()
                transform.header.stamp = rospy.Time.now()
                transform.header.frame_id = "world"
                transform.child_frame_id = "odom"
                transform.transform.translation.x = position[0]
                transform.transform.translation.y = position[1]
                transform.transform.translation.z = position[2]
                transform.transform.rotation.x = orientation[0]
                transform.transform.rotation.y = orientation[1]
                transform.transform.rotation.z = orientation[2]
                transform.transform.rotation.w = orientation[3]
                tf_broadcaster.sendTransform(transform)
                self.last_link_tf_time = current_time
            

if __name__ == "__main__":
    rospy.init_node('gazebo_world_publisher')
    GazeboWorldPublisher()
