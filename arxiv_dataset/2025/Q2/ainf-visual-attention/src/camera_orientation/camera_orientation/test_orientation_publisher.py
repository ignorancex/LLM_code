from camera_orientation.turn_cam import euler_to_quaternion, quaternion_to_euler
import rclpy
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
import numpy as np

class OrientationPublisher(Node):
    def __init__(self):
        super().__init__('orientation_publisher')
        self.cam_orientation_publisher = self.create_publisher(Quaternion, '/cam_orientation_setter', 10)

        self.update_period = 3
        self.create_timer(self.update_period, self.timer_callback)

    def timer_callback(self):
        r=0
        p = np.random.randint(-45, 10)
        y = np.random.randint(-45, 45)
        q = euler_to_quaternion(r,np.deg2rad(p),np.deg2rad(y))
        print(p,y)

        msg = q
        self.cam_orientation_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    or_pub = OrientationPublisher()
    rclpy.spin(or_pub)
    or_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
