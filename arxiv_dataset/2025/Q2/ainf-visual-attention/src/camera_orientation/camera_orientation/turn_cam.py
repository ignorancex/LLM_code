import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Quaternion
import math
from rclpy.node import Node
import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return Quaternion(x=qx, y=qy, z=qz, w=qw)

def quaternion_to_euler(quaternion):
    qx = quaternion.x
    qy = quaternion.y
    qz = quaternion.z
    qw = quaternion.w

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class CamOrientationControl(Node):
    def __init__(self):
        super().__init__('cam_orientation_control')
        self.cam_orientation_subscriber = self.create_subscription(
            Quaternion, '/cam_orientation_setter', self.cam_orientation_callback, 1)
        self.cam_orientation_publisher = self.create_publisher(Quaternion, '/actual_cam_orientation', 10)
        
        self.set_entity_state_client = self.create_client(SetEntityState, '/sim/set_entity_state')
        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set Entity State service not available, waiting again...')

        self.get_entity_state_client = self.create_subscription(ModelStates, '/sim/model_states', self.get_cam_orientation, 1)

        self.update_period = 1/20
        self.create_timer(self.update_period, self.timer_callback)

        self.direction = (0,0)

        self.request = SetEntityState.Request()
        self.model_name = 'camera_model'
        self.desired_position = Quaternion()
        self.current_position = Quaternion()
        self.moving = False
        self.speed = 1 # 0.5 rad/s

    def calculate_direction(self):
        r1, p1, y1 = quaternion_to_euler(self.current_position)
        r2, p2, y2 = quaternion_to_euler(self.desired_position)

        # if abs(r1)>1e-5 or abs(r2)>1e-5:
        #     self.get_logger().warning("Roll not possible."+str(r1)+" "+str(r2))
        #     return np.zeros((1,3))

        direction = np.array([0, p2-p1, y2-y1])
        direction/=np.linalg.norm(direction)

        self.direction = (direction[1],direction[2])

    def cam_orientation_callback(self, msg):
        self.desired_position = msg
        print("New desired position",msg)
        self.calculate_direction()
        self.moving = True

    def angle_error(self, desired, current):
        r1,p1,y1 = quaternion_to_euler(desired)
        r2,p2,y2 = quaternion_to_euler(current)
        return np.array((p1-p2, y1-y2))

    def add_step(self, step):
        r, p, y = quaternion_to_euler(self.current_position)
        p+= step*self.direction[0]
        y+= step*self.direction[1]
        return euler_to_quaternion(r,p,y)

    def move_joint(self):
        if self.moving:
            error = self.angle_error(self.desired_position,self.current_position)
            step = self.speed * self.update_period
            if np.linalg.norm(error)> np.deg2rad(1):
                new_position = self.add_step(step)
                self.set_cam_orientation(new_position)
                self.moving = False
            else:
                self.set_cam_orientation(self.desired_position)
                self.direction = (0,0)
                self.moving = False

    def get_cam_orientation(self, msg):
        names = msg.name
        i = names.index(self.model_name)

        self.current_position = msg.pose[i].orientation

    def set_cam_orientation(self, orientation):
        # Set camera orientation in Gazebo
        state = EntityState()
        state.name = self.model_name

        # position
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 1.0

        # orientation
        state.pose.orientation.x = orientation.x
        state.pose.orientation.y = orientation.y
        state.pose.orientation.z = orientation.z
        state.pose.orientation.w = orientation.w
        print("Settng cam orientation to ",orientation)

        req = SetEntityState.Request()
        req._state = state
        self.set_entity_state_client.call_async(req)

    def timer_callback(self):
        # print("timer")
        self.move_joint()

        msg = Quaternion()
        msg.x = self.current_position.x
        msg.y = self.current_position.y
        msg.z = self.current_position.z
        msg.w = self.current_position.w
        self.cam_orientation_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    cam_orientation_control = CamOrientationControl()
    rclpy.spin(cam_orientation_control)
    cam_orientation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
