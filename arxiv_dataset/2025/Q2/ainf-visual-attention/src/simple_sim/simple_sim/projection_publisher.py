import rclpy
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import EntityState, ModelStates
from geometry_msgs.msg import Quaternion
import math
from rclpy.node import Node
import numpy as np
import aif_model.config as c
from simple_sim.object_spawner import CustomObject
#from itertools import product

# Quaternion o rotation matrix
def quaternion_rotation_matrix(Q):
    # normalize Q
    Q /= np.linalg.norm(Q)

    # Extract the values from Q
    qx = Q[0]
    qy = Q[1]
    qz = Q[2]
    qw = Q[3]
     
    # First row of the rotation matrix
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
     
    # Second row of the rotation matrix
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy * qz - qx * qw)
     
    # Third row of the rotation matrix
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx**2 + qy**2) 
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
    
def project(position, R, t, K):
    #print("original",position)
    r_p = np.array([[0,0,1],
                    [-1,0,0],
                    [0,-1,0]])
    transformed = r_p.T.dot(R.T.dot(position - t))
    #print("transformed",transformed)
    #print("K",K)
    projected = K.dot(transformed)
    #print("projected",projected)
    normalized = np.clip(projected/projected[2],-10,42)
    return normalized[0], normalized[1] # mirror y

class ProjPublisher(Node):
    def __init__(self):
        super().__init__('projection_publisher')
        self.projection_publisher = self.create_publisher(Float32MultiArray, '/object_projections', 10)
        
        self.get_entity_state_client = self.create_subscription(ModelStates, '/sim/model_states', self.get_object_states, 1)

        self.update_period = 1/20
        self.create_timer(self.update_period, self.timer_callback)
        
        self.colors = list(CustomObject.colors.keys())
        self.shapes = list(CustomObject.shapes.keys())

        self.camera_name = 'camera_model'
        self.cam_position = np.array([0,0,1])
        self.cam_orientation = np.array([0,0,0,1])
        
        # intrinsic matrix
        f = c.width / (2 * np.tan(c.horizontal_fov/2))
        cent = (c.width/2, c.height/2) # get center point
        self.K = np.array([[f, 0, cent[0]],
                           [0, f, cent[1]],
                           [0, 0, 1]])
        self.coordinates = np.zeros((2,2))


    def get_object_states(self, msg):
        names = msg.name
        poses = msg.pose
        
        # camera info
        cam_index = names.index(self.camera_name)
        position = poses[cam_index].position
        orientation = poses[cam_index].orientation
        self.cam_position = np.array([position.x,position.y,position.z])
        self.cam_orientation = np.array([orientation.x,orientation.y,orientation.z,orientation.w])
        R = quaternion_rotation_matrix(self.cam_orientation)
                
        for i in range(len(names)):
            if not names[i].startswith("invisible") and names[i] != self.camera_name:
                name = names[i]
                name = name.split("_")
                index = 0 if name[0]=="red" else 1 #list(CustomObject.colors.keys()).index(name[0])
                
                # calculate projection
                position = poses[i].position
                position = np.array([position.x,position.y,position.z])
                u, v = project(position, R, self.cam_position, self.K)
                #print(u,v)
                
                self.coordinates[index,0] = u
                self.coordinates[index,1] = v

    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self.coordinates.flatten().tolist()
        self.projection_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    projpub = ProjPublisher()
    rclpy.spin(projpub)
    projpub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
