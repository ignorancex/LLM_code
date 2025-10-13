import rclpy
from std_msgs.msg import Float32MultiArray
import math
from rclpy.node import Node
import numpy as np
import aif_model.config as c
from aif_model.utils import add_gaussian_noise

        
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)    

class NeedsPublisher(Node):
    def __init__(self):
        super().__init__('needs_publisher')
        self.needs_publisher = self.create_publisher(Float32MultiArray, '/needs', 10)
        
        self.get_projections_client = self.create_subscription(Float32MultiArray, '/object_projections', self.get_projections, 1)

        self.update_period = 1/20
        self.create_timer(self.update_period, self.timer_callback)
        self.projections = np.ones(c.num_intentions*2)*100 # out of bounds
        rnd = np.random.randn(c.needs_len)
        self.needs = rnd/np.linalg.norm(rnd)
        self.t = 0
        
        self.params = np.abs(np.random.randn(c.needs_len,3))

    def calc_needs(self):
        needs = []
        for i in range(c.needs_len):
            p = self.params[i]
            n = p[0] * np.sin(0.001*p[1] * self.t + p[2]) + p[0]
            n = add_gaussian_noise(n)
            
            dist = np.linalg.norm(self.projections[2*i:2*i+2] - np.array([16,16]))
            n = n - np.exp(-dist/np.e) # if object is close to center, decrease need
            n = np.clip(n,0,1)
            needs.append(n)
            
        self.needs = softmax(needs)#np.array(needs)/np.linalg.norm(np.array(needs))#(0.6 * self.needs + 0.4 * np.array(needs))

    def get_projections(self, msg):
        self.projections = np.array(msg.data)
        
    def timer_callback(self):
        self.t+=1
        self.calc_needs()
        
        msg = Float32MultiArray()
        msg.data = self.needs.tolist()
        self.needs_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    needpub = NeedsPublisher()
    rclpy.spin(needpub)
    needpub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
