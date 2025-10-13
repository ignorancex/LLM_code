from aif_model.agent import Agent
import numpy as np
import torch
import aif_model.config as c
import aif_model.utils as utils
import time
import sys

from camera_orientation.turn_cam import euler_to_quaternion, quaternion_to_euler
from geometry_msgs.msg import Quaternion
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def project(position):
    '''
    Project 3D position onto image coordinates
    '''
    f = c.width / (2 * np.tan(c.horizontal_fov/2))
    cent = (c.width/2, c.height/2) # get center point
    K = np.array([[f, 0, cent[0]],
                  [0, f, cent[1]],
                  [0, 0, 1]])
    
    R = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    
    t = np.array([0,0,1])

    r_p = np.array([[0,0,1],
                    [-1,0,0],
                    [0,-1,0]])
    transformed = r_p.T.dot(R.T.dot(position - t))

    projected = K.dot(transformed)

    normalized = np.clip(projected/projected[2],-10,42)
    return normalized[0:2]

class AutoTrial(Node):
    '''
    Auto trial node
    '''

    def __init__(self, num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act):
        super().__init__('automatic_trial_execution')
        self.cam_orientation_publisher = self.create_publisher(Quaternion, '/cam_orientation_setter', 1)
        self.cam_orientation_subscriber = self.create_subscription(Quaternion, '/actual_cam_orientation', self.cam_orientation_callback, 1)
        self.image_subscriber = self.create_subscription(Image,'cam/camera1/image_raw', self.image_callback, 1)
        self.gazebo_client = self.create_client(SetEntityState, '/sim/set_entity_state')
        self.bridge = CvBridge()

        # Trial variables
        self.num_trials = num_trials
        self.init_t = init_t # steps
        self.cue_t = cue_t # steps
        self.coa_t = coa_t # steps
        self.step_max = step_max # steps
        self.endogenous = endo
        self.valid = valid
        self.action_enabled = act
        
        # init agent
        self.agent = None
        self.step = 0 

        # init sensory input
        self.proprioceptive = np.zeros(c.prop_len)
        self.visual = np.zeros((1,c.channels,c.height,c.width))
        self.needs = np.zeros((c.needs_len))

        self.got_data = np.full((2),False)

        formatted = "_".join(map(str,[num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act]))
        self.log_name = f"act_inf_logs/experiments/log_{formatted}.csv"
        self.target_dist = 0

    def reset(self):
        '''
        Reset for next trial
        '''
        self.proprioceptive = np.zeros(c.prop_len)
        self.visual = np.zeros((1,c.channels,c.height,c.width))
        self.needs = np.zeros((c.needs_len))
        self.agent = None
        self.step = 0 
        self.move_ball((-1.0, 0.0, 1.0))
        self.reset_cam()

    def wait_data(self):
        '''
        Wait for data until beginning
        '''
        print("<Auto Trials> Waiting for data...")
        while not self.got_data.all()==True:
            rclpy.spin_once(self)

        self.trials()

    def cam_orientation_callback(self, msg):
        r, p, y = quaternion_to_euler(msg)
        p = np.rad2deg(p)
        y = np.rad2deg(y)
        self.proprioceptive = np.array([p,y])
        self.got_data[0] = True

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.visual = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.visual = torch.tensor(np.transpose(self.visual,(2,0,1))).unsqueeze(0)
            
            # scale to [0,1]
            self.visual = self.visual/255
            self.got_data[1] = True
        except Exception as e:
            self.get_logger().error('Error processing image: {0}'.format(e))

    def publish_action(self, action):
        '''
        Action publisher
        '''
        desired = self.proprioceptive + c.dt * action
        q = euler_to_quaternion(0,np.deg2rad(desired[0]),np.deg2rad(desired[1]))

        msg = q
        self.cam_orientation_publisher.publish(msg)

    def log(self, v):
        '''Log given values'''
        with open(self.log_name,"a") as l:
            l.write(','.join(map(str, v))+"\n")

    def log2(self):
        '''Log perceived target position and covert focus'''
        targets = self.agent.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len] # grab visual positions of objects
        focus = self.agent.mu[0,-2:]
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        focus = utils.denormalize(focus)
        concat = np.concatenate((targets,focus,))
        with open(self.log_name,"a") as l:
            l.write(','.join(map(str, concat))+"\n")

    def generate_cues(self):
        '''
        Generate cues and target position
        '''
        endo_cue = np.ones(3)
        exo_cue = np.array([-1.0,0.0,1.0])
        ball_true = np.array([-1.0,0.0,1.0])

        exo_cue = np.array([4,np.random.random(1)[0]*4 - 2, np.random.random(1)[0]*4 - 1])#np.array([4,-1.5,1])#
        projection = project(exo_cue)
        normalized  = utils.normalize(projection)
        endo_cue[0] = normalized[0]
        endo_cue[1] = normalized[1]

        if self.valid:
            ball_true = exo_cue
        else:
            # mirror exo_cue
            ball_true = np.array([4,-exo_cue[1], 2 - exo_cue[2]])

        self.target_dist = np.linalg.norm(project(ball_true)-np.array([16,16]))
        return endo_cue, exo_cue, ball_true

    def move_ball(self, position):
        '''
        Translate sphere to given position
        '''
        state = EntityState()
        state.name = "red_sphere"

        # position
        state.pose.position.x = position[0]
        state.pose.position.y = position[1]
        state.pose.position.z = position[2]

        req = SetEntityState.Request()
        req._state = state
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self, future) 

    def reset_cam(self):
        '''
        Reset camera to home position
        '''
        state = EntityState()
        state.name = "camera_model"

        # position
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 1.0

        # orientation
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        req = SetEntityState.Request()
        req._state = state
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)  # Wait for the result

    def trials(self):
        '''
        Begin trials
        '''

        print("<Auto Trials> Starting trials")
        for i in range(self.num_trials):
            self.reset_cam()
            
            # initialize logging variables
            reaction_time = -1
            perceived = False
            reach_time = -1
            reached = False
            cue_center_dist = -1
            target_center_dist = -1
            cue_target_dist = -1
            
            # init agent
            self.agent = Agent()
            self.agent.init_belief(self.needs,self.proprioceptive,self.visual)

            # one trial
            print("<Auto Trials> Starting trial " + str(i+1) + "/" +str(self.num_trials))
            # INIT
            for _ in range(self.init_t):
                self.update()
                # if self.ball_perceived() and not perceived:
                #     reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     perceived=True
                # if self.ball_reached() and not reached:
                #     reach_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     reached = True

            # GENERATE CUES
            endo_cue, exo_cue, ball_true = self.generate_cues()

            cue_pos = project(exo_cue)
            ball_pos = project(ball_true)
            cue_center_dist = np.linalg.norm(cue_pos - np.array([16,16]))
            target_center_dist = np.linalg.norm(ball_pos - np.array([16,16]))
            cue_target_dist = np.linalg.norm(cue_pos - ball_pos)

            # SET CUES
            if self.endogenous: # set self.needs
                self.needs = endo_cue
            else: # move ball
                self.move_ball(exo_cue)
            
            # CUE
            print("<Auto Trials> Cueing...")
            for _ in range(self.cue_t):
                self.update()
                # if self.ball_perceived() and not perceived:
                #     reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     perceived = True
                # if self.ball_reached() and not reached:
                #     reach_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     reached = True

            # REMOVE CUE
            if self.endogenous: # set self.needs
                self.needs = np.zeros((c.needs_len))
            else: # move ball
                self.move_ball((-1.0, 0.0, 1.0))

            # COA: Cue Onset Asynchrony
            print("<Auto Trials> COA...")
            for _ in range(self.coa_t):
                self.update()
                # if self.ball_perceived() and not perceived:
                #     reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     perceived = True
                # if self.ball_reached() and not reached:
                #     reach_time = self.step - self.init_t - self.cue_t - self.coa_t
                #     reached = True

            # SET TARGET
            print("<Auto Trials> Setting Target...")
            self.move_ball(ball_true)

            # TARGET
            print("<Auto Trials> Perception...")
            while (self.step - self.init_t - self.cue_t - self.coa_t) <= self.step_max:
                try:
                    self.update()
                except:
                    print("<Auto Trials> Failed trial")
                    break
                if self.ball_perceived() and not perceived:
                    reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                    perceived = True
                    if not self.action_enabled:
                        break
                if self.ball_reached() and not reached:
                    reach_time = self.step - self.init_t - self.cue_t - self.coa_t
                    reached = True
                    break

            if not perceived:
                reaction_time = self.step_max
            if not reached and self.action_enabled:
                reach_time = self.step_max

            # log data
            v = [reaction_time,reach_time,cue_center_dist,target_center_dist,cue_target_dist]
            self.log(v)

            # reset
            print("<Auto Trials> Resetting", end="")
            self.reset()
            for _ in range(10):
                print(".",end="")
                time.sleep(0.5)
                rclpy.spin_once(self) 
            print("")

        print("<Auto Trials> Finished",self.target_dist)

    def ball_perceived(self):
        '''
        True if sphere is perceived (presence greater than 0.1)
        '''
        return self.agent.mu[0,c.needs_len+c.prop_len+c.prop_len]>0.1
    
    def ball_reached(self):
        '''
        True if sphere is focused on in the center of the image
        '''
        ball_coords = self.agent.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len]
        return np.linalg.norm(ball_coords) < (2/16) and self.ball_perceived()
    
    def update(self):
        '''
        One inference step
        '''
        rclpy.spin_once(self)
        # get sensory input
        S =  self.needs, self.proprioceptive, self.visual

        action = self.agent.inference_step(S)
        action = utils.add_gaussian_noise(action)

        if self.action_enabled:
            self.publish_action(action)

        self.step+=1
        # self.log2()

def parse_custom_args():
    '''
    Parse command line arguments
    '''
    parsed_args = {}
    key = None
    
    for arg in sys.argv[1:]:
        if arg.startswith("--"):  # Detect argument name
            key = arg.lstrip("-")  # Remove leading '--'
            parsed_args[key] = True  # Default to True if no value follows
        elif key is not None:  # Detect argument value
            parsed_args[key] = arg
            key = None  # Reset key after storing value
    
    return parsed_args

def parse_trial_args(num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act):
    """
    Argument parsing
    """
    dic = parse_custom_args()
    if "trials" in dic.keys():
        num_trials = int(dic["trials"])
    if "init" in dic.keys():
        init_t = int(dic["init"])
    if "cue" in dic.keys():
        cue_t = int(dic["cue"])
    if "coa" in dic.keys():
        coa_t = int(dic["coa"])
    if "max" in dic.keys():
        step_max = int(dic["max"])
    if "endo" in dic.keys():
        endo = dic["endo"] == "true"
    if "valid" in dic.keys():
        valid = dic["valid"] == "true"
    if "act" in dic.keys():
        act = dic["act"] == "true"
    
    return num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act


def main():
    rclpy.init(args=sys.argv)

    # Default args
    num_trials = 1
    init_t = 10
    cue_t = 50
    coa_t = 100
    step_max = 1000
    endo = True
    valid = True
    act = False

    num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act = parse_trial_args(num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act)

    print(num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act)
    auto = AutoTrial(num_trials, init_t, cue_t, coa_t, step_max, endo, valid, act)
    auto.wait_data()
    auto.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()