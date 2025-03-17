import numpy as np
import pdb

def main(self, severity=0):
    pesticide_reward = {0: 0., 1: -.2, 2:-.5, 3:-.9}
    # if len(self.threat.infect_list)!=0:
    #     pdb.set_trace()
    print(f"infect_beforepest: {len(self.threat.infect_list_before_pest)}")
    if pesticide_reward[self.action] != 0 and len(self.threat.infect_list_before_pest) == 0:
        # if len(self.threat.alive_sprayed)!=0:
        #     # reward = (self.state_space) * -(self.unit_pesticide_per_acre[self.action])*10
        # else:
        reward = (self.state_space) * -(self.unit_pesticide_per_acre[self.action])
        # reward= -1
    # elif pesticide_reward[self.action] == 0 and len(self.threat.infect_list_before_pest) == 0:
        # reward= 0.5
    else:
        reward = len(self.threat.infect_list_before_pest) * -(self.unit_pesticide_per_acre[self.action])
        # reward = len(self.threat.infect_list) * -(self.unit_pesticide_per_acre[self.action])
    # print(len(self.threat.infect_list_before_pest))   
    print(f"infect_afterpest: {len(self.threat.infect_list)}")
    # print(f"pesticide reward: ({reward}) = ({pesticide_reward[self.action] * len(self.threat.infect_list)})")
    print(f"pesticide reward: ({reward}) = ({-(self.unit_pesticide_per_acre[self.action]) * len(self.threat.infect_list)})")
    print((self.state_space) * -(self.unit_pesticide_per_acre[self.action]))
    print(np.round((self.state_space) * pesticide_reward[self.action] * (int(17/self.num_of_plots_per_acre )), 5))
    print(np.round(len(self.threat.infect_list)* pesticide_reward[self.action] * (int(17/self.num_of_plots_per_acre )),5))
    print(-(self.unit_pesticide_per_acre[self.action]))
    # print('yield pesticide={}'.format(reward))
    return reward

