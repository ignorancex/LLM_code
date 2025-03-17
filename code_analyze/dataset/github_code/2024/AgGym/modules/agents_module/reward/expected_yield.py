import numpy as np

def main(self, severity=0):
    # inverse sigmoid shifted function
    # https://www.reddit.com/r/learnmachinelearning/comments/csmrg5/reverse_sigmoid_function/

    reward = 0
    for y, x in self.threat.infect_list:
        infect_duration = self.threat.infect_day_mat[y, x]
        if 46<=self.timestep<=56:
            reward +=(( ((np.exp(-infect_duration+12) / (1+np.exp(-infect_duration+12)))*0.7) - 0.7 ) * self.unit_potential_yield * self.price_per_bushel * self.severity)*500
        elif 57<=self.timestep<=74:
            reward += ( (((np.exp(-infect_duration+12) / (1+np.exp(-infect_duration+12)))*0.6) - 0.6 ) * self.unit_potential_yield * self.price_per_bushel * self.severity)*500
        else:
            reward += ( (((np.exp(-infect_duration+12) / (1+np.exp(-infect_duration+12)))*0.5) - 0.5 ) * self.unit_potential_yield * self.price_per_bushel * self.severity)*500

        # print(f"EY reward: ({reward}) += ({(np.exp(-infect_duration+6) / (1+np.exp(-infect_duration+6)))})")
    # reward += (self.state_space - len(self.threat.infect_list)) - (1 * (self.state_space - len(self.threat.infect_list)))
    # print(f"healthy: {self.state_space - len(self.threat.infect_list)}")
    print('yield_reward={}'.format(np.round(reward, 5)))
    return np.round(reward, 2)