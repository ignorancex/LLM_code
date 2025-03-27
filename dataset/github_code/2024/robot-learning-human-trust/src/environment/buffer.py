import random
import torch

from typing import Tuple


class ReplayBuffer(object):

    def __init__(self,
                 capacity: int) -> object:
        
        self.capacity = capacity
        self.buffer = []
        self.current_trajectory = []
        self.position = 0
    
    def push(self,
             state: torch.Tensor,
             action: torch.Tensor,
             reward: torch.Tensor,
             next_state: torch.Tensor,
             done: bool,
             log_probability: torch.Tensor) -> None:
        
        if len(self.current_trajectory) == 0:
            self.buffer.append([])
        
        self.current_trajectory.append((state.clone(),
                                        action.clone(),
                                        reward.clone(),
                                        next_state.clone(),
                                        done,
                                        log_probability.clone()))
        
        if done:
            self.buffer[self.position] = self.current_trajectory
            self.current_trajectory = []
            self.position = (self.position + 1) % self.capacity

    def sample_trajectory(self) -> Tuple[torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor,
                                         torch.Tensor]:
        
        if len(self.buffer) == 0:
            return None
        
        else:
            sampled_trajectory = random.choice(self.buffer)

            return self.extraction(zip_object=zip(*sampled_trajectory))
    
    def sample_batch(self,
                     batch_size: int) -> Tuple[torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor,
                                               torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        
        else:
            flat_buffer = [experience for trajectory in self.buffer for experience in trajectory]
            sampled_batch = random.sample(flat_buffer, k=batch_size)

            return self.extraction(zip_object=zip(*sampled_batch))
    
    def extraction(self,
                   zip_object: zip) -> Tuple[torch.Tensor,
                                             torch.Tensor,
                                             torch.Tensor,
                                             torch.Tensor,
                                             torch.Tensor,
                                             torch.Tensor]:
        
        states, actions, rewards, next_states, dones, log_probabilities = zip_object

        s = torch.stack([state.clone() for state in states])
        a = torch.stack([action.clone() for action in actions])
        r = torch.stack([reward.clone() for reward in rewards])
        ns = torch.stack([next_state.clone() for next_state in next_states])
        d = torch.tensor([done for done in dones])
        p = torch.stack([probability.clone() for probability in log_probabilities])

        return s, a, r, ns, d, p
    
    def __len__(self):
        return len(self.buffer)
