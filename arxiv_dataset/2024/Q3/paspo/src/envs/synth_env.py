import gym as gymnasium
import torch
import numpy as np
import polytope as pc
import cvxpy as cp
import math

def distance_to_polytope_surface(point, A, b):
    # Define the optimization variable
    x = cp.Variable(len(point))

    # Define the objective function (the square of the Euclidean distance)
    objective = cp.Minimize(cp.sum_squares(x - point))

    # Define the constraints (the point x must be inside the polytope)
    constraints = [A @ x <= b for A, b in zip(A, b)]

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the minimum distance
    return np.sqrt(problem.value), x.value


class MLPRewardNetwork(torch.nn.Module):
    def __init__(self, n_dimension, n_states, polytope, seed):
        super(MLPRewardNetwork, self).__init__()

        self._polytope = polytope
        self._n_dimension = n_dimension
        self._n_states = n_states

        self.scale_factor = 30.0
        
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(1 + n_dimension, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        
        self.np_rng = np.random.default_rng(seed)
        
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)
        
        self._mlp.apply(self.init_weights)
        

    def init_weights(self, m):
        with torch.no_grad():
            if type(m) == torch.nn.Linear:
                if m.out_features == 1:
                    torch.nn.init.uniform_(m.weight, 0.0, 0.1, generator=self.torch_rng)
                    m.bias.fill_(self.np_rng.random()*0.3)
                else:
                    torch.nn.init.xavier_normal_(m.weight, generator=self.torch_rng)
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(m.bias, -bound, bound, generator=self.torch_rng)

                    
                
    def forward(self, state, action):
        if self._polytope is None:
            if len(action.shape) > 1:
                return self._mlp(torch.cat([torch.tensor(state, dtype=torch.float32)[None].repeat((action.shape[0],  1)), action], dim=-1)) * self.scale_factor
            else:
                return self._mlp(torch.cat([torch.tensor(state, dtype=torch.float32)[None], action], dim=-1)) * self.scale_factor
        
        if len(action.shape) == 1 and not np.isclose(action.sum().item(), 1.0):  # in case the action is not a simplex
            if False:
                return torch.tensor([-1.0]).repeat(len(action.shape)).view(-1, 1)
            else:
                return -torch.abs(action.sum(dim=-1) - 1.0)

        if len(action.shape) > 1:
            reward = self._mlp(torch.cat([torch.tensor(state, dtype=torch.float32)[None].repeat((action.shape[0],  1)), action], dim=-1)) * self.scale_factor

            indices = self._polytope.contains(action.detach().numpy().T)
            if not indices.any():
                reward[~indices] = torch.sqrt(torch.sum((action[~indices] - torch.tensor(np.array(
                    [distance_to_polytope_surface(ac, self._polytope.A, self._polytope.b)[1] for ac in
                     action[~indices]]), dtype=torch.float32)) ** 2,
                                                        dim=-1)) * -5.0 - 10.0  # needs to be differentiable

            return reward
        else:

            if self._polytope.__contains__(action.detach().numpy()):
                reward = self._mlp(torch.cat([torch.tensor(state, dtype=torch.float32)[None], action], dim=-1)) * self.scale_factor
            else:
                reward = torch.sqrt(torch.sum((action - torch.tensor(
                    distance_to_polytope_surface(action.detach().numpy(), self._polytope.A, self._polytope.b)[1],
                    dtype=torch.float32)) ** 2)) * -5.0 - 10.0  # needs to be differentiable

            return reward

class DistRewardNetwork(torch.nn.Module):
    def __init__(self, n_dimension, n_states, polytope, seed):
        super(DistRewardNetwork, self).__init__()

        self._polytope = polytope
        self._n_dimension = n_dimension
        self._n_states = n_states

        n_dist = 14 if n_dimension == 3 else 14 # 10 + 2 ** n_dimension

        self._dist = []
        
        rng = np.random.default_rng(seed)
        
        for state in range(n_states):
            #p_mixture = torch.tensor([1.0 / n_dist] * n_dist)
            p_mixture = torch.from_numpy(rng.rand(n_dist))

            if self._n_dimension == 3:
                self._dist.append(torch.distributions.MixtureSameFamily(torch.distributions.Categorical(p_mixture),
                                                                        torch.distributions.Dirichlet(torch.from_numpy(rng.random(
                                                                            (n_dist, self._n_dimension))) * 10.0 + 1.0)))
            else:
                self._dist.append(torch.distributions.MixtureSameFamily(torch.distributions.Categorical(p_mixture),
                                                                        torch.distributions.Dirichlet(torch.from_numpy(rng.random(
                                                                            (n_dist, self._n_dimension))) * 1.0/self._n_dimension + torch.from_numpy(rng.random(n_dist,self._n_dimension)) * 0.2)))


    def forward(self, state, action):
        # if len(action.shape) == 1 and not (action.abs().sum().item() == 1.0):  # in case the action is not a simplex
        #     return self._dist[state].log_prob(action.abs()/action.abs().sum()).exp() #map to simplex

        if self._polytope is None:            
            return self._dist[state].log_prob(action).exp()
        
        if len(action.shape) == 1 and not np.isclose(action.sum().item(), 1.0):  # in case the action is not a simplex
            if False:
                return torch.tensor([-1.0]).repeat(len(action.shape)).view(-1, 1)
            else:
                return -torch.abs(action.sum(dim=-1) - 1.0)

        if len(action.shape) > 1:
            reward = self._dist[state].log_prob(action).exp()

            indices = self._polytope.contains(action.detach().numpy().T)
            if not indices.any():
                reward[~indices] = torch.sqrt(torch.sum((action[~indices] - torch.tensor(np.array(
                    [distance_to_polytope_surface(ac, self._polytope.A, self._polytope.b)[1] for ac in
                     action[~indices]]), dtype=torch.float32)) ** 2,
                                                        dim=-1)) * -5.0 - 10.0  # needs to be differentiable

            return reward
        else:

            if self._polytope.__contains__(action.detach().numpy()):
                reward = self._dist[state].log_prob(action).exp()
            else:
                reward = torch.sqrt(torch.sum((action - torch.tensor(
                    distance_to_polytope_surface(action.detach().numpy(), self._polytope.A, self._polytope.b)[1],
                    dtype=torch.float32)) ** 2)) * -5.0 - 10.0  # needs to be differentiable

            return reward



class SynthEnv(gymnasium.Env):

    def __init__(self, n_dimension=3, n_states=4, env_gen_seed=1, A=None, b=None, use_mlp_reward_function=True):

        self._n_dimension = n_dimension
        self._n_states = n_states

        self.action_space = gymnasium.spaces.Box(low=0, high=1, shape=(self._n_dimension,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(self._n_states + 1,),
                                                      dtype=np.float32)  # +1 for the terminal state


        if A is not None and b is not None:
            self.A = A
            self.b = b


        self.A = A
        self.b = b
        
        if self.A is not None:
            self.polytope = pc.Polytope(A, b, normalize=False)
        else:
            self.polytope = None

        if use_mlp_reward_function:
            self._reward_function = MLPRewardNetwork(self._n_dimension, self._n_states, self.polytope, env_gen_seed)
        else:
            self._reward_function = DistRewardNetwork(self._n_dimension, self._n_states, self.polytope, env_gen_seed)

    def _encode_state(self, state):
        encoded_state = np.zeros(self._n_states + 1, dtype=np.float32)
        encoded_state[state] = 1

        return encoded_state

    def reset(self):#,
              #*,
              #seed: int | None = None,
              #options: dict[str, Any] | None = None):
        self.state = 0
        return self._encode_state(self.state)

    def step(self, action):

        with torch.no_grad():
            reward = self._reward_function(self.state, torch.tensor(action, dtype=torch.float32)).item()

        self.state += 1

        terminated = self.state >= self._n_states

        return self._encode_state(self.state), reward, terminated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def _grid_search(self, state, step_size=0.01):
        #n_points = int(1 / step_size)
        n_points = 50000 #TODO do not hardcode 

        # Generate points on the simplex directly
        # points = [coords for coords in itertools.product(range(n_points + 1), repeat=self._n_dimension) if sum(coords) == n_points]
        uniform_points = []
        points_already_sampled = 0
        while points_already_sampled < n_points: 
            n_points_raw = 100000 #1000000
            points = torch.distributions.Dirichlet(torch.tensor([1.0]*self._n_dimension, dtype=torch.float32)).sample((n_points_raw,)).numpy()
            
            if self.polytope is not None:
                indices = self.polytope.contains(points.T)
                points = points[indices]
            
            uniform_points.append(points)
            actual_batch_size = len(points)
            points_already_sampled += actual_batch_size

        uniform_points = np.concatenate(uniform_points, axis=0)
        points = uniform_points[:n_points]

        # points = torch.distributions.Dirichlet(torch.tensor([1.0] * self._n_dimension)).sample(
        #     (1000000,)).numpy()
        # indices = self.polytope.contains(points.T)
        # points = points[indices]
        # points = np.array(points, dtype=np.float32) / n_points

        max_batch_size = 10000

        rewards = []

        points_calculated = 0
        while points_calculated < len(points):
            points_batch = torch.tensor(points[points_calculated:points_calculated + max_batch_size],
                                        dtype=torch.float32)

            reward = self._reward_function(state, points_batch)

            rewards.append(reward.detach().numpy())
            points_calculated += points_batch.shape[0]

        rewards = np.concatenate(rewards)

        return points, rewards

    def plot_reward_function(self):
        import matplotlib.pyplot as plt
        import ternary

        for state in range(self._n_states):
            encoded_state = self._encode_state(state)
            encoded_state = torch.tensor(encoded_state, dtype=torch.float32)

            step_size = 0.01

            points, rewards = self._grid_search(state, step_size)

            print(f"State {state} Max Reward {rewards.max()} at {points[rewards.argmax()]}")

            if self._n_dimension == 3:
                scale = 100
                figure, tax = ternary.figure(scale=scale)

                tax.boundary(linewidth=1)
                tax.gridlines(color="black", multiple=0.1 * scale)
                # tax.gridlines(color="blue", multiple=0.1*scale)

                offset = 0.15
                tax.left_axis_label("Dim 3", offset=offset)
                tax.right_axis_label("Dim 2", offset=offset)
                tax.bottom_axis_label("Dim 1", offset=offset)

                # tax.scatter(points, c=rewards, colormap="viridis")
                tax.heatmap(
                    {tuple(p[:-1]): rewards[i].item() for i, p in enumerate((points * (scale)).round().astype(int))},
                    scale=scale, style="triangular", cmap="viridis", colorbar=True, cbarlabel="Reward")
                tax.ticks(axis='lbr', multiple=0.1 * scale, tick_formats="%.1f", offset=0.05)

                # Remove default Matplotlib Axes
                tax.clear_matplotlib_ticks()
                ternary.plt.show()

            if self._n_dimension == 2:
                fig, ax = plt.subplots(1, 1)
                ax.scatter(points[:, 0], points[:, 1], c=rewards, cmap="viridis")
                ax.set_title(f"Reward function for state {state}")
                plt.show()


if __name__ == "__main__":
    import sys
    from utils.polytope_loader import load_polytope
    
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(8)
    np.random.seed(8)
    random.seed(7)
    
    n_dim = 3
    A, b = load_polytope(n_dim,"seed", "points", {"seed": 1, "n_dim": n_dim, "n_points": 30})
        
    env = SynthEnv(n_dimension=n_dim, n_states=2, env_gen_seed=1, A=A, b=b, use_mlp_reward_function=True)

    #env.plot_reward_function()
    obs = env.reset()
    
    #_, reward, _, _ = env.step(np.array([0.35181, 0.4246 , 0.22359], dtype=np.float32))

    for i in range(10):
        action = env.action_space.sample()

        print(f"Taking action {action}")

        obs, reward, terminated, _, info = env.step(action)

        print(f"Step {i}: {obs}, {reward}, {terminated}, {info}")

        if terminated:
            break