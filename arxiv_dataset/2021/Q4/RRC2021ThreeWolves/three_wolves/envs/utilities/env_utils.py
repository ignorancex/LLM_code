from trifinger_simulation.visual_objects import CubeMarker
import pybullet
import numpy as np

U = []

def tag(xyz):
    U.append(CubeMarker(
        width=0.03,
        position=xyz,
        orientation=(0, 0, 0, 1),
        color=(1, 0, 0, 0.7),
        pybullet_client_id=0,
    ))

def clean():
    for o in U:
        pybullet.removeBody(o.body_id, physicsClientId=0)

def resetCamera():
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=0,
        cameraPitch=-41,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=0
    )

class HistoryWrapper:
    def __init__(self,
                 history_num=3):
        self._history_obs = {}
        self.history_num = history_num
        self.dt = {}

    def reset(self, init_obs_dict):
        for k, v in init_obs_dict.items():
            self._history_obs.update({k: [v]*self.history_num})
        return self.get_history_obs()

    def update(self, obs_dict):
        if not self.dt:
            self.dt = obs_dict
            return self.reset(obs_dict)

        self.dt = obs_dict
        for k, v in obs_dict.items():
            assert len(v) == len(self._history_obs[k][0]), 'wrong shape'
            assert k in self._history_obs.keys(), 'wrong key'
            self._history_obs[k].pop()
            self._history_obs[k].insert(0, v)
            assert len(self._history_obs[k]) == self.history_num

        return self.get_history_obs()

    def get_history_obs(self):
        _obs = []
        for _, v in self._history_obs.items():
            _obs.append(np.hstack(v))
        return np.hstack(_obs)

    def search(self, k):
        return np.array(self._history_obs[k])


if __name__ == '__main__':
    his = HistoryWrapper(3)
    his.reset({'pos': [1], 'orn': [6]})
    q = his.get_history_obs()