import torch as t

from ..configs import SIMULATOR
from ..modules import NeuralNetwork, get_feature_network, get_policy_network, get_value_network
from ..modules.commons import Conv


class AbstractSolver(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.f_features_for_policy = get_feature_network(SIMULATOR)
        self.f_policy = get_policy_network(SIMULATOR)
        self.f_features_for_value = get_feature_network(SIMULATOR)
        self.f_value = get_value_network(SIMULATOR)

    def features_for_policy(self, state: t.FloatTensor):
        return self.f_features_for_policy(state).squeeze(1)  # squeeze for tensorrt

    def policy_network(self, state: t.FloatTensor, is_features=False):
        if not is_features:
            state = self.features_for_policy(state)
        return self.f_policy(state).squeeze(1)  # squeeze for tensorrt

    def features_for_value(self, state: t.FloatTensor):
        return self.f_features_for_value(state).squeeze(1)  # squeeze for tensorrt

    def value_network(self, state: t.FloatTensor, is_features=False):
        if not is_features:
            state = self.features_for_value(state)
        return self.f_value(state).squeeze(1)  # squeeze for tensorrt

    def forward(self, state: t.FloatTensor):
        return self.policy_network(state), self.value_network(state)

    def get_action(self, state):
        raise NotImplementedError

    def update(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def eval(self, trt=True):
        super(AbstractSolver, self).eval()
        for m in self.modules():
            if type(m) is Conv:
                m.fuse()

        if trt:
            try:
                from torch2trt import torch2trt
                sample_state = SIMULATOR.get_sample_tensor()
                self.f_policy = torch2trt(self.f_policy, [self.f_features_for_policy(sample_state)], use_onnx=True)
                self.f_features_for_policy = torch2trt(self.f_features_for_policy, [sample_state], use_onnx=True)
                self.f_value = torch2trt(self.f_value, [self.f_features_for_value(sample_state)], use_onnx=True)
                self.f_features_for_value = torch2trt(self.f_features_for_value, [sample_state], use_onnx=True)
            except ImportError as e:
                print("TensorRT not installed! Using Pytorch...")
                pass
