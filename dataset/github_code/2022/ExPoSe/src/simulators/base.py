class BaseSimulator:
    def __init__(self):
        self.state = None

    def reset(self, idx=None, level=None):
        raise NotImplementedError

    def step(self, action):
        self.state, r, t = self.simulate(self.state, action)
        return self.state, r, t

    @staticmethod
    def simulate(state, action):
        raise NotImplementedError

    @staticmethod
    def render(state, as_image=True, searched_paths=()):
        raise NotImplementedError

    @staticmethod
    def load_levels_from(file):
        raise NotImplementedError

    @staticmethod
    def hash(state):
        raise NotImplementedError

    @staticmethod
    def is_solved(state):
        raise NotImplementedError

    @staticmethod
    def to_tensor(state):
        raise NotImplementedError

    @staticmethod
    def legal_actions(state):
        return

    @staticmethod
    def get_sample_tensor():
        raise NotImplementedError
