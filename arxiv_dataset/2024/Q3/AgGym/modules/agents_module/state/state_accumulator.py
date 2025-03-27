import importlib
import numpy as np

def init(self):
    self.state_mod = importlib.import_module(f'modules.agents_module.state.{self.state}')

def main(self):
    return self.state_mod.main(self)
