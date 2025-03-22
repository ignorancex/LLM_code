import importlib
import numpy as np

def init(self):
    self.done_mod = importlib.import_module(f'modules.agents_module.done.{self.done}')

def main(self):
    return self.done_mod.main(self)