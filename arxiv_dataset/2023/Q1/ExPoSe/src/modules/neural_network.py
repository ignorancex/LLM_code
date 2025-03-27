import os

import torch as t
from rich import print


class NeuralNetwork(t.nn.Module):
    def save(self, file: str):
        if file is None or file == '':
            print('File name empty!!')
            return
        folder = os.path.dirname(file)
        if not os.path.exists(folder):
            os.makedirs(folder)

        t.save(self.state_dict(), file)

    def load(self, file, strict=True, verbose=False):
        print(f'loading from file: "{file}" with strict matching: {strict}') if verbose else None
        if file is None or file == '':
            print('file name empty.') if verbose else None
            return False
        if not os.path.exists(file):
            print('file does not exist.') if verbose else None
            return False

        try:
            self.load_state_dict(t.load(file, map_location='cpu'), strict=strict)
            return True
        except RuntimeError as e:
            print(f'failed to load model.\nException: {e}')
            return False
