from datetime import datetime
from typing import Any, Dict, Optional, Union
import os

import torch as th
import torch.optim as optim

import hion as hn
from hion.types import ConfigDict

"""
Author: Josue N Rivera
"""

def progress_bar(percentage, size=100, end=False, fill_char='|'):
    percentage = max(0.0, min(1.0, percentage)) # clamp to range [0.0, 1.0]
    percentage_int = int(percentage*100) # convert to integer percentage
    count = int(percentage*size) # determine number of characters to fill
    
    print(f"({percentage_int:3d}%)[{fill_char*count+' '*(size-count)}]", end="\n" if percentage == 1.0 or end else "\r")

class ControllerLogger():
    
    def __init__(self,
                 configuration:ConfigDict,
                 controller:hn.controller.Controller,
                 optimizer:optim.Optimizer,
                 checkpoint:Optional[dict] = None, **kargs) -> None:
        
        self.options = configuration['checkpoint']
        self.system:str = configuration['name']
        self.type = 'controller'

        self.controller = controller
        self.optimizer = optimizer

        if checkpoint is None:
            self.checkpoint = {
                '__child__': False,
                '__configuration__': configuration,
                '__training__': kargs,
                '__log__': {}
            } 
        else: 
            self.checkpoint = checkpoint
            self.checkpoint['__child__'] = True
            for key in self.checkpoint['__log__'].keys():
                self.checkpoint['__log__'][key] = [self.checkpoint['__log__'][key]]
            
            self.checkpoint['__configuration__'] = configuration

        dateinfo = datetime.now()
        self.print(f"Log started at: {dateinfo}")

    def log(self, key:str, item:Any):
        self.checkpoint['__log__'][key] = [item]

    def get_log(self, key:str):
        return self.checkpoint['__log__'][key][0]

    def start_ref_log(self, key:str) -> None:
        reference = [None]
        self.checkpoint['__log__'][key] = reference

        return reference

    def get_ref_log(self, key:str) -> None:
        return self.checkpoint['__log__'][key]

    def update_ref_log(self, reference, item:Any) -> None:
        reference[0] = item

    def print(self, txt:str) -> None:
        if not self.options['printless']:
            print(txt)

    def print_progress(self, progress:Union[float, int], loss:Optional[Dict[str, float]] = None, resolution:int = 100):

        """
        - loss argument needed if progress is a int
        - resolution needed if progress in a float
        """

        if isinstance(progress, float):
            count = int(progress*resolution)
            print(f"({progress*100:.2f}%)[{'|'*count+' '*(resolution-count)}]", end="\r" if progress != 1.0 else "\n")

        elif isinstance(progress, int):

            def dict_to_print(loss)->str:
                stats = []
                for key, value in loss.items():
                    if isinstance(value, dict):
                        stats.append('[' + dict_to_print(value) + ']')
                    else:
                        stats.append(f'{key}: {value:.5f}')

                return ' '.join(stats)

            self.print(f'[{progress + 1:5d}] ' + dict_to_print(loss))

    def close(self) -> None:

        dateinfo = datetime.now()
        filename = dateinfo.strftime(self.options['format']).replace('$name$', self.system).replace('$type$', self.type)

        self.print(f"Log closed at: {dateinfo}")

        for key in self.checkpoint['__log__'].keys():
            self.checkpoint['__log__'][key] = self.checkpoint['__log__'][key][0]

        self.checkpoint['__training__']['controller'] = self.controller.state_dict()
        self.checkpoint['__training__']['optimizer'] = self.optimizer.state_dict()

        os.makedirs(self.options['folder'], exist_ok=True)

        path = os.path.join(self.options['folder'], f"{filename}.checkpoint.pth")
        self.print(f"Checkpoint saved to \"{path}\"")

        th.save(self.checkpoint, path)
    
    def __del__(self) -> None:
        self.close()