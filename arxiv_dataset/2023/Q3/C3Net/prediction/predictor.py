import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Predictor:
    def __init__(
        self, 
        model_path: str, 
        model: nn.Module, 
        loader: Dataset, 
        device: str
        ):

        self.model_path = model_path
        self.device = device
        self.checkpoint_path = model_path
        self.loader = loader
        self._model = model
        
        self.restore_checkpoint()
        
    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    @property
    def state_dict(self):
        state_dict = {
            
        }

        if self._check_is_parallel():
            state_dict['model'] = self._model.module.state_dict()
        else:
            state_dict['model'] = self._model.state_dict()
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self._load_model_state_dict(state_dict['model'])

    def restore_checkpoint(self):
        chkpt = os.path.join('./model', 'checkpoint-1.pth.tar')
        if self.device == 'cuda':
            self.state_dict = torch.load(chkpt)
        else:
            self.state_dict = torch.load(chkpt, map_location=torch.device('cpu'))
        


    def run(self, device):
        self._model.to(device)

        with open('%s/prediction.txt'%self.checkpoint_path,'w') as pre:
            pre.write('Solvent_ID\tSolute_ID\tPredicted\n')
            for data in self.loader:
                # move input to gpu, if needed
                batch = {}
                for k, v in data.items():
                    batch[k] = v.to(device) 
                
                result = self._model(batch)

                y = result['y'].tolist()
                #print (y)
                for i in range(len(y)):
                    pre.write('%s\t%s\t%5.3f\n'%(batch['Solvent_ID'][i].tolist()[0], batch['Solute_ID'][i].tolist()[0], y[i][0]))
                    #pre.write('%5.3f\n'%(y[i][0]))