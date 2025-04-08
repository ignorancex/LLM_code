import logging

import numpy as np
import torch
from torch.utils.data import DataLoader



def collate_molecule(examples):  
    max_size = {}
    for prop, val in examples[0].items():
        max_size[prop] = np.array(val.size(), dtype=np.int)
            
    # get maximum sizes within batch
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(max_size[prop], np.array(val.size(), dtype=np.int))

    # initialize batch
    batch = {}
    for prop, size in max_size.items():
        batch[prop] = torch.zeros(len(examples), *[int(ss) for ss in size]).type(examples[0][prop].type())
        
    # build batch and pad
    for k, properties in enumerate(examples):#k=batch number
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val
    
    atom_max_num = torch.max(batch['Atom_Grid_Num'])

    mask = torch.zeros(len(examples),atom_max_num)
    
    for k, properties in enumerate(examples):#k=batch number
        atom_num = batch['Atom_Num'][k]
        mask[k][:atom_num] = 1#atom
    
    batch['Mask'] = mask

    return batch


class MoleculeLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=collate_molecule, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        super(MoleculeLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)