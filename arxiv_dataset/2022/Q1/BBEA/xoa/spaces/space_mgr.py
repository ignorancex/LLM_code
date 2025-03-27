import os
import time
import json
import copy
import sys
import random

from xoa.commons.logger import * 
from xoa.managers.proto import ManagerPrototype 
from xoa.spaces import create_space_from_table, create_search_space


class SearchSpaceManager(ManagerPrototype):

    def __init__(self, *args, **kwargs):
        super(SearchSpaceManager, self).__init__(type(self).__name__)
        self.spaces = {} 

    def create(self, hp_cfg_dict, space_spec):
        if "surrogate" in space_spec:
            surrogate = space_spec["surrogate"]
            grid_order = None

            if "grid_order" in space_spec:
                grid_order = space_spec["grid_order"]
            s = create_space_from_table(surrogate, grid_order)
            cfg = surrogate

        else:    
            if not "num_samples" in space_spec:
                space_spec["num_samples"] = 20000
            
            if not "seed" in space_spec:
                space_spec["seed"] = 1

            s = create_search_space(hp_cfg_dict, space_spec)
                    
        space_id = s.name
        space_obj = {"id" : space_id, "config": hp_cfg_dict, "space": s }
        space_obj["created"] = time.strftime('%Y%m%dT%H:%M:%SZ',time.gmtime())
        space_obj["status"] = "created"    
        
        self.spaces[space_id] = space_obj

        return space_id

    def get_available_spaces(self):
        return list(self.spaces.keys())

    def get_active_space_id(self):
        for s in self.spaces:
            if self.spaces[s]['status'] == "active":
                return s
        debug("No space is active now")
        return None                  

    def set_space_status(self, space_id, status):
        if space_id == "active":
            space_id = get_active_space_id()
                                
        elif space_id in self.spaces:
            self.spaces[space_id]['status'] = status
            self.spaces[space_id]['updated'] = time.strftime('%Y%m%dT%H:%M:%SZ',time.gmtime())
            return True
        else:
            debug("No such space {} existed".format(space_id))
            return False

    def get_space(self, space_id):
        if space_id == "active":
            for s in self.spaces:
                if self.spaces[s]['status'] == "active":
                    return self.spaces[s]['space']
            return None
        elif space_id in self.spaces:
            return self.spaces[space_id]['space']
        else:
            debug("No such {} space existed".format(space_id))
            return None

    def get_space_config(self, space_id):
        if space_id == "active":
            space_id = get_active_space_id()
                    
        if space_id in self.spaces:
            return self.spaces[space_id]['config']
        else:
            debug("No such space space {} existed".format(space_id))
            return None        


