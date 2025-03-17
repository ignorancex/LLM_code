import math
import os
from datetime import datetime

import git
import wandb
from pathlib import Path
import json

def get_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

class FormattedLogItem:
    def __init__(self, item, fmt):
        self.item = item
        self.fmt = fmt
    def __str__(self):
        return self.fmt.format(self.item)

def rm_format(dict_obj):
    ret = dict_obj
    for key in ret:
        if isinstance(ret[key], FormattedLogItem):
            ret[key] = ret[key].item
    return ret

def get_log_str(log_info, title='Expt Log', sep_ch='-', default_float_fmt='{:.4f}'):
    """
    Generates a formatted log string from the log_info dictionary
    """
    now = str(datetime.now().strftime('%H:%M %d-%m-%Y'))
    log_str = (sep_ch * math.ceil((80 - len(title))/2.) + title
               + (sep_ch * ((80 - len(title))//2)) + '\n')
    log_str += '{:<25} : {}\n'.format('Time', now)
    for key in log_info:
        if isinstance(log_info[key], float):
            log_str += '{:<25} : {}\n'.format(key, default_float_fmt.format(log_info[key]))
        else:
            log_str += '{:<25} : {}\n'.format(key, log_info[key])
    log_str += sep_ch * 80
    return log_str

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WandbWrapper():
    """
    Wrapper around wandb to handle stuff like logging to disk, init retries, etc.
    """
    def __init__(
        self, debug=False, silent=False, 
        write_to_disk=True, ignore_globs=['*.pth.tar']):
        self.debug = debug
        self.num_tries = 10
        self.write_to_disk = write_to_disk
        if debug and not silent:
            print('Wandb Wrapper : debug mode. No logging with wandb')
        
        os.environ['WANDB_IGNORE_GLOBS'] = ','.join(ignore_globs)

    def init(self, *args, **kwargs):
        self.log_file = Path(kwargs['dir']) / 'log.json'
        if not isinstance(kwargs['config'], dict):
            save_config = vars(kwargs['config'])
        else:
            save_config = kwargs['config']
            
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.log_data = json.load(f)
            if 'resume' in save_config:
                self.log_data['resume'] = save_config['resume']
            if self.log_data['config'] != save_config:
                self.log_data = {'config' : save_config, 'history': []}
        else:
            self.log_data = {'config' : save_config, 'history': []}
        self.num_logs = 0
        if not self.debug:
            init_tries = 0
            while True:
                try:
                    wandb.init(*args, **kwargs)
                    break
                except Exception as e:
                    print('[Trial:{}] wandb could not init : {}'.format(init_tries, e))
                    if init_tries > self.num_tries:
                        wandb.alert(
                            'Expt name \'{}\' : Could not init in {} attempts'.format(
                                kwargs['name'], self.num_tries))
                    init_tries += 1
            self.run = wandb.run
        else:
            self.run = AttrDict({'dir' : kwargs['dir']})
        self.commit_to_disk()

    def commit_to_disk(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log(self, *args, **kwargs):
        if 'commit' in kwargs and kwargs['commit'] == False:
            self.log_data['history'][-1].update(args[0])
        else:
            self.log_data['history'].append(args[0])
        if not self.debug:
            wandb.log(*args, **kwargs)
        self.num_logs += 1

        if self.write_to_disk and self.num_logs%10 == 0:
            self.commit_to_disk()

    def join(self, *args, **kwargs):
        if self.write_to_disk:
            self.commit_to_disk()
        if not self.debug:
            wandb.join(*args, **kwargs)

        