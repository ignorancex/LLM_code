import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, FLAG, project_path, base_dir=None):
        cfg_id = FLAG.cfg
        self.id = cfg_id
        self.project_path = project_path
        cfg_path = os.path.join(project_path, "design_opt", "cfg", f"{cfg_id}.yml")
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        cfg = yaml.safe_load(open(files[0], 'r'))
        # create dirs
        if base_dir is not None:
            self.base_dir = base_dir + "/results"
        else:
            self.base_dir = os.getcwd() + "/results"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.agent_specs = cfg.get('agent_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.policy_specs.update(FLAG.get('policy_specs', dict()))
        self.obs_specs = cfg.get('obs_specs', dict())
        self.obs_specs.update(FLAG.get('obs_specs', dict()))
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.eval_batch_size = cfg.get('eval_batch_size', 10000)
        self.seed_method = cfg.get('seed_method', 'deep')
        
        # training config (from global flag)
        self.lr_decay = FLAG.get('lr_decay', False)
        self.policy_optimizer = FLAG.get('policy_optimizer', 'Adam')
        self.policy_lr = FLAG.get('policy_lr', 5e-5)
        self.policy_momentum = FLAG.get('policy_momentum', 0.0)
        self.policy_weightdecay = FLAG.get('policy_weightdecay', 0.0)
        self.value_specs = FLAG.get('value_specs', dict())
        self.value_optimizer = FLAG.get('value_optimizer', 'Adam')
        self.value_lr = FLAG.get('value_lr', 3e-4)
        self.value_momentum = FLAG.get('value_momentum', 0.0)
        self.value_weightdecay = FLAG.get('value_weightdecay', 0.0)
        self.clip_epsilon = FLAG.get('clip_epsilon', 0.2)
        self.num_optim_epoch = FLAG.get('num_optim_epoch', 10)
        self.min_batch_size = int(FLAG.get('min_batch_size', 50000))
        self.mini_batch_size = int(FLAG.get('mini_batch_size', self.min_batch_size))
        self.max_epoch_num = int(FLAG.get('max_epoch_num', 1000))
        self.seed = FLAG.get('seed', 1)
        self.save_model_interval = FLAG.get('save_model_interval', 100)
        self.norm_advantage = FLAG.get('norm_advantage', False)
        self.max_grad_norm = FLAG.get('max_grad_norm', 40)
        self.uni_obs_norm = FLAG.get('uni_obs_norm', False)
        self.norm_return = FLAG.get('norm_return', True)
        self.reward_shift = FLAG.get('reward_shift', 0.0)
        self.xml_name = FLAG.get('xml_name', 'default')
        self.planner_demean = FLAG.get('planner_demean', False)
        
        self.enable_wandb = FLAG.get('enable_wandb', True)
        self.group = FLAG.get('group', 'group')

        # anneal parameters
        self.scheduled_params = cfg.get('scheduled_params', dict())
        
        self.skel_entropy_coef = FLAG.get('skel_entropy_coef', 0.0)
        self.attr_entropy_coef = FLAG.get('attr_entropy_coef', 0.0)
        self.control_entropy_coef = FLAG.get('control_entropy_coef', 0.0)

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.skel_transform_nsteps = cfg.get('skel_transform_nsteps', 5)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())


