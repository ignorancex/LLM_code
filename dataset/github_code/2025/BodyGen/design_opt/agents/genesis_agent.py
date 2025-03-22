import math
import pickle
import time
from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from torch.utils.tensorboard import SummaryWriter
from design_opt.envs import env_dict
from design_opt.models.bodygen_policy import BodyGenPolicy
from design_opt.models.bodygen_critic import BodyGenValue
from design_opt.utils.logger import LoggerRLV1
from design_opt.utils.tools import TrajBatchDisc
import multiprocessing
from khrylib.rl.core.running_norm import RunningNorm
from torch.optim.lr_scheduler import LambdaLR

import wandb

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class BodyGenAgent(AgentPPO):

    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.setup_env()
        self.env.seed(seed)
        self.setup_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        if self.cfg.norm_return:
            self.design_ret_norm = RunningNorm(1, demean=self.cfg.planner_demean, clip=False)
            self.control_ret_norm = RunningNorm(1, demean=False, clip=False)
            self.ret_norm = RunningNorm(1, demean=False, clip=False)
        else:
            self.design_ret_norm = self.control_ret_norm = self.ret_norm = None
        if cfg.uni_obs_norm:
            self.obs_norm = RunningNorm(self.state_dim).to(self.device)
        else:
            self.obs_norm = None
        if checkpoint != 0:
            self.load_checkpoint(checkpoint)
        super().__init__(env=self.env, dtype=dtype, device=device, running_state=self.running_state,
                         custom_reward=None, logger_cls=LoggerRLV1, traj_cls=TrajBatchDisc, num_threads=num_threads,
                         policy_net=self.policy_net, value_net=self.value_net,
                         optimizer_policy=self.optimizer_policy, optimizer_value=self.optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)

    ## Setting Ups        
    def setup_env(self):
        env_class = env_dict[self.cfg.env_name]
        self.env = env = env_class(self.cfg, self)
        self.attr_fixed_dim = env.attr_fixed_dim
        self.attr_design_dim = env.attr_design_dim
        self.sim_obs_dim = env.sim_obs_dim
        self.state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.control_action_dim = env.control_action_dim
        self.skel_num_action = env.skel_num_action
        self.action_dim = self.control_action_dim + self.attr_design_dim
        self.running_state = None
        
    def setup_logger(self):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'), file_handle=True)
        self.best_rewards = -1000.0
        self.save_best_flag = False
        
    def setup_policy(self):
        cfg = self.cfg
        self.policy_net = BodyGenPolicy(cfg.policy_specs, self)
        to_device(self.device, self.policy_net)
        
    def setup_value(self):
        cfg = self.cfg
        self.value_net = BodyGenValue(cfg.value_specs, self)
        to_device(self.device, self.value_net)
        
    def setup_optimizer(self):
        cfg = self.cfg
        # policy optimizer
        if cfg.policy_optimizer == 'Adam':
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
        else:
            self.optimizer_policy = torch.optim.SGD(self.policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
        # value optimizer
        if cfg.value_optimizer == 'Adam':
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
        else:
            self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
        
        # learning rate decay
        if self.cfg.lr_decay:
            self.scheduler_policy = LambdaLR(self.optimizer_policy, lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
            self.scheduler_value = LambdaLR(self.optimizer_value, lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
        else:
            self.scheduler_policy = None
            self.scheduler_value = None

    ## Sampling
    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads
        t_start = time.time()

        to_test(*self.sample_modules)
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads)) ## 共同采样 min_batch_size
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)

                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    ## Per worker sampling
    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        ## make seed for the worker
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            if hasattr(self.env, 'np_random'):
                self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
        
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < min_batch_size:
            state = self.env.reset()
            logger.start_episode(self.env)
        
            while True:
                state_var = tensorfy([state])
                
                ## do obs norm (none-updated)
                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)

                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64)
                next_state, env_reward, termination, truncation, info = self.env.step(action)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, env_reward, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                    
                if info['stage'] == 'execution':
                    reward += self.cfg.reward_shift 
                
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                done = (termination or truncation)
                exp = 1 - use_mean_action
                
                memory.push(state, action, termination, done, next_state, reward, exp)

                if done:
                    break
                state = next_state
                
            logger.end_episode(self.env)        
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def optimize(self, epoch):
        info = self.optimize_policy(epoch)
        if self.scheduler_policy is not None:
            self.scheduler_policy.step()
        if self.scheduler_value is not None:
            self.scheduler_value.step()
        self.log_optimize_policy(epoch, info)

    def optimize_policy(self, epoch):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batch, log = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()

        """evaluate policy"""
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time() 

        info = {
            'log': log, 'log_eval': log_eval, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0
        }
        return info
    
    def estimate_advantages(self, states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types):
        design_masks = (state_types!=2).bool().to(self.device)
        control_masks = (state_types==2).bool().to(self.device)
        next_design_masks = (next_state_types!=2).bool().to(self.device)
        next_control_masks = (next_state_types==2).bool().to(self.device)
        
        self.design_ret_norm.to(self.device)
        self.control_ret_norm.to(self.device)
        
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = self.value_net(states_i)
                    current_range = np.arange(i, min(i + chunk, len(states)))
                    if self.design_ret_norm is not None:
                        local_design_masks = design_masks[current_range]
                        values_i[local_design_masks] = self.design_ret_norm.unscale(values_i[local_design_masks])
                    if self.control_ret_norm is not None:
                        local_control_masks = control_masks[current_range]
                        values_i[local_control_masks] = self.control_ret_norm.unscale(values_i[local_control_masks])
                        
                    values.append(values_i)
                values = torch.cat(values)
                
                next_values = torch.zeros_like(values)
                next_values[:-1] = values[1:]
                
                indices = torch.where(next_dones)[0]
                compute_next_states = [next_states[i] for i in indices]
                if compute_next_states:
                    computed_values = self.value_net(compute_next_states)
                    if self.design_ret_norm is not None:
                        local_next_design_masks = next_design_masks[indices]
                        computed_values[local_next_design_masks] = self.design_ret_norm.unscale(computed_values[local_next_design_masks])
                    if self.control_ret_norm is not None:
                        local_next_control_masks = next_control_masks[indices]
                        computed_values[local_next_control_masks] = self.control_ret_norm.unscale(computed_values[local_next_control_masks])
                        
                    next_values[indices] = computed_values
                        
        self.design_ret_norm.to('cpu')
        self.control_ret_norm.to('cpu')
        
        device = rewards.device
        rewards, next_terminations, next_dones, values, next_values = batch_to(torch.device('cpu'), rewards, next_terminations, next_dones, values, next_values)
        design_masks, control_masks = batch_to(torch.device('cpu'), design_masks, control_masks)
        
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        design_returns = tensor_type(rewards.size(0), 1)

        next_advantage = 0
        next_design_return = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + self.gamma * next_values[i] * (1 - next_terminations[i]) - values[i]
            advantages[i] = next_advantage = deltas[i] + self.gamma * self.tau * next_advantage * (1 - next_dones[i])
            design_returns[i] = next_design_return = rewards[i] + next_design_return * (1 - next_dones[i])

        design_advantages = design_returns - values
        returns = values + advantages
        
        if self.design_ret_norm is not None:
            returns[design_masks] = self.design_ret_norm(design_returns[design_masks])
        else:
            returns[design_masks] = design_returns[design_masks]
            
        if self.control_ret_norm is not None:
            returns[control_masks] = self.control_ret_norm(returns[control_masks])
        
        if self.cfg.norm_advantage:
            advantages[design_masks] = (design_advantages[design_masks] - design_advantages[design_masks].mean()) / design_advantages[design_masks].std()
            advantages[control_masks] = (advantages[control_masks] - advantages[control_masks].mean()) / advantages[control_masks].std()

        advantages, returns = batch_to(device, advantages, returns)
        return advantages, returns

    def update_params(self, batch):
        t0 = time.time()
        
        to_train(*self.update_modules)
        
        states = tensorfy(batch.states, self.device)
        next_states = tensorfy(batch.next_states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        next_terminations = torch.from_numpy(batch.next_terminations).to(self.dtype).to(self.device)
        next_dones = torch.from_numpy(batch.next_dones).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.to(self.device)
            self.obs_norm.train()
            states = self.normalize_observation(states)
            self.obs_norm.eval()
            next_states = self.normalize_observation(next_states)

        self.update_policy(states, next_states, rewards, next_terminations, next_dones, actions, exps)

        return time.time() - t0
    
    def normalize_observation(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE = zip(*x)
        obs_cat = torch.cat(obs)
        obs_norm = self.obs_norm(obs_cat)
        indices = np.cumsum(num_nodes)
        obs_split = [obs_norm[start:end] for start, end in zip([0] + list(indices[:-1]), indices)]
        x = [list(item) for item in zip(obs_split, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE)]
        return x

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, LongTensor(perm).to(self.device)

    def update_policy(self, states, next_states, rewards, next_terminations, next_dones, actions, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(states_i, actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)
        
        state_types = torch.tensor([item[2] for item in states], dtype=int) # [0, 1, 2] for ['skel_trans', 'attr_trans', 'execution']
        next_state_types = torch.tensor([item[2] for item in next_states], dtype=int)
        
        advantages, returns = self.estimate_advantages(states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types)

        for _ in range(self.opt_num_epochs):
                        
            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = LongTensor(perm_np).to(self.device)

                rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                    index_select_list(states, perm_np), index_select_list(actions, perm_np), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                if self.cfg.agent_specs.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(rnd_states)
                    rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                        index_select_list(rnd_states, perm_design_np), index_select_list(rnd_actions, perm_design_np), rnd_returns[perm_design].clone(), rnd_advantages[perm_design].clone(), \
                        rnd_fixed_log_probs[perm_design].clone(), rnd_exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        rnd_states[ind], rnd_actions[ind], rnd_advantages[ind], rnd_returns[ind], rnd_fixed_log_probs[ind], rnd_exps[ind]
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                self.optimizer_policy.step()

    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer_value.step()

    def ppo_loss(self, states, actions, advantages, fixed_log_probs):
        log_probs = self.policy_net.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        advantages = advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = - torch.min(surr1, surr2).mean()
        return surr_loss
                            
    def load_checkpoint(self, checkpoint):
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/epoch_%04d.p' % (cfg.model_dir, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        
        new_policy_dict = {}
        for key, value in model_cp['policy_dict'].items():
            new_key = key
            if ".lin_r." in new_key:
                new_key = new_key.replace(".lin_r.", ".lin_root.")
            if  ".lin_l." in new_key:
                new_key = new_key.replace(".lin_l.", ".lin_rel.")
            new_policy_dict[new_key] = value

        new_value_dict = {}
        for key, value in model_cp['value_dict'].items():
            new_key = key
            if ".lin_r." in new_key:
                new_key = new_key.replace(".lin_r.", ".lin_root.")
            if  ".lin_l." in new_key:
                new_key = new_key.replace(".lin_l.", ".lin_rel.")
            new_value_dict[new_key] = value
        
        self.policy_net.load_state_dict(new_policy_dict)
        self.value_net.load_state_dict(new_value_dict)
        self.loss_iter = model_cp['loss_iter']
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        self.obs_norm.load_state_dict(model_cp['obs_norm'])
    
    def save_checkpoint(self, epoch):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {
                            'obs_norm': self.obs_norm.state_dict() if self.obs_norm is not None else None,
                            'policy_dict': self.policy_net.state_dict(),
                            'value_dict': self.value_net.state_dict(),
                            'loss_iter': self.loss_iter,
                            'best_rewards': self.best_rewards,
                            'epoch': epoch}
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = self.cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save('%s/epoch_%04d.p' % (cfg.model_dir, epoch + 1))
        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('%s/best.p' % cfg.model_dir)

    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\tT_eval {info["T_eval"]:.2f}\t'\
            f'ETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\ttrain_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'\
            f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t{cfg.id}'
        logger.info(log_str)

        if log_eval.avg_exec_episode_reward > self.best_rewards:
            self.best_rewards = log_eval.avg_exec_episode_reward
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train_R_avg ', log.avg_reward, epoch)
        tb_logger.add_scalar('policy_learning_rate', self.optimizer_policy.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('value_learning_rate', self.optimizer_value.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)
        tb_logger.add_scalar('reward_shift', self.cfg.reward_shift, epoch)
        
        if self.cfg.enable_wandb:
            wandb.log({
                'train_R_avg': log.avg_reward,
                'policy_learning_rate': self.optimizer_policy.param_groups[0]["lr"],
                'value_learning_rate': self.optimizer_value.param_groups[0]["lr"],
                'train_R_eps_avg': log.avg_episode_reward,
                'eval_R_eps_avg': log_eval.avg_episode_reward,
                'exec_R_avg': log_eval.avg_exec_reward,
                'exec_R_eps_avg': log_eval.avg_exec_episode_reward,
                'reward_shift': self.cfg.reward_shift 
            }, step = epoch * self.cfg.min_batch_size)

    def visualize_agent(self, num_episode=1, mean_action=True, save_video=False, pause_design=False, max_num_frames=1000):
        fr = 0
        env = self.env
        paused = not save_video and pause_design
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        
        for _ in range(num_episode):
            state = env.reset()

            env._get_viewer('human')._paused = paused
            env.render()
            for t in range(10000):
                state_var = tensorfy([state])
                
                ## do obs norm (none-updated)
                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)
                        
                with torch.no_grad():
                    action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)
                next_state, env_reward, termination, truncation, info = env.step(action)
                done = (termination or truncation)
                
                if t < self.cfg.skel_transform_nsteps + 1:
                    env._get_viewer('human')._paused = paused
                    env._get_viewer('human')._hide_overlay = save_video
                for _ in range(15 if save_video else 1):
                    env.render()
                if save_video:
                    frame_dir = f'out/videos/{self.cfg.id}_frames'
                    os.makedirs(frame_dir, exist_ok=True)
                    save_screen_shots(env.viewer.window, f'{frame_dir}/%04d.png' % fr)
                    fr += 1
                    if fr >= max_num_frames:
                        break

                if done:
                    break
                state = next_state

            if save_video and fr >= max_num_frames:
                break

        if save_video:
            save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/videos/{self.cfg.id}.mp4', fps=30)
            shutil.rmtree(frame_dir)