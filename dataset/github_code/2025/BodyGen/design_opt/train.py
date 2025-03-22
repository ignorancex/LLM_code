# import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

project_path = os.getcwd()

def main_loop(FLAGS):
    if FLAGS.render:
        FLAGS.num_threads = 1
    cfg = Config(FLAGS, project_path)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=FLAGS.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    start_epoch = int(FLAGS.epoch) if FLAGS.epoch.isnumeric() else FLAGS.epoch

    """create agent"""
    agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=FLAGS.num_threads, training=True, checkpoint=start_epoch)    

    if FLAGS.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not FLAGS.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):          
            agent.optimize(epoch)
            agent.save_checkpoint(epoch)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info('training done!')


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    FLAGS = cfg
    
    if FLAGS.enable_wandb:
        wandb.init(
            project=str(FLAGS.project),
            group=str(FLAGS.group),
            name="genesis"
        )
    
    main_loop(FLAGS)
    
    if FLAGS.enable_wandb:
        wandb.finish()
    
if __name__ == '__main__':
    main()
