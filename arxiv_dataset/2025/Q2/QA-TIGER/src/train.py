from __future__ import print_function
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())

import torch
import torch.nn as nn
import torch.distributed as dist

from src.utils import (
    arg_parse, seed_everything, setting, get_logger,
    set_logger, logging_config
)
from src.trainutils import (
    get_model, get_dloaders, get_optim, 
    train, evaluate, sync_processes, test
)

def main():
    args = arg_parse()
    cfg, device, cur_rank = setting(args)
    writer, timestamp = set_logger(cfg)
    logger = get_logger()
    save_dir = os.path.join(cfg.output_dir, timestamp)
    
    logging_config(cfg)
    seed_everything(cfg.seed)
    d_loaders = get_dloaders(cfg)
    model = get_model(cfg, device)
    optim, sched = get_optim(cfg, model, d_loaders['train'])
    
    best_acc = 0
    best_epoch = -1
    criterion = nn.CrossEntropyLoss()
    
    sync_processes()
    for epoch in range(1, cfg.epochs + 1):
        # training
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if writer is not None:
                for idx, param_group in enumerate(optim.param_groups):
                    current_lr = param_group['lr']
                    writer.add_scalar(f'train/lr', current_lr, epoch)
                
        logger.info(f"\n-------------- training epoch {epoch} --------------")
        train(cfg, epoch, device, d_loaders['train'], optim, criterion, model, writer)
        
        if (dist.is_initialized() and cur_rank == 0) or not dist.is_initialized():
            logger.info(f"\n-------------- validation epoch {epoch} --------------")
        
        # evaluation
        sync_processes()
        acc, loss = evaluate(cfg, epoch, device, d_loaders['val'], criterion, model, writer)
        
        # scheduling
        if cfg.hyper_params.sched.name == 'ReduceLROnPlateau':
            if cfg.hyper_params.sched.mode == 'max':
                sched.step(acc)
            elif cfg.hyper_params.sched.mode == 'min':
                sched.step(loss)
        else:
            # cosine annleaing
            sched.step(epoch)
        
        if acc >= best_acc and not cfg.debug:
            best_acc = acc
            best_epoch = epoch
            sd = model.module.state_dict()
            new_sd = {}
            for k, v in sd.items():
                if 'video_encoder' not in k:
                    new_sd[k] = v
            
            logger.info(f"best model saved at epoch {epoch} with acc {best_acc}")
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    torch.save(new_sd, os.path.join(save_dir, f'best.pt'))
            else:
                torch.save(new_sd, os.path.join(save_dir, f'best.pt'))
        
        logger.info(f"Epoch {epoch} done with {acc:3.2f} and loss {loss:.5f}.")
        logger.info(f"At epoch{best_epoch} best acc: {best_acc:3.2f}.")

    if not cfg.debug:
        logger.info(f"\nTesting with Best validation model... {cfg.data.test_annot}")
        cfg.mode = 'test'
        d_loaders = get_dloaders(cfg)['test']
        save_dir = Path(save_dir).absolute() 
        best_path = save_dir / f'best.pt'
        original_dict = torch.load(best_path.as_posix())
        update_dict = {}
        for name, param in original_dict.items():
            if hasattr(model, 'module'):
                name = 'module.' +name
            update_dict[name] = param
        model.load_state_dict(update_dict, strict=False)

        test(cfg, device, d_loaders, model)
        if isinstance(cfg.data.test_annots, (list, tuple)):
            for idx, test_annot in enumerate(cfg.data.test_annots):
                logger.info(f"\nTesting with Best validation model... {test_annot}")
                cfg.data.test_annot = test_annot
                d_loaders = get_dloaders(cfg)['test']
                test(cfg, device, d_loaders, model)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()