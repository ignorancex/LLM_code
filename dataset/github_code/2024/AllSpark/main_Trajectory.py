import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.backends import cudnn
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader

from models import builder
from models.llm.modeling_llama import _expand_mask, _make_causal_mask
from models.modal_tokenizer import TrajTokenizer, TrajectoryEnc
from models.task_head import TrajHead
from models.utils import get_text_embed
from mst_datasets import TrajectoryDataset
from utils import (Avg_values, MST_Logger, adjust_learning_rate,
                   get_motion_modes)


def get_args_parser():
    parser = argparse.ArgumentParser('AllSpark for Trajectory', add_help=False)
    
    # Common parameters
    parser.add_argument('--cfg', default="configs/Trajectory_ETH_test.yaml", type=str, help='config file')
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=2333, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--parallel-split-layer', default=[13], type=int, nargs="+")
    parser.add_argument('--port', default=23600, type=int)

    # Dataset parameters
    parser.add_argument('--fix-prompt', action='store_true', help='whether fix prompt to id 0, default False (eg, random)')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--pipe-chunks', default=1, type=int)

    return parser.parse_args()

def main(args, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

    mstLogger = MST_Logger(args.output_dir)
    mstLogger.logger.info(args)
    mstLogger.logger.info(cfg)
    
    if args.seed is not None:
        mstLogger.logger.info(f"set seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True
        
    args.parallel_split_layer.append(32)
    parallel_list = args.parallel_split_layer
    assert len(parallel_list) <= torch.cuda.device_count()
    mstLogger.logger.info(f"use {len(parallel_list)} GPUs")

    # build dataset
    train_dataset = TrajectoryDataset(cfg['root_path'], dataset_name='eth', dataset_type='train', 
                                                    translation=True, rotation=True, scaling=True, obs_len=cfg['obs_len'], 
                                                    dist_threshold=cfg['dist_threshold'], smooth=False)
    test_dataset = TrajectoryDataset(cfg['root_path'], dataset_name='eth', dataset_type='test', 
                                                    translation=True, rotation=True, scaling=False, obs_len=cfg['obs_len'])
    
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.coll_fn, batch_size=args.batch_size, 
                              shuffle=True, num_workers=int(args.num_workers))
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, batch_size=args.batch_size, 
                             shuffle=True, num_workers=int(args.num_workers))
    
    # build model
    modal_tokenizer = TrajTokenizer(obs_len=cfg['obs_len'], pred_len=cfg['pred_len'], embed_size=cfg['embed_size'])
    if cfg['modal_tokenizer_checkpoint'] or args.eval:
        print(f"### Load modal_tokenizer params")
        state_dict = torch.load(cfg['modal_tokenizer_checkpoint'], map_location='cpu')
        modal_tokenizer.load_state_dict(state_dict, strict=True)
    modal_backbone = TrajectoryEnc.TrajEncoder(cfg['embed_size'], num_layers=12, heads=16, forward_expansion=2, islinear=True)
    if cfg['modal_backbone_checkpoint'] or args.eval:
        print(f"### Load modal_backbone params")
        state_dict = torch.load(cfg['modal_backbone_checkpoint'], map_location='cpu')
        modal_backbone.load_state_dict(state_dict, strict=True)
    bridge = builder.build_bridge(cfg['bridge'])
    task_head = TrajHead(embed_size=cfg['embed_size'], obs_len=cfg['obs_len'], int_num_layers_list=[1,1], pred_len=cfg['pred_len'])
    if cfg['task_head_checkpoint'] or args.eval:
        print(f"### Load task_head params")
        state_dict = torch.load(cfg['task_head_checkpoint'], map_location='cpu')
        task_head.load_state_dict(state_dict, strict=True)
    llm_tokenizer, llm_backbone = builder.build_LLM(cfg)
    model = build_ModelwLLM_Pipe(modal_tokenizer, modal_backbone, bridge, llm_backbone, task_head, cfg['embed_size'], parallel_list)
    model = Pipe(model, chunks=args.pipe_chunks)

    prompts = [
        "Based on their past positions and movements in a crowded environment, predict the future trajectory of a selected pedestrian",
        "Using the pedestrian trajectory data, along with additional information about the surrounding environment, predict the future path of the pedestrian",
        "Given the current and past positions of a pedestrian and their neighboring pedestrians, predict the main pedestrian's trajectory"
    ]
    text_embeds, input_atts_pads = [], []
    with torch.no_grad():
        for prompt in prompts:
            text_embed, input_atts_pad = get_text_embed(prompt, args.batch_size, llm_tokenizer, llm_backbone)
            text_embeds.append(text_embed)
            input_atts_pads.append(input_atts_pad)
    
    if args.eval:
        print(f"### Load motion from {cfg['motion_path']}")
        motion_modes_file = cfg['motion_path']
    else:
        motion_modes_file = os.path.join(cfg['root_path'], 'eth_motion_modes.pkl')
    if not os.path.exists(motion_modes_file):
        print('motionm modes generating ... ')
        motion_modes = get_motion_modes(train_dataset, cfg['obs_len'], cfg['pred_len'], cfg['n_clusters'], cfg['root_path'], 'eth',
                                        smooth_size=cfg['smooth_size'], random_rotation=cfg['random_rotation'], traj_seg=cfg['traj_seg'])
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to('cuda:0')
    
    if os.path.exists(motion_modes_file):
        print('motion modes loading ... ')
        import pickle
        f = open(motion_modes_file, 'rb+')
        motion_modes = pickle.load(f)
        f.close()
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to('cuda:0')
    
    if args.eval:
        ade = evaluation(model, test_loader, motion_modes, args, eva_text_embed=text_embeds[0], 
                            eva_input_atts_pad=input_atts_pads[0], mstLogger=mstLogger)
    else:
        # optimizer
        optimizer_cfg = cfg['optimizer']
        optimizer = torch.optim.AdamW(model.parameters(), optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'],
                                    eps=optimizer_cfg['adam_epsilon'])
        
        reg_criterion = nn.SmoothL1Loss()
        cls_criterion = nn.CrossEntropyLoss()
        
        # summary params
        freeze_param, trainable_param = 0, 0
        freeze_name, trainable_name = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_name.append(n)
                trainable_param += p.numel()
            else:
                freeze_name.append(n)
                freeze_param += p.numel()
                
        mstLogger.logger.info("-" * 40)    
        mstLogger.logger.info(f"### Trainable Params")
        for n in trainable_name:
            mstLogger.logger.info(f"### {n}")
        mstLogger.logger.info("-" * 40)       
        mstLogger.logger.info(f"### Freeze Params")
        for n in freeze_name:
            mstLogger.logger.info(f"### {n}")
        mstLogger.logger.info("-" * 40)  
        mstLogger.logger.info(f"### Total Params: {(freeze_param + trainable_param) / 1e6:.2f}M")
        mstLogger.logger.info(f"### Freeze Params: {freeze_param / 1e6:.2f}M")
        mstLogger.logger.info(f"### Trainable Params: {trainable_param / 1e6:.2f}M")
        mstLogger.logger.info("-" * 40) 
        
        cur_iters = 0
        iters_per_epoch = len(train_loader)
        max_epoch = int(optimizer_cfg['max_epochs'])
        max_iters = max_epoch * iters_per_epoch
        warmup_iters = int(optimizer_cfg['warmup_epochs']) * iters_per_epoch
        print_loss = 0.
        run_time = Avg_values()
        min_ade = 99999.
        for cur_epoch in range(max_epoch):
            model.train()
            for i, (ped, neis, mask) in enumerate(train_loader):
                model[-1].task_head.set_train()
                
                start_time = datetime.now()
                
                cur_iters += 1
                lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, optimizer_cfg)

                # random choose prompt
                if args.fix_prompt:
                    choose_idx = 0
                else:
                    choose_idx = random.randint(0, len(text_embeds)-1)
                cur_text_embed = text_embeds[choose_idx][0:ped.size(0)].to('cuda:0')
                cur_input_atts_pad = input_atts_pads[choose_idx][0:ped.size(0)].to('cuda:0')
                assert cur_text_embed.size(0) == ped.size(0)
                
                ped = ped.to('cuda:0')
                neis = neis.to('cuda:0')
                mask = mask.to('cuda:0')

                ped[:, :, 0] = ped[:, :, 0] * cfg['data_scaling'][0]
                ped[:, :, 1] = ped[:, :, 1] * cfg['data_scaling'][1]
                
                scale = torch.randn(ped.shape[0])*0.05+1
                scale = scale.to('cuda:0')
                scale = scale.reshape(ped.shape[0], 1, 1)
                ped = ped * scale
                scale = scale.reshape(ped.shape[0], 1, 1, 1)
                neis = neis * scale

                ped_obs = ped[:, :cfg['obs_len']]
                gt = ped[:, cfg['obs_len']:]
                neis_obs = neis[:, :, :cfg['obs_len']]
                
                with torch.no_grad():
                    soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)
                
                pred_traj, scores = model(ped_obs, motion_modes, cur_text_embed, cur_input_atts_pad, 
                                        closest_mode_indices, neis_obs, mask).to_here()
                
                reg_label = gt.reshape(pred_traj.shape)
                reg_loss = reg_criterion(pred_traj, reg_label.to(pred_traj.device)) 
                clf_loss = cls_criterion(scores.squeeze(), soft_label.to(scores.device)) 
                loss = reg_loss + clf_loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 
                    
                print_loss += loss.item()
                
                end_time = datetime.now()
                run_time.update(end_time-start_time, 1)
                eta_str = str(run_time.avg * (max_iters - run_time.count))
                
                if cur_iters % args.print_freq == 0:
                    mstLogger.logger.info(f"epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                    f"total_iters:{cur_iters} lr:{lr:.6e} loss:{print_loss / args.print_freq:.6f} eta:{eta_str}")
                    print_loss = 0.
                
            if (cur_epoch+1) % optimizer_cfg['eval_epochs'] == 0 or cur_epoch == max_epoch-1:
                ade = evaluation(model, test_loader, motion_modes, args, eva_text_embed=text_embeds[0], 
                            eva_input_atts_pad=input_atts_pads[0], mstLogger=mstLogger)
                if ade < min_ade:
                    min_ade = ade
                    print(f"min_ade: {min_ade}")    
                    print(f"### Save params")
                    torch.save(modal_tokenizer.state_dict(), os.path.join(args.output_dir, "modal_tokenizer.pth"))
                    torch.save(modal_backbone.state_dict(), os.path.join(args.output_dir, "modal_backbone.pth"))
                    torch.save(bridge.state_dict(), os.path.join(args.output_dir, "bridge.pth"))
                    torch.save(task_head.state_dict(), os.path.join(args.output_dir, "task_head.pth"))
                    llm_backbone.save_pretrained(os.path.join(args.output_dir, "llm"))
                    llm_tokenizer.save_pretrained(os.path.join(args.output_dir, "llm"))
                    save_motion_modes_file = os.path.join(args.output_dir, "eth_motion_modes.pkl")
                    f = open(save_motion_modes_file, 'wb')
                    pickle.dump(motion_modes, f)
                    f.close()
        
    print("Done!")
    

@torch.no_grad()
def evaluation(model, test_loader, motion_modes, args, eva_text_embed, eva_input_atts_pad, mstLogger):
    model.eval()
    model[-1].task_head.set_test()
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    
    ade = 0
    fde = 0
    num_traj = 0
    for i, (ped, neis, mask) in enumerate(test_loader):
        start_time = datetime.now()
        
        cur_text_embed = eva_text_embed[0:ped.size(0)].to('cuda:0')
        cur_input_atts_pad = eva_input_atts_pad[0:ped.size(0)].to('cuda:0')
        assert cur_text_embed.size(0) == ped.size(0)
        
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda() 

        ped_obs = ped[:, :cfg['obs_len']]
        gt = ped[:, cfg['obs_len']:]
        neis_obs = neis[:, :, :cfg['obs_len']]
        
        num_traj += ped_obs.shape[0]
        pred_trajs, scores = model(ped_obs, motion_modes, cur_text_embed, cur_input_atts_pad, 
                                      None, neis_obs, mask).to_here()
        pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
        gt_ = gt.unsqueeze(1)
        norm_ = torch.norm(pred_trajs - gt_.to(pred_trajs.device), p=2, dim=-1)
        ade_ = torch.mean(norm_, dim=-1)
        fde_ = norm_[:, :, -1]
        min_ade, min_ade_index = torch.min(ade_, dim=-1)
        min_fde, min_fde_index = torch.min(fde_, dim=-1)
        min_ade = torch.sum(min_ade)
        min_fde = torch.sum(min_fde)
        ade += min_ade.item()
        fde += min_fde.item()
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            mstLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")
    
    ade = ade / num_traj
    fde = fde / num_traj
    mstLogger.logger.info("-" * 40)
    mstLogger.logger.info(f"ADE: {ade}")
    mstLogger.logger.info(f"FDE: {fde}")
    mstLogger.logger.info("-" * 40)

    return ade


class BeginBlock(nn.Module):
    def __init__(self, modal_tokenizer, modal_backbone, bridge, llm_layers):
        super().__init__()
        self.modal_tokenizer = modal_tokenizer
        self.modal_backbone = modal_backbone
        self.bridge = bridge
        self.llm_layers = llm_layers
        
    def forward(self, ped_obs, motion_modes, text_embed, input_atts_pad, closest_mode_indices, neis_obs, mask):
        
        src_embed = self.modal_backbone(self.modal_tokenizer(ped_obs, motion_modes))
        src2t_embed, src2t_atts = self.bridge(src_embed)
        
        inputs_embeds = torch.cat([src2t_embed, text_embed], dim=1)
        attention_mask = torch.cat([src2t_atts, input_atts_pad], dim=1)
        
        batch_size, seq_length, _ = inputs_embeds.shape
        
        past_key_values_length = 0
        
        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states, attention_mask, position_ids, closest_mode_indices, neis_obs, mask
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    

class CoreBlock(nn.Module):
    def __init__(self, llm_layers):
        super().__init__()
        self.llm_layers = llm_layers
        
    def forward(self, hidden_states, attention_mask, position_ids, closest_mode_indices, neis_obs, mask):
        
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states, attention_mask, position_ids, closest_mode_indices, neis_obs, mask
    
    
class EndBlock(nn.Module):
    def __init__(self, llm_layers, norm, task_head, num_latents, embed_size):
        super().__init__()
        self.llm_layers = llm_layers
        self.norm = norm
        self.task_head = task_head
        self.num_latents = num_latents
        
    def forward(self, hidden_states, attention_mask, position_ids, closest_mode_indices, neis_obs, mask):
        
        for idx, decoder_layer in enumerate(self.llm_layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )
            hidden_states = layer_outputs[0]
        
        ped_feat = hidden_states[:, 0:self.num_latents]
        
        pred, scores = self.task_head(ped_feat, closest_mode_indices, neis_obs, mask)
        return pred, scores
    

def build_ModelwLLM_Pipe(modal_tokenizer, modal_backbone, bridge, llm_backbone, 
                         task_head, embed_size, parallel_list):
    beginBlock = BeginBlock(modal_tokenizer, modal_backbone, bridge, 
                            llm_backbone.model.layers[0:parallel_list[0]]).to("cuda:0")
    coreBlocks = []
    core_num = len(parallel_list) - 2
    for i in range(core_num):
        coreBlocks.append(CoreBlock(
            llm_backbone.model.layers[parallel_list[i]:parallel_list[i+1]]
            ).to(f"cuda:{i+1}"))
    endBlock = EndBlock(llm_backbone.model.layers[parallel_list[-2]:], llm_backbone.model.norm,
                        task_head, bridge.num_latents, embed_size).to(f"cuda:{core_num+1}")
    return nn.Sequential(
        beginBlock,
        *coreBlocks,
        endBlock
    )


def get_cls_label(gt, motion_modes, soft_label=True):

    # motion_modes [K pred_len 2]
    # gt [B pred_len 2]

    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)  # [B 1 pred_len*2]
    motion_modes = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1 K pred_len*2]
    distance = torch.norm(gt - motion_modes, dim=-1)  # [B K]
    soft_label = F.softmax(-distance, dim=-1) # [B K]
    closest_mode_indices = torch.argmin(distance, dim=-1) # [B]
 
    return soft_label, closest_mode_indices
        
        
if __name__ == "__main__":
    args = get_args_parser()
    if args.exp_name is None:
        print("It is recommended to specific your experiment name")
        args.exp_name = ""
    
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f.read())
        
    args.output_dir = os.path.join("./workdirs", args.exp_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, cfg)