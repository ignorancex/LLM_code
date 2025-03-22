import numpy as np
import torch
from torchinfo import summary
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import argparse
import json
import yaml
import math
import os
import random
from collections import OrderedDict
from labml import monit, lab, tracker, experiment, logger
from labml.logger import Text
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.optimizers.noam import Noam
from labml_nn.optimizers.adam_warmup_cosine_decay import AdamWarmupCosineDecay
from radam import RAdam
from labml_nn.transformers.retro import model as retro
from dataset import Dataset, RetroIndex
from model import NearestNeighborEncoder, RetroFittedGPT2
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
class Trainer:
    def __init__(self, device: torch.device, model: retro.RetroModel, dataloader: DataLoader, val_dataloader, optimizer: torch.optim.Optimizer, model_dir):
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.model_dir = model_dir
        self.val_dataloader = val_dataloader
        with open(model_dir+'/run.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)
        self.minifit = True if ("minifit" in config_dict["config_json"] and config_dict["config_json"]["minifit"] == "True") else False
        self.distance_metric = config_dict["config_json"]["quantizer"]
        self.n_neighbors = config_dict["config_json"]["n_neighbors"]
        if "noisy_distance_metric" in config_dict["config_json"]:
            self.noisy_distance_metric = True if (config_dict["config_json"]["noisy_distance_metric"]=="True") else False
        else:
            self.noisy_distance_metric = False
    def __call__(self, epoch, tb_writer):
        train_loss_sum = 0
        n = 0
        if self.minifit: #for minifit experiments only save one ckp per epoch
            checkpoints_modulo = len(self.dataloader)+1
        else:
            checkpoints_modulo = len(self.dataloader)//5 #save five checkpoints per epoch
        for i, data in monit.enum('Train', self.dataloader):
            if len(data) == 4:
                src, tgt, neighbors, _ = data  #the fourth argument would be the distance matrix D
            else:
                src, tgt, neighbors = data
            src, tgt, neighbors = src.to(self.device), tgt.to(self.device), neighbors.to(self.device)
            neighbors = neighbors[:, :, :self.n_neighbors]
            res = self.model(src, neighbors)
            curr_train_loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))
            if i == 0 and epoch==0:
                #Track performance soon after loading the checkpoint
                temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
                tb_writer.add_scalar("Loss/train", curr_train_loss, 0)
                tb_writer.add_scalar("ppl/train", math.exp(curr_train_loss), 0)
                tb_writer.add_scalar("Loss/val", temp_val_loss, 0)
                tb_writer.add_scalar("ppl/val", temp_val_ppl, 0)

            self.optimizer.zero_grad()
            curr_train_loss.backward()
            self.optimizer.step()
            #remove duplicate tensorboard saving
            #tracker.save({'loss.train': curr_train_loss.item()})
            #tracker.add_global_step(len(src))
            train_loss_sum += curr_train_loss.item()
            n += 1
            if i%checkpoints_modulo == 0 and i != 0:
                temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
                if os.path.exists(self.model_dir+'/best-checkpoint-model.pt'): #only save the best checkpoint
                    fb = open(self.model_dir+'/best-checkpoint-model.pt', 'rb')
                    best_curr_model = torch.load(fb)
                    best_curr_val_loss = best_curr_model["val_loss"]
                    if temp_val_loss < best_curr_val_loss:
                        torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
                else:
                    torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
                tb_writer.add_scalar("Loss/train", train_loss_sum/i, (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("ppl/train", math.exp(train_loss_sum/i), (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("Loss/val", temp_val_loss, (epoch)*len(self.dataloader)+i)
                tb_writer.add_scalar("ppl/val", temp_val_ppl, (epoch)*len(self.dataloader)+i)
        train_loss = train_loss_sum/n
        with torch.no_grad():
            train_perplexity  = torch.exp(torch.tensor(train_loss))
        tb_writer.add_scalar("Loss/train", train_loss_sum/i, (epoch)*len(self.dataloader)+n)
        tb_writer.add_scalar("ppl/train", math.exp(train_loss_sum/i), (epoch)*len(self.dataloader)+n)
        temp_val_loss, temp_val_ppl = validate(self.val_dataloader, self.device, self.model, self.loss_func, i, self.model_dir, epoch, self.n_neighbors)
        tb_writer.add_scalar("Loss/val", temp_val_loss, (epoch)*len(self.dataloader)+n)
        tb_writer.add_scalar("ppl/val", temp_val_ppl, (epoch)*len(self.dataloader)+n)
        if os.path.exists(self.model_dir+'/best-checkpoint-model.pt'):
            fb = open(self.model_dir+'/best-checkpoint-model.pt', 'rb')
            best_curr_model = torch.load(fb)
            best_curr_val_loss = best_curr_model["val_loss"]
            if temp_val_loss < best_curr_val_loss:
                #save checkpoint
                torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
        else:
            torch.save({'epoch': epoch,'i': i,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),'train_loss': train_loss_sum/i,'val_loss': temp_val_loss,}, self.model_dir+'/best-checkpoint-model.pt')
def validate(val_dl, device, model, loss_func, training_step, model_dir, epoch, n_neighbors):
    with torch.no_grad():
        val_loss_sum = 0
        m = 0
        for i, data in monit.enum('Val', val_dl):
            if len(data) == 4:
                src, tgt, neighbors, _ = data #the fourth argument would be the distance matrix D
            else:
                src, tgt, neighbors = data
            src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
            neighbors = neighbors[:, :, :n_neighbors]
            res = model(src, neighbors, validate=True)
            curr_val_loss = loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))
            val_loss_sum += curr_val_loss.item()
            m +=1
        val_loss = val_loss_sum/m
        val_perplexity  = torch.exp(torch.tensor(val_loss))
    return val_loss, val_perplexity.item()
def load_optimizer(model_parameters, config_json):
    if config_json["optimizer"] == "Noam":
        optimizer = Noam(model_parameters, lr=config_json["optimizer_lr"], betas=(0.9,0.95), eps=1e-8, d_model=config_json["d_model"], warmup=config_json["optimizer_warmup"])
    elif config_json["optimizer"] == "RAdam":
        optimizer = RAdam(model_parameters)
    elif config_json["optimizer"] == "AdamWarmupCosineDecay":
        optimizer = AdamWarmupCosineDecay(model_parameters)
    else:
        optimizer = torch.optim.AdamW(model_parameters)
    return optimizer
def train(random_seed, train_dataset_filepath, val_dataset_filepath, device, config_json):
    model_configurations = {"random_seed": random_seed, "train_dataset_filepath": train_dataset_filepath, "val_dataset_filepath": val_dataset_filepath, "device":device, "config_json":config_json}
    lab.configure({"experiments_path":"test/retro/retro_small/"})
    print(args.val_dataset_filepath)
    if "minifit" in config_json and config_json["minifit"] == "True" and train_dataset_filepath=="":
        #infer dataset name from validate filepath
        experiment_name = 'minifit_'+config_json['minifit']+'_retro_'+config_json['retro']+'_data_'+args.val_dataset_filepath.split('/')[4]
        val_dataset = Dataset(val_dataset_filepath)
        train_indices = random.sample(range(0, len(val_dataset)-1), int(len(val_dataset)/10))
        train_indices.sort(reverse=True)
        train_dl = DataLoader(val_dataset,batch_size=2,sampler=RandomSampler(train_indices, replacement=False))
        for i in train_indices:
            del val_dataset.samples[i]
        val_dl = DataLoader(val_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(val_dataset, replacement=False))
    else:
        experiment_name = 'minifit_off'+'_retro_'+config_json['retro']+'_data_'+args.val_dataset_filepath.split('/')[4]
        train_dataset, val_dataset = Dataset(train_dataset_filepath), Dataset(val_dataset_filepath)
        train_dl, val_dl = DataLoader(train_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(train_dataset, replacement=False)), DataLoader(val_dataset,batch_size=config_json["dl_batch_size"],sampler=RandomSampler(val_dataset, replacement=False))
    experiment.create(name=experiment_name)
    device = torch.device(device)
    model_dir = str(lab.get_experiments_path())+'/'+experiment_name+'/'+str(experiment.get_uuid())
    chunk_len, d_model, d_ff, n_heads, d_k, n_encoder_layers, encoder_ca_layers, n_decoder_layers, decoder_ca_layers = config_json["chunk_len"], config_json["d_model"], config_json["d_ff"], config_json["n_heads"], config_json["d_k"], config_json["n_encoder_layers"], set(config_json["encoder_ca_layers"]), config_json["n_decoder_layers"], set(config_json["decoder_ca_layers"])
    retro_flag = True if (config_json["retro"] == "On") else False
    # where to put the noise and what kind
    if "noisy_embed_neighbors" in config_json.keys() and "noisy_embed_sequence" in config_json.keys() and "noise_alpha" in config_json.keys():
        noisy_embed_neighbors = True if (config_json["noisy_embed_neighbors"] == "True") else False
        noisy_embed_sequence = True if (config_json["noisy_embed_sequence"] == "True") else False
        noise_alpha, noise_coeff = config_json["noise_alpha"], None
    elif "noisy_embed_neighbors" in config_json.keys():
        noisy_embed_sequence = False
        noisy_embed_neighbors = True if (config_json["noisy_embed_neighbors"] == "True") else False
        if "noise_alpha" in config_json.keys():
            noise_alpha, noise_coeff = config_json["noise_alpha"], None
        if "noise_coeff" in config_json.keys():
            noise_alpha, noise_coeff = None, config_json["noise_coeff"]
    elif "noisy_embed_sequence" in config_json.keys() and "noise_alpha" in config_json.keys():
        noisy_embed_neighbors = False
        noisy_embed_sequence = True if (config_json["noisy_embed_sequence"] == "True") else False
        noise_alpha, noise_coeff = config_json["noise_alpha"], None
    else:
        noisy_embed_neighbors, noisy_embed_sequence = False, False
        noise_alpha, noise_coeff = None, None
    # load from checkpoint or no
    if config_json["gpt2"] != "True":
        raise NotImplementedError("Only retro-fitted GPT-2 is implemented.")
    gpt2_config = GPT2Config()
    if config_json["load_from_checkpoint"] != "":
        model = RetroFittedGPT2.from_pretrained(config_json["load_from_checkpoint"])
        if retro_flag != model.retro_flag and "true_retrofit" in config_json and config_json["true_retrofit"] == "True": #we are retro-fitting a retro-off checkpoint into a retro-on model
                nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, n_encoder_layers, encoder_ca_layers, gpt2_config.n_embd, n_heads, d_k, d_ff, retro_flag=retro_flag)
                model = RetroFittedGPT2(ca_layers=decoder_ca_layers, chunk_len=chunk_len,d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder,retro_flag=retro_flag, gpt2_config=gpt2_config,noisy_embed_sequence=noisy_embed_sequence, noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff)
                f = open(config_json["load_from_checkpoint"], 'rb')
                state_dict = torch.load(f, map_location=device)["model_state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k[:7] == "module.":
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict, strict=False)# load params
                optimizer = load_optimizer(model.parameters(), config_json)
        else:
            optimizer = load_optimizer(model.parameters(), config_json)
            with open(config_json["load_from_checkpoint"], 'rb') as f:
                state_dict = torch.load(f)
                optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    else:
        nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, n_encoder_layers, encoder_ca_layers, gpt2_config.n_embd, n_heads, d_k, d_ff, retro_flag=retro_flag)
        model = RetroFittedGPT2(ca_layers=decoder_ca_layers, chunk_len=chunk_len,d_k=d_k, d_ff=d_ff, encoder=nearest_neighbor_encoder,retro_flag=retro_flag, gpt2_config=gpt2_config,noisy_embed_sequence=noisy_embed_sequence, noisy_embed_neighbors=noisy_embed_neighbors, noise_alpha=noise_alpha, noise_coeff=noise_coeff)
        optimizer = load_optimizer(model.parameters(), config_json)
        
    # freeze or unfreeze layers
    if "unfreeze_gpt2" not in config_json: #keep it frozen.
        for param in model.transformer.parameters():
            param.requires_grad = False 
    else:
        for param in model.transformer.parameters():
            param.requires_grad = True 
    if retro_flag and "true_retrofit" in config_json and config_json["true_retrofit"] == "True":
            for param in model.ffw.parameters():
                param.requires_grad = False
            for param in model.read.parameters():
                param.requires_grad = False

    #over ride param updates for minfit
    if config_json["minifit"] == "True":
        for param in model.ffw.parameters():
            param.requires_grad = True
        for param in model.read.parameters():
            param.requires_grad = True
        if config_json["retro"] == "On":
            for param in model.encoder.ca.parameters():
                param.requires_grad = False
            for param in model.cca.parameters():
                param.requires_grad = False

    model = nn.DataParallel(model)
    model = model.to(device)
    experiment.add_pytorch_models(model=model)
    with experiment.start():
        with open(model_dir+'/model_summary.txt', 'w') as f:
            f.write(str(summary(model)))
        with open(model_dir+'/run.yaml', 'a') as fp:
            yaml.dump(model_configurations, fp)
        if "minifit" in config_json and config_json["minifit"]=="True" and train_dataset_filepath=="":
            #save what samples we validated on, what samples we trained on
            with open(model_dir+"/val_train_split.csv", "w") as f:
                f.write("val_len, train_ids")
                f.write("\n")
                f.write(str(len(val_dataset)) +","+str(train_indices))
        trainer = Trainer(device, model, train_dl, val_dl, optimizer, model_dir, )
        loss_func = nn.CrossEntropyLoss()
        tb_writer = SummaryWriter(log_dir=model_dir+'/runs/')
        for epoch in monit.loop(config_json["epochs"]):
            trainer(epoch, tb_writer)
            tracker.new_line()
            torch.cuda.empty_cache()
            tb_writer.flush()
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_dir+'/last-checkpoint-model.pt')
    tb_writer.close()
    f.close()
    fp.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", help="Random seed.", default=42)
    parser.add_argument("--train_dataset_filepath", help="Path to the train dataset.", default="")
    parser.add_argument("--val_dataset_filepath", help="Path to the val dataset.")
    parser.add_argument("--config_filepath", help="Path to the config json file.")
    args = parser.parse_args()
    config_json = json.load(open(args.config_filepath))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = int(args.random_seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    train(random_seed=seed, train_dataset_filepath=args.train_dataset_filepath, val_dataset_filepath=args.val_dataset_filepath, device=device, config_json=config_json)
