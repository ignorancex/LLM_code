import numpy as np
from pathlib import Path
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
import argparse
import json
import os
import math
import yaml
from labml import monit, lab, tracker, experiment, logger
from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.transformers.retro import model as retro
from dataset import Dataset
from model import NearestNeighborEncoder, RetroFittedGPT2
from transformers import GPT2Config, GPT2Tokenizer
from tqdm import tqdm
import math


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)
import torch.nn.functional as F
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            Args:
                logits: logits distribution shape (..., vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs >= top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
            logits[indices_to_remove] = filter_value
        return logits
def generate(res, length=64, temperature=1, top_p=0.9, top_k=50, strategy = "top_p"):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #TODO add "default" prompt
    if strategy == "top_p":
        top_p_output = []
        for i in range(1, length+1):
            logits = res[:, length+i, :] / temperature
            logits = _top_k_top_p_filtering(logits, top_p, top_k)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)[0][0]
            top_p_output.append(pred)
        res = ["".join([gpt2_tokenizer.decode(x) for x in pred])]
        return res
    elif strategy == "greedy":
        pred = res[0].argmax(-1) #this starts breaking down and repeating endlessly eventually.
        pred = pred[64:128] #next chunk
        res = ["".join([gpt2_tokenizer.decode(x) for x in pred])]
        return res
    elif strategy == "multinomial":
        temperature = 1.0
        word_weights = res[0].squeeze().div(temperature).exp().cpu() #this doesn't repeat AS much but still repeats a lot.
        #TODO rewrite this like topp
        pred = torch.multinomial(word_weights, 1)[64:128]
        assert(len(pred) == 64), len(pred)
        res = ["".join([gpt2_tokenizer.decode(x) for x in pred])]
        return res
    else:
        raise NotImplementedError

def generate_manually(res, src, tgt, model, neighbors, index):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    output_top_p = []
    output_greedy = []
    output_multinomial = []
    src_list = src.clone().detach()
    src_list = src_list.tolist()[0]
    for i in range(0,64):
        aux_top_p = src_list[i:].copy()+output_top_p[:i].copy()
        aux_greedy = src_list[i:].copy()+output_greedy[:i].copy()
        aux_multinomial = src_list[i:].copy()+output_multinomial[:i].copy()
        next_top_p = model(torch.tensor([aux_top_p], device=device), neighbors,validate=True) #no teacher forcing
        next_greedy = model(torch.tensor([aux_greedy], device=device), neighbors,validate=True) #no teacher forcing
        next_multinomial = model(torch.tensor([aux_multinomial], device=device), neighbors,validate=True) #no teacher forcing
        temperature=1
        logits = next_top_p[:, -1, :] / temperature #top-p
        logits = top_k_top_p_filtering(logits, top_p=0.9, top_k=0)
        probs = F.softmax(logits, dim=-1)
        probs[0][aux_top_p[-1]] = 0 #i do not want to predict as the next token the same one i already had.
        pred = torch.multinomial(probs, num_samples=1)[0][0]
        output_top_p.append(pred)
        pred = next_greedy[0].argmax(-1)  # greedy #this starts breaking down and repeating endlessly eventually.
        pred = pred[-1] #only the last token is the actual "prediction"
        output_greedy.append(pred)
        temperature = 1.0 #multinomial
        word_weights = next_multinomial[0].squeeze().div(temperature).exp().cpu() #this doesn't repeat AS much but still repeats a lot.
        word_weights[word_weights == float("Inf")] = 0
        pred = torch.multinomial(word_weights, 1)[-1][0]
        output_multinomial.append(pred)
    with open(outfile_name.replace(".txt","-"+str(index)+".txt"),"w")as f:
        f.write("\n")
        f.write("top-p\n")
        f.write(str("".join([gpt2_tokenizer.decode(x) for x in output_top_p]))+"\n")
        f.write("greedy\n")
        f.write(str("".join([gpt2_tokenizer.decode(x) for x in output_greedy]))+"\n")
        f.write("multinomial\n")
        f.write(str("".join([gpt2_tokenizer.decode(x) for x in output_multinomial]))+"\n")
        f.write("src\n")

def validate(checkpoint_path, val_dataset_path, config_dict, device, output_dir_path, window, shuffled_neighbors, no_neighbors, generate_manually_flag, generate_manually_index, noisy_flag, noisy_coeff, outfile_name,is_aihwkit=0):
    print("retrieval "+str(retrieval_on))

    if val_dataset_path == None:
        val_dataset = Dataset(config_dict["val_dataset_filepath"])
    else:
        val_dataset = Dataset(val_dataset_path)

    if "minifit" in config_dict["config_json"] and config_dict["config_json"]["minifit"] == "True": #in this case we need to evaluate on the same indices

        valsplit_file = ("/").join(checkpoint_path.split("/")[:-1]+["val_train_split.csv"])
        valsplit_file_path = Path(valsplit_file)
        if valsplit_file_path.is_file():
            with open(valsplit_file, "rb") as fb:
                f.readline() #header
                b = f.readline() #list
                b_list = b.split("[")[1].replace("]","")

            train_indices = np.array(b_list.split(",")).astype(int)
            for i in train_indices:
                del val_dataset.samples[i]
        
        seed = int(config_dict["random_seed"]) #and we need to set the same random seed because of the sampler
        torch.random.manual_seed(seed)
        np.random.seed(seed)

    val_dl = DataLoader(val_dataset,batch_size = 1,sampler=RandomSampler(val_dataset, replacement=False))
    model = RetroFittedGPT2.from_pretrained(checkpoint_path,is_aihwkit=is_aihwkit)
    model.eval()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    if generate_manually_flag == "True":
        for i, data in monit.enum('Val', val_dl):
            if i == generate_manually_index: #ugly but works
                if len(data) == 4:
                    src, tgt, neighbors,_ = data
                else:
                    src, tgt, neighbors = data
                src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
                if retrieval_on:
                    if noisy_flag == True:
                        model.noise_coeff = noisy_coeff #this is the coefficient for the gaussian noise on the word embeddings
                        res = model(src, neighbors, validate=False)
                    else:
                        res = model(src, neighbors, validate=True)
                else:
                    res = model(src, [])
                
                generate_manually(res, src, tgt, model, neighbors, i)
                gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                with open(outfile_name.replace(".txt","-"+str(i)+".txt"),"a+")as f:
                    f.write("Context\n")
                    print(gpt2_tokenizer.batch_decode(src[:,:768]),file=f)
                    f.write("Real continuation\n")
                    print(gpt2_tokenizer.batch_decode(src[:,768:768+64]),file=f)
                return
    ###### val ########
    with torch.no_grad():
        val_loss_sum = 0
        m = 0
        # Iterate through training data
        aux = []
        for i, data in monit.enum('Val', val_dl):
            #print(i)
            # Move data to the device
            if len(data) == 3:
                src, tgt, neighbors = data
                src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
            elif len(data) == 4:
                src, tgt, neighbors, _ = data
                src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
                #D = D.cpu()

            if shuffled_neighbors == "True":
                src, tgt, neighbors, D = data
                src, tgt, neighbors = src.to(device), tgt.to(device), neighbors.to(device)
                D = D.cpu()
                #TODO at the moment this is hardcoded for batch size 2
                #add noise to the distance matrix
                #print(D.shape) #2,16,14 so batch size x num chunks per seq x num neighbors max
                D = D[:,:,:10]
                sigma = torch.abs(D)
                sigma = torch.mean(sigma, 2) #this should be a vector of length 16
                #assert(len(sigma) == 16), len(sigma)
                shuffled_neighbors_top_k = torch.empty(neighbors.shape[0], neighbors.shape[1], config_dict["config_json"]["n_neighbors"], neighbors.shape[3], dtype=neighbors.dtype, device=device)
                for chunk_i in range(sigma.shape[1]):
                    #add noise to distances
                    if len(sigma[:,chunk_i]) == 2:
                        #this depends on the batch size unfortunately
                        gaussian_noise_0 = torch.empty(D.shape[2], device=D.device).normal_(mean=0,std=0.196*sigma[0,chunk_i])
                        gaussian_noise_1 = torch.empty(D.shape[2], device=D.device).normal_(mean=0,std=0.196*sigma[1,chunk_i])
                        D[0,chunk_i] = D[0,chunk_i]+gaussian_noise_0.cpu()
                        D[1,chunk_i] = D[1,chunk_i]+gaussian_noise_1.cpu()
                        idx1 = torch.topk(D[0,chunk_i], config_dict["config_json"]["n_neighbors"], largest=False).indices
                        idx2 = torch.topk(D[1,chunk_i], config_dict["config_json"]["n_neighbors"], largest=False).indices
                        shuffled_neighbors_top_k[0,chunk_i,:] = torch.index_select(neighbors[0,chunk_i,:], 0, idx1.to(device))
                        shuffled_neighbors_top_k[1,chunk_i,:] = torch.index_select(neighbors[1,chunk_i,:], 0, idx2.to(device))
                    else:
                        gaussian_noise_0 = torch.empty(D.shape[2], device=D.device).normal_(mean=0,std=0.196*sigma[0,chunk_i])
                        D[0,chunk_i] = D[0,chunk_i]+gaussian_noise_0.cpu()
                        idx1 = torch.topk(D[0,chunk_i], config_dict["config_json"]["n_neighbors"]).indices
                        shuffled_neighbors_top_k[0,chunk_i,:] = torch.index_select(neighbors[0,chunk_i,:], 0, idx1.to(device))
                neighbors = shuffled_neighbors_top_k
            
            """
            if False: #not done yet but this is putting the retrieval online
                #put retrieval online
                import pickle
                from retro_index import RetroIndex

                val_index = RetroIndex(device, None, val_index_filepath, config_dict["config_json"], emb="SentenceTransformer")
                with open(flattened_text_tokenized_path_val, "rb") as fb:
                    val_retrieval_database_tokens = pickle.load(fb)
                D, neighbor_offsets = val_index(chunks, chunk_offsets) #D is the distance matrix
                D = D.tolist()
                neighbors = [[val_retrieval_database_tokens[j: j + chunk_len * 2] for j in n_off] for n_off in neighbor_offsets]
                neighbors = neighbors[:, :, :config_dict["config_json"]["n_neighbors"]]
            """
            if no_neighbors == "True": #mask out continuation, do not retrieve neighbors
                neighbors = neighbors[:, :, :1]
                for chunk_i in range(len(neighbors[0])):
                    begin = 64*chunk_i
                    x = torch.full(src[0][begin:begin+64].shape, 50256, device="cuda:0") #50256 is the token for EOS so for gpt2 the padding token.
                    neighbors[0][chunk_i] = torch.cat((src[0][begin:begin+64],x),0).repeat(len(neighbors[0][chunk_i]),1)
            # Forward pass
            if retrieval_on:
                if noisy_flag == "True":
                    model.noise_coeff = noisy_coeff #this is the coefficient for the gaussian noise on the word embeddings
                    res = model(src, neighbors, validate=False)
                else:
                    res = model(src, neighbors, validate=True)
            else:
                res = model(src, [])
            # Calculate loss
            if window == "True":
                #on only the part where we have at least 75% of the context? so.. 768 tokens at least? okay...
                #for some of them retro used 14/16 as context so 896 or 87.5%
                trunc_res = res[:,768:,:]
                trunc_tgt = tgt[:,768:]
                curr_val_loss = loss_func(trunc_res.reshape(-1, trunc_res.shape[-1]), trunc_tgt.reshape(-1))
            else: #On all of it, indiscriminately
                curr_val_loss = loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))
            aux.append(curr_val_loss.item())
            val_loss_sum += curr_val_loss.item()
            m +=1
        val_loss = val_loss_sum/m
        val_perplexity  = torch.exp(torch.tensor(val_loss))
    
    print(val_perplexity)
    if output_dir_path == "":
        return
    ## val ##
    inference_dir_path = output_dir_path + "/inference"
    if not os.path.isdir(inference_dir_path):
        os.makedirs(inference_dir_path)
    f = open(inference_dir_path+'/val_loss.csv', 'a+')
    writer = csv.writer(f)
    writer.writerow(["val_loss"])
    writer.writerows([[x] for x in aux])
    h = open(inference_dir_path+'/val_perplexity.csv', 'a+')
    writer = csv.writer(h)
    writer.writerow(["val_ppl"])
    writer.writerows([[np.exp(x)] for x in aux])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", help="Path to the model checkpoint.")
    parser.add_argument("--val_dataset_path", help="Specify what dataset to evaluate on.", default=None)
    parser.add_argument("--output_dir_path", help="Specify the output directory.", default="")
    parser.add_argument("--seventy_five_perc_window", help="Specify if you would like to compute the validation perplexity for the 75% context window.", default="True")
    parser.add_argument("--shuffled_neighbors", help="True or False, if you want to shuffle the neighbors and then take the new top-k. Careful! At the moment this only works for batch size 2.", default="False")
    parser.add_argument("--no_neighbors", help="True or False, if you want to replace real neighbors with the sequence itself and a masked out continuation.", default="False")
    parser.add_argument("--generate_manually_flag", help="True or False, if you want to generate a specific validation sample continuation manually",default="False")
    parser.add_argument("--generate_manually_index", help="An integer, denoting which validation sample to use for the manual generation.", default="-1")
    parser.add_argument("--outfile_name", help="Which file to save the generated samples into.", default=None)
    parser.add_argument("--noisy_flag", help="True or False, if you would like to add Gaussian noise to the neighbor word embeddings.", default="False" )
    parser.add_argument("--noisy_coeff", help="A floating point number denoting the coefficient for the Gaussian noise, usually 0.196, 0.4, or 1.0", default="0")
    parser.add_argument("--is_aihwkit", help="Enable noise sim with AIHWKIT", default="0")

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    val_dataset_path = args.val_dataset_path
    output_dir_path = args.output_dir_path
    window = args.seventy_five_perc_window
    shuffled_neighbors = args.shuffled_neighbors
    no_neighbors = args.no_neighbors
    generate_manually_flag = args.generate_manually_flag
    generate_manually_index = int(args.generate_manually_index)
    noisy_flag = args.noisy_flag
    noisy_coeff = float(args.noisy_coeff)
    outfile_name = args.outfile_name

    if output_dir_path == None:
        raise Exception("Please specify an output_dir_path.")

    experiment_dir_path = "/".join(checkpoint_path.split("/")[:-1])
    config_filepath = experiment_dir_path + "/run.yaml"
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    retrieval_on = True if (config_dict["config_json"]["retro"] == "On") else False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    validate(checkpoint_path, val_dataset_path, config_dict, device, output_dir_path, window, shuffled_neighbors, no_neighbors,
            generate_manually_flag, generate_manually_index, noisy_flag, noisy_coeff, outfile_name=outfile_name,is_aihwkit=args.is_aihwkit)
    
