import random
import warnings
import copy
import torch
import gc
import pickle
import sys
import time
sys.path.append("../models")

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy as sp

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')


class ExtractAct:
    def __init__(self, model, device, final_conv_layer, classes, trainloader_clean, trainloader_noise, bins_list, model_type):
        self.model = model
        self.device = device
        self.final_conv_layer = final_conv_layer #To identify and extract final layer embeddings
        self.classes = classes #To identify final output layer
        self.trainloader_clean = trainloader_clean
        self.trainloader_noise = trainloader_noise

        self.bins_list = bins_list #Bin size for creating PDFs

        self.output_dict = {} #Holds data as key-value pairs. Keys: Layer name, Value: Activation values of filter/neuron
        self.final_layer_embedding = [] #Holds final layer embeddings for each batch. Non-averaged filter activation outputs
        self.clean_final_embeddings = [] #Holds all final layer embeddings for all batches of clean data
        self.noise_final_embeddings = [] #Holds all final layer embeddings for all batches of noise data
        self.groundtruths = [] #Holds ground truth label for each sample

        self.distances = {} #Holds layer wise distances (not averaged by number of samples)
        self.final_distances = {} #Holds final distances values (averaged at the end by number of layers and samples)

        self.model_type = model_type #CNN or ViT

        self.model.to(self.device)
        self.distance_names = ["Wasserstein", "Bhattacharya", "Jenson_Shannon", "KL_divergence"]
        
        for each_bin in self.bins_list:
            self.final_distances[str(each_bin)] = {}
            for each_dist in self.distance_names:
                self.final_distances[str(each_bin)][each_dist] = 0        


        for name, module in self.model.named_modules():
            #Iterate through all named modules in the network
            if hasattr(module, "out_features") or hasattr(module, "out_channels"):
                self.hook_driver(name, self.final_conv_layer, module)


    def hook_driver(self, name, final_conv, module=None):
        #Driver function to set up forward hooks

        if name not in self.output_dict.keys() and len(name) > 1:
            #If layer name is not present in the main dictionary and has a name, add key to dictionary
            self.output_dict[name] = []
            module.register_forward_hook(self.hook(name, final_conv)) #Register the forward hook for current layer
            print(f"Forward hook for {name} set")


    def hook(self, layer_name, final_conv):
        #Functions to setup forward hook
        def hook_function(module, input, output):

                if self.model_type.lower() == "cnn":
                    activation_data = output.cpu().detach().numpy()

                    #If activation data has 4 dimensions, it is the output of a convolutional layer
                    #The four dimensions -> (batch_size, number_of_filters, activation_output_height, activation_output_weight)
                    if len(activation_data.shape) == 4:
                        activation_data = activation_data.reshape(*activation_data.shape[:2], -1)
                        if layer_name == final_conv:
                            #If layer is final convolution layer, extract activation data without averaging
                            self.final_layer_embedding.append(activation_data)
                        activation_data = activation_data.mean(axis=-1, keepdims=True)   

                    self.output_dict[layer_name].append(activation_data)        

                elif self.model_type.lower() == "vit":
                    activation_data = output.cpu().detach().numpy()
                    activation_data = activation_data.reshape(*activation_data.shape[:1], -1) 
                        
                    self.output_dict[layer_name].append(activation_data) 
                    if layer_name == final_conv:
                        self.final_layer_embedding.append(activation_data)  
                
                else:
                    print("Invalid model type")
                    exit()
                    
        return hook_function


    def create_pdfs(self, data_dict):
        pdf_dict = {}
        for key, value in data_dict.items():
            if value.shape[1] > self.classes:
                pdf_dict[key] = []

                for bins in self.bins_list:
                    sample_s_pdfs = []
                    for each_sample_s_data in value:
                        hist, _ = np.histogram(each_sample_s_data, bins=bins, density=True)
                        hist = hist / hist.sum()
                        sample_s_pdfs.append(hist)
                    pdf_dict[key].append(sample_s_pdfs)
        return pdf_dict


    def wasserstein_dist(self, prob1, prob2):
        return wasserstein_distance(prob1, prob2)


    def jensen_shannon_dist(self, prob1, prob2, base=np.e):
        prob1, prob2 = np.asarray(prob1), np.asarray(prob2)
        prob1, prob2 = prob1/prob1.sum(), prob2/prob2.sum()
        m = 1./2*(prob1 + prob2)
        return sp.stats.entropy(prob1, m, base=base)/2. +  sp.stats.entropy(prob2, m, base=base)/2.


    def bhattacharya_dist(self, prob1, prob2):
        return 1 - np.sum(np.sqrt(np.multiply((prob1/np.sum(prob1)), (prob2/np.sum(prob2)))))


    def kl_divergence(self, prob1, prob2):
        pre_sum = rel_entr(prob1, prob2)
        pre_sum = np.nan_to_num(pre_sum, nan=0.0, posinf=0.0, neginf=0.0)
        kl_div = sum(pre_sum)
        return kl_div


    def clear_list(self):
        for key, value in self.output_dict.items():
            self.output_dict[key] = []
        self.final_layer_embedding = []


    def flatten_vector(self):
        for key, value in self.output_dict.items():
            #self.output_dict[key] = []
            self.output_dict[key] = np.array(self.output_dict[key])
            self.output_dict[key] = self.output_dict[key].reshape(-1, *self.output_dict[key].shape[2:]) 
            self.output_dict[key] = self.output_dict[key].squeeze()


    def create_layer_distance_structure(self, data_dict):
        for key, value in data_dict.items():
            if value.shape[1] > self.classes:
                if key not in self.distances:
                    self.distances[key] = {}
                    for each_bin in self.bins_list:
                        self.distances[key][str(each_bin)] = {}
                        for each_dist in self.distance_names:
                            self.distances[key][str(each_bin)][each_dist] = 0   


    def compute_distances(self, clean_dict_pdf, noise_dict_pdf):
        for each_layer in clean_dict_pdf.keys():
            for i, bins in enumerate(self.bins_list):
                for (each_pdf_clean, each_pdf_noise) in zip(clean_dict_pdf[each_layer][i], noise_dict_pdf[each_layer][i]):
                    self.distances[each_layer][str(bins)]["Wasserstein"] += self.wasserstein_dist(each_pdf_clean, each_pdf_noise)
                    self.distances[each_layer][str(bins)]["Jenson_Shannon"] += self.jensen_shannon_dist(each_pdf_clean, each_pdf_noise)
                    self.distances[each_layer][str(bins)]["Bhattacharya"] += self.bhattacharya_dist(each_pdf_clean, each_pdf_noise)
                    self.distances[each_layer][str(bins)]["KL_divergence"] += self.kl_divergence(each_pdf_clean, each_pdf_noise)


    def distance_average_aggregate(self):
        for each_layer in self.distances.keys():
            for bins in self.bins_list:
                for each_distance in self.distances[each_layer][str(bins)].keys():
                    self.final_distances[str(bins)][each_distance] += self.distances[each_layer][str(bins)][each_distance]

        for bins in self.bins_list:
            for each_distance in self.final_distances[str(bins)].keys():
                self.final_distances[str(bins)][each_distance] /= len(self.distances.keys())
                self.final_distances[str(bins)][each_distance] /= len(self.groundtruths)


    def forward_pass(self):
        with torch.no_grad():
            self.model.eval()
            for i, ((x_1, y_1), (x_2, y_2)) in enumerate(zip(self.trainloader_clean, self.trainloader_noise)):
                self.groundtruths.extend(y_1)

                if not (y_1 == y_2).all():
                    print("Error") 
                    exit()
                x_1 = x_1.to(self.device)
                out = self.model(x_1)

                self.flatten_vector()
                self.create_layer_distance_structure(self.output_dict) #Create layer wise distances structure, done only once (Not created again due to check within function)
                self.clean_final_embeddings.append(self.final_layer_embedding) #Append current batch clean data embeddings
                clean_vector = copy.deepcopy(self.output_dict) 
                
                self.clear_list() #Clear output_dict and final_layer_embeddings for noise data
               
                #Forward pass batch of noise data
                x_2 = x_2.to(self.device)
                out = self.model(x_2)

                self.flatten_vector()
                self.noise_final_embeddings.append(self.final_layer_embedding) #Append current batch noise data embeddings
                noise_vector = copy.deepcopy(self.output_dict)
                self.clear_list() #Clear output_dict and final_layer_embeddings for next batch

                #Create PDFs for calculating distance for current batch
                clean_dict = self.create_pdfs(clean_vector) 
                noise_dict = self.create_pdfs(noise_vector)            
                
                #Compute and add distances for current batch
                self.compute_distances(clean_dict, noise_dict)
                print(f"\r{i+1}/{len(self.trainloader_clean)}",end="")
                
        
        #Reshape embedding layer output to be of shape -> (Sample, Neuron values) . Reshaped from (Number of batches, 1, Number of samples per batch, Neuron value)
        self.clean_final_embeddings = np.array(self.clean_final_embeddings)

        self.clean_final_embeddings = self.clean_final_embeddings.reshape(-1, *self.clean_final_embeddings.shape[3:])
        self.clean_final_embeddings = self.clean_final_embeddings.reshape(self.clean_final_embeddings.shape[0], -1)

        self.noise_final_embeddings = np.array(self.noise_final_embeddings)
        self.noise_final_embeddings = self.noise_final_embeddings.reshape(-1, *self.noise_final_embeddings.shape[3:])
        self.noise_final_embeddings = self.noise_final_embeddings.reshape(self.noise_final_embeddings.shape[0], -1)

        self.groundtruths = np.array(self.groundtruths)
        self.distance_average_aggregate()

        return self.clean_final_embeddings, self.noise_final_embeddings, self.groundtruths, self.final_distances, self.distances