import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset, ConcatDataset
import clip
import os

from models import HugeModel
from losses import return_task_loss
from .client import *
from copy import deepcopy
from utils.viz import plot_data_distribution

client_types = {
    "fedavg": FedAvgClient,
    "fedprox": FedProxClient
}

# Global model is validated on the data of all clients
# Mean and standard deviation are recorded for the performance
class SimpleNetwork():

    def __init__(self, sample_dataset, assignments, device, args, save_dir) -> None:
        
        # Setting attributes based on inputs
        self.method = args.method
        self.n_clients = args.n_clients
        self.sgd_per_epoch = args.sgd_per_epoch
        self.epoch_per_round = args.epoch_per_round
        self.device = device
        self.alpha = args.alpha
        self.task = args.task
        self.n_rounds = args.n_rounds
        self.save_dir = save_dir
        self.rand_seed = args.rand_seed
        self.mean_loss_history = []
        self.std_loss_history = []
        self.best_global_mean_loss = np.Inf
        self.criterion = return_task_loss(self.task)

        # Initializing global model
        if args.start_warm:
            init_model_path = os.path.join(args.init_model_dir, f"trained_{args.task}_model.pth")
            assert os.path.exists(init_model_path), "Warm model not does not exist!"
        else:
            init_model_path = os.path.join(args.init_model_dir, f"init_model_{self.task}.pth")
            assert os.path.exists(init_model_path), "Initial model to load global model from does not exist! Use 'model_initialization.py' to create it!"
        
        sample_clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.global_model = HugeModel(sample_clip_model, self.task).to(self.device)
        self.global_model.load_state_dict(torch.load(init_model_path))

        # Distribute the data based on dirichlet across clients
        client_class = client_types[args.method]
        self.clients = self.distribute_data(args, assignments, sample_dataset, self.global_model, device, client_class)
    
    def train(self):
        
        # Record global model initial performance on client datasets
        val_loss_m, val_loss_s = self.validate_global_model()
        self.mean_loss_history.append(val_loss_m)
        self.std_loss_history.append(val_loss_s)

        # Loop for the federated training
        for i in range(self.n_rounds):
            for client in self.clients:
                # Downlink
                client.sync_params(self.global_model)
                # Local training
                for _ in range(self.epoch_per_round):
                    if self.method in client_types.keys():
                        client.train_once()
                    else:
                        raise ValueError(f"Method {self.method} not supported!")
            
            # Uplink & Aggregation
            self.aggregate_weights()

            # Record global model performance on client datasets
            val_loss_m, val_loss_s = self.validate_global_model()
            self.mean_loss_history.append(val_loss_m)
            self.std_loss_history.append(val_loss_s)
            print(f"Round {i+1}/{self.n_rounds}, Global model mean validation Loss: {val_loss_m}")

            if val_loss_m < self.best_global_mean_loss:
                self.best_global_mean_loss = val_loss_m
                torch.save(self.global_model.state_dict(), os.path.join(self.save_dir, f"eetf_{self.rand_seed}_global_{self.task}_model_{str(self.alpha)}_{self.sgd_per_epoch}.pth"))
            
            # Save loss history to csv file
            loss_df = pd.DataFrame(list(zip(self.mean_loss_history, self.std_loss_history)), columns=['mean', 'std'])
            loss_df.to_csv(os.path.join(self.save_dir, f'eetf_{self.rand_seed}_global_{self.task}_model_loss_{self.alpha}_{self.sgd_per_epoch}.csv'), index=False)
        
        
        
        return self.mean_loss_history, self.std_loss_history
    
    def aggregate_weights(self):
        # Get the weighted average of the model parameters based on client data size
        global_dict = deepcopy(self.global_model.state_dict())
        for key in global_dict.keys():
            # global_dict[key] = torch.stack([client.model.state_dict()[key].float() for client in self.clients], dim=0).mean(dim=0)
            global_dict[key] = torch.stack([client.model.state_dict()[key].float() * len(client.dataset) for client in self.clients], dim=0).sum(dim=0) / sum([len(client.dataset) for client in self.clients])

        
        self.global_model.load_state_dict(global_dict)
    
    def validate_global_model(self):
        
        self.global_model.eval()
        client_based_losses = []
        for client in self.clients:
            total_loss = 0
            with torch.no_grad():
                for idx, (image, label) in enumerate(client.dataloader):

                    image = image.to(self.device).float()
                    if self.task == "class_scene":
                        label = label[self.task].to(self.device)
                    else:
                        label = label[self.task].to(self.device).float()

                    _, outputs = self.global_model(image)
                    loss = self.criterion(outputs, label)
                    total_loss += loss.item()
            
            client_based_losses.append(total_loss/len(client.dataloader))
        
        return np.mean(np.array(client_based_losses)), np.std(np.array(client_based_losses))
    
    def distribute_data(self, args, assignments, sample_dataset, global_model, device, client_class=FedAvgClient):
        data_clustered = {}
        for i in range(args.n_clusters):
            cluster_idxs = assignments == i
            data_clustered[i] = Subset(sample_dataset, np.where(cluster_idxs)[0])
        avail_idxs = {i: list(range(len(data_clustered[i]))) for i in range(args.n_clusters)}

        cluster_nums = np.array([len(data_clustered[i]) for i in range(args.n_clusters)] )
        sum_num = sum(cluster_nums)
        cluster_ratios = np.array([len(data_clustered[i])/sum_num for i in range(args.n_clusters)])
        print(f"Cluster ratios = {cluster_ratios}\nCluster nums = {cluster_nums}")
        
        trial = 0
        while(True):
            trial += 1
            dirichlet_distribution = np.random.dirichlet(self.alpha * cluster_ratios, self.n_clients)
            pt_inv = np.linalg.pinv(dirichlet_distribution.T)
            n_data_per_client = np.matmul(pt_inv, cluster_nums)
            n_data_per_client = n_data_per_client.astype(int)
            datapoints_per_cluster = np.dot(n_data_per_client, dirichlet_distribution)
            bad_clusters = np.where(datapoints_per_cluster > cluster_nums)[0]
            if trial%10000 == 0:
                print(f"Trial {trial}")
            if sum(n_data_per_client<args.min_client_ds_len) == 0 and (sum(n_data_per_client) <= sum_num) and len(bad_clusters) == 0:
                
                print(f"datapoints from each cluster?\n{np.array(list(zip(datapoints_per_cluster.astype(int), cluster_nums))).reshape(-1, 2)}")
                print(f"clusters with more than available datapoints:\n \
                      {bad_clusters}")
                print(f"using {sum(n_data_per_client)}/{sum_num} datapoints")
                # raise KeyboardInterrupt("Stop here")
                break
        
        if args.plot_distribution:
            plot_data_distribution(self, len(list(avail_idxs.keys())), dirichlet_distribution)
        
        clients = []
        for client in range(args.n_clients):
            client_prob_per_cluster = dirichlet_distribution[client]
            print(f"Creating dataset for client {client}")
            print(client_prob_per_cluster)
            client_data = {}
            for cluster_idx, cluster_val in enumerate(list(avail_idxs.keys())):    
                num_samples = int(n_data_per_client[client] * client_prob_per_cluster[cluster_idx])         # Get the number of samples for the cluster
                print(f"{num_samples} samples from cluster {cluster_val}")                                  
                if len(avail_idxs[cluster_val]) < num_samples:                                              # If there are not enough samples in the cluster
                    raise ValueError("Not enough samples in cluster")                                       # Raise an error
                sampled_indexes = np.random.choice(avail_idxs[cluster_val], num_samples, replace=False)     # Sample the indexes for cluster
                cluster_subds = Subset(data_clustered[cluster_val], sampled_indexes)                        # Create a subset of the cluster dataset for the client data
                client_data[cluster_val] = cluster_subds
                avail_idxs[cluster_val] = list(set(avail_idxs[cluster_val]) - set(sampled_indexes))         # Remove the sampled indexes from the available indexes
                # print(f"Cluster {cluster_val} has {len(avail_idxs[cluster_val])} samples left")
            client_ds = ConcatDataset([client_data[i] for i in client_data.keys()])                         # Concatenate the datasets for the client to create the final client ds
            print(f"final client dataset len: {len(client_ds)}")
            clients.append(client_class(client_ds, global_model, device, args, self))                         # Create a client using the created ds and add it to the clients list
        
        return clients