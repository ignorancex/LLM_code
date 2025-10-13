import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset, ConcatDataset

from models import HugeModel
from losses import EdgeLoss
from .client import SimpleClient
from copy import deepcopy
import clip
import os
from .client import SingleTaskClient
from losses import return_task_loss


# Validation data is chosen separately                   #
# for the global model based on the original             #
# ratio of the clusters existing in the "global dataset" #
class SimpleNetwork():

    def __init__(self, n_device:int, sample_dataset, assignments, n_clusters, alpha:float, global_model:HugeModel, sgd_per_epoch:int) -> None:
        self.n_device = n_device
        self.sgd_per_epoch = sgd_per_epoch
        # Distribute the data based on dirichlet across devices
        ## Create a dataset that treats clusters like labels and the datapoints like data
        data_clustered = {}
        for i in range(n_clusters):
            cluster_idxs = assignments == i
            data_clustered[i] = Subset(sample_dataset, np.where(cluster_idxs)[0])

        avail_idxs = {i: list(range(len(data_clustered[i]))) for i in range(n_clusters)}
        
        # Get the ratio of each cluster
        sum_num = sum([len(data_clustered[i]) for i in range(n_clusters)])
        cluster_ratios = np.array([len(data_clustered[i])/sum_num for i in range(n_clusters)])
        print(cluster_ratios)
        # print(avail_idxs)
        has_enough_data = np.where(cluster_ratios >= 0.1)[0]
        new_cluster_nums = cluster_ratios[has_enough_data] * sum_num
        new_cluster_ratios = new_cluster_nums / sum(new_cluster_nums)
        print(f"new_cluster_nums: {new_cluster_nums}")
        print(f"new cluster ratios: {new_cluster_ratios}")
        n_data_per_device = min(new_cluster_nums) // (n_device+new_cluster_ratios[np.argmin(new_cluster_nums)])
        print(f"data_per_device = {n_data_per_device}")

        # Separate the validation data with uniform distribution across devices before distributing the rest across devices

        val_data = {}
        for cluster_idx, cluster_val in enumerate(has_enough_data):
            num_samples = int(n_data_per_device * new_cluster_ratios[cluster_idx])
            print(f"{num_samples} samples from cluster {cluster_val}")
            if len(avail_idxs[cluster_val]) < num_samples:
                raise ValueError("Not enough samples in cluster")
            sampled_indexes = np.random.choice(avail_idxs[cluster_val], num_samples, replace=False)
            cluster_subds = Subset(data_clustered[cluster_val], sampled_indexes)
            val_data[cluster_val] = cluster_subds
            avail_idxs[cluster_idx] = list(set(avail_idxs[cluster_idx]) - set(sampled_indexes))
        val_ds = ConcatDataset([val_data[i] for i in val_data.keys()])
        print(f"final val dataset len: {len(val_ds)}")
        self.val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        # Initializing Clients' datasets
        self.clients = []
        self.global_model = global_model
        self.loss_history = []
        self.criterion = EdgeLoss()
        self.alpha = alpha
        print(f"alpha = {alpha}")
        for device in range(n_device):
            client_prob_per_cluster = np.random.dirichlet(alpha*new_cluster_ratios)                         # Dirichlet distribution for client
            print(f"Creating dataset for client {device}")
            print(client_prob_per_cluster)
            client_data = {}
            for cluster_idx, cluster_val in enumerate(has_enough_data):                                     # For each cluster
                num_samples = int(n_data_per_device * client_prob_per_cluster[cluster_idx])                 # Get the number of samples for the cluster
                print(f"{num_samples} samples from cluster {cluster_val}")                                  
                if len(avail_idxs[cluster_val]) < num_samples:                                              # If there are not enough samples in the cluster
                    raise ValueError("Not enough samples in cluster")                                       # Raise an error
                sampled_indexes = np.random.choice(avail_idxs[cluster_val], num_samples, replace=False)     # Sample the indexes for cluster
                cluster_subds = torch.utils.data.Subset(data_clustered[cluster_val], sampled_indexes)       # Create a subset of the cluster dataset for the client data
                client_data[cluster_val] = cluster_subds
                avail_idxs[cluster_idx] = list(set(avail_idxs[cluster_idx]) - set(sampled_indexes))         # Remove the sampled indexes from the available indexes
            client_ds = torch.utils.data.ConcatDataset([client_data[i] for i in client_data.keys()])        # Concatenate the datasets for the client to create the final client ds
            print(f"final client dataset len: {len(client_ds)}")
            self.clients.append(SimpleClient(client_ds, sgd_per_epoch))                                     # Create a client using the created ds and add it to the clients list
    
    def train(self, n_rounds):
        validation_loss = self.validate_global_model()
        self.loss_history.append(validation_loss)
        for i in range(n_rounds):
            for client in self.clients:
                
                client.sync_params(self.global_model)
                client.train_once()
            
            # Aggregate the weights
            self.aggregate_weights()
            validation_loss = self.validate_global_model()
            self.loss_history.append(validation_loss)
            print(f"Round {i+1}/{n_rounds}, Global model validation Loss: {validation_loss}")
            # Save loss history to csv file
            loss_df = pd.DataFrame(self.loss_history, columns=['loss'])
            loss_df.to_csv(f'eetf_14_global_reshading_model_loss_{self.alpha}_{self.sgd_per_epoch}.csv', index=False)
        
        return self.loss_history
    
    def aggregate_weights(self):
        # Get the average of the weights
        global_dict = deepcopy(self.global_model.state_dict())
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client.model.state_dict()[key].float() for client in self.clients], dim=0).mean(dim=0)
        
        self.global_model.load_state_dict(global_dict)
    
    def validate_global_model(self):
        
        self.global_model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, (image, label) in enumerate(self.val_dataloader):

                image = image.to(torch.device("cuda")).float()
                label = label['reshading'].to(torch.device("cuda")).float()

                _, outputs = self.global_model(image)
                loss = self.criterion(outputs, label)
                total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
############################################################################################################
### Data is distributed with fixed dataset size across devices ###


class SimpleNetwork():

    def __init__(self, sample_dataset, assignments, device, args, save_dir) -> None:
        
        # Setting attributes based on inputs
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
        ## Create a dataset that treats clusters like labels and the datapoints like data
        data_clustered = {}
        for i in range(args.n_clusters):
            cluster_idxs = assignments == i
            data_clustered[i] = Subset(sample_dataset, np.where(cluster_idxs)[0])
        avail_idxs = {i: list(range(len(data_clustered[i]))) for i in range(args.n_clusters)}

        # Get the ratio of each cluster
        cluster_nums = [len(data_clustered[i]) for i in range(args.n_clusters)] 
        sum_num = sum(cluster_nums)
        cluster_ratios = np.array([len(data_clustered[i])/sum_num for i in range(args.n_clusters)])
        print(f"Cluster ratios = {cluster_ratios}")
        n_data_per_client = sum(cluster_nums) // (self.n_clients + 1)

        has_enough_data = np.where(cluster_ratios >= args.cluster_threshold)[0]
        new_cluster_nums = cluster_ratios[has_enough_data] * sum_num
        new_cluster_ratios = new_cluster_nums / sum(new_cluster_nums)
        print(f"new_cluster_nums: {new_cluster_nums}")
        print(f"new cluster ratios: {new_cluster_ratios}")
        n_data_per_client = min(new_cluster_nums) // (self.n_clients)
        print(f"data_per_client = {n_data_per_client}")
        

        # Initializing Clients' datasets
        self.clients = []
        print(f"alpha = {self.alpha}")
        for client in range(self.n_clients):
            # client_prob_per_cluster = np.random.dirichlet(self.alpha*cluster_ratios)                         # Dirichlet distribution for client
            client_prob_per_cluster = np.random.dirichlet(self.alpha*new_cluster_ratios)
            print(f"Creating dataset for client {client}")
            print(client_prob_per_cluster)
            client_data = {}
            # for cluster_idx, cluster_val in enumerate(list(avail_idxs.keys())):                                     # For each cluster
            for cluster_idx, cluster_val in enumerate(list(has_enough_data)):
                num_samples = int(n_data_per_client * client_prob_per_cluster[cluster_idx])                 # Get the number of samples for the cluster
                print(f"{num_samples} samples from cluster {cluster_val}")                                  
                if len(avail_idxs[cluster_val]) < num_samples:                                              # If there are not enough samples in the cluster
                    raise ValueError("Not enough samples in cluster")                                       # Raise an error
                sampled_indexes = np.random.choice(avail_idxs[cluster_val], num_samples, replace=False)     # Sample the indexes for cluster
                cluster_subds = Subset(data_clustered[cluster_val], sampled_indexes)                        # Create a subset of the cluster dataset for the client data
                client_data[cluster_val] = cluster_subds
                avail_idxs[cluster_val] = list(set(avail_idxs[cluster_val]) - set(sampled_indexes))         # Remove the sampled indexes from the available indexes
            client_ds = ConcatDataset([client_data[i] for i in client_data.keys()])        # Concatenate the datasets for the client to create the final client ds
            print(f"final client dataset len: {len(client_ds)}")
            self.clients.append(SingleTaskClient(client_ds, self.global_model, self.device, args))                                     # Create a client using the created ds and add it to the clients list
    
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
                    client.train_once()
            
            # Uplink & Aggregation
            self.aggregate_weights()

            # Record global model performance on client datasets
            val_loss_m, val_loss_s = self.validate_global_model()
            self.mean_loss_history.append(val_loss_m)
            self.std_loss_history.append(val_loss_s)
            print(f"Round {i+1}/{self.n_rounds}, Global model mean validation Loss: {val_loss_m}")
        
        # Save loss history to csv file
        loss_df = pd.DataFrame(list(zip(self.mean_loss_history, self.std_loss_history)), columns=['mean', 'std'])
        loss_df.to_csv(os.path.join(self.save_dir, f'eetf_{self.rand_seed}_global_{self.task}_model_loss_{self.alpha}_{self.sgd_per_epoch}.csv'), index=False)
        
        return self.mean_loss_history, self.std_loss_history
    
    def aggregate_weights(self):
        # Get the weighted average of the model parameters based on client data size
        global_dict = deepcopy(self.global_model.state_dict())
        for key in global_dict.keys():
            # global_dict[key] = torch.stack([client.model.state_dict()[key].float() for client in self.clients], dim=0).mean(dim=0)
            global_dict[key] = torch.stack([client.model.state_dict()[key].float() * len(client.dataloader) for client in self.clients], dim=0).sum(dim=0) / sum([len(client.dataloader) for client in self.clients])

        
        self.global_model.load_state_dict(global_dict)
    
    def validate_global_model(self):
        
        self.global_model.eval()
        client_based_losses = []
        for client in self.clients:
            total_loss = 0
            with torch.no_grad():
                for idx, (image, label) in enumerate(client.dataloader):

                    image = image.to(self.device).float()
                    label = label[self.task].to(self.device).float()

                    _, outputs = self.global_model(image)
                    loss = self.criterion(outputs, label)
                    total_loss += loss.item()
            client_based_losses.append(total_loss/len(client.dataloader))
        
        return np.mean(np.array(client_based_losses)), np.std(np.array(client_based_losses))