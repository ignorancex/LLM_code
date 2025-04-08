import os.path as osp
import numpy
import torch
import torch.nn as nn
import warnings
from sklearn.model_selection import KFold
import copy
from torch.utils.data import DataLoader, Subset
import copy
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from models.ResNet18 import ResNet10
import constants
from utils.utils import *

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on ")

random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

class SGRepClassifier():
    def __init__(self, run_index, client_index, is_replica=False, replica_index=0, replica_depth=0, fed_alg="FedAvg", no_classes=2, training=True):
        self.generator_save_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights/weight_run{run_index}_client{client_index}_replica{replica_index}.model'
        self.is_replica = is_replica

        self.replicas_list = []
        if fed_alg == "FedRep" and training:
            if replica_depth < constants.MAX_REPLICA_DEPTH:
                for new_replica_index in range(constants.NUMBER_REPLICAS):
                    self.replicas_list.append(SGRepClassifier(run_index=run_index, client_index=client_index, is_replica=True, 
                                                            replica_index=new_replica_index, replica_depth=replica_depth + 1, 
                                                            fed_alg=fed_alg, no_classes=no_classes))

        self.client_index = client_index
        self.replica_index = replica_index
        self.replica_depth = replica_depth

        self.fed_alg = fed_alg
        self.no_classes = no_classes

        self.best_train_loss = float('inf')
        self.source_test = None
        self.best_predicted_test = None
        self.best_target = None

        self.acc = None
        self.spec = None
        self.sens = None
        self.f1 = None

        self.no_epochs_trained = 0

        self.model = ResNet10(no_classes)
        self.replica_aggregated_model = copy.deepcopy(self.model)

        self.model.to(device)
        self.replica_aggregated_model.to(device)

        if constants.OPTIM == "sgd":
            self.generator_optimizer = torch.optim.SGD(self.model.parameters(), lr=constants.LR)
        elif constants.OPTIM == "adam":
            self.generator_optimizer = torch.optim.Adam(self.model.parameters(), lr=constants.LR)

        self.global_model_param = None

        old_grad = copy.deepcopy(self.model)
        for param in old_grad.parameters():
            param.data = torch.zeros_like(param.data)
        self.old_grad = get_mdl_params_resnet([old_grad])[0]

    def train(self):
        self.model.train()

        global_loss_train = []
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

        for epochs in range(constants.NUM_LOCAL_EPOCHS):
            global_loss_train_current_epoch = []

            with torch.autograd.set_detect_anomaly(True):
                for inputs, targets in self.data_loader:
                    self.generator_optimizer.zero_grad()

                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    # print(f"outputs shape: {outputs.shape}")
                    torch.cuda.empty_cache()

                    targets = targets.to(device)
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)
                    torch.cuda.empty_cache()

                    if self.fed_alg == "FedDyn":
                        # Get linear penalty on the current parameter estimates
                        local_par_list = None
                        for name, param in self.model.named_parameters():
                            if "conv" in name:
                                if not isinstance(local_par_list, torch.Tensor):
                                    # Initially nothing to concatenate
                                    local_par_list = param.reshape(-1)
                                else:
                                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                        old_grad_tensor = torch.tensor(self.old_grad, dtype=torch.float32, device=device)
                        global_model_param_tensor = torch.tensor(self.global_model_param, dtype=torch.float32, device=device)
                        loss_algo = constants.ALPHA_COEF / 2 * torch.norm(local_par_list - global_model_param_tensor, 2)
                        loss_algo -= torch.dot(local_par_list, old_grad_tensor)
                        loss = loss + loss_algo

                    torch.cuda.empty_cache()
                    
                    loss.backward()

                    self.generator_optimizer.step()
                    torch.cuda.empty_cache()

                    global_loss_train_current_epoch.append(loss.detach().cpu().numpy())
                    torch.cuda.empty_cache()

                global_loss_train.append(np.mean(global_loss_train_current_epoch))


        if self.fed_alg == "FedDyn":
            local_param_list = get_mdl_params_resnet([self.model])[0]
            self.old_grad = self.old_grad - constants.ALPHA_COEF * (local_param_list - self.global_model_param)

        torch.cuda.empty_cache()

        return global_loss_train

    def test(self, test_data_loader):
        self.model.load_state_dict(torch.load(self.generator_save_path), strict=False)
        self.model.eval()

        y_true = torch.tensor([])
        y_score = torch.tensor([])

        with torch.no_grad():
            for inputs, targets in test_data_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)

                targets = targets.squeeze()
                outputs = outputs.softmax(dim=-1)

                y_true = torch.cat((y_true, targets), 0)
                outputs = outputs.detach().cpu().clone()
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.numpy().astype(int)
            y_score = y_score.detach().numpy()
            predicted_classes = np.argmax(y_score, axis=1)

            accuracy = np.mean(predicted_classes == y_true)

            # Compute confusion matrix
            cm = confusion_matrix(y_true, predicted_classes)

            # Compute sensitivity/recall for each class
            sensitivities = []
            for i in range(self.no_classes):
                tp = cm[i, i]
                fn = sum(cm[i, :]) - tp
                sensitivity = tp / (tp + fn)
                sensitivities.append(sensitivity)
            sensitivity = sum(sensitivities) / len(sensitivities)

            # Compute specificity for each class
            specificities = []
            for i in range(self.no_classes):
                tn = sum(sum(cm)) - sum(cm[i, :]) - sum(cm[:, i]) + cm[i, i]
                fp = sum(cm[:, i]) - cm[i, i]
                specificity = tn / (tn + fp)
                specificities.append(specificity)
            specificity = sum(specificities) / len(specificities)

            # Compute F1 score for each class
            f1_scores = []
            for i in range(self.no_classes):
                precision = precision_score(y_true == i, predicted_classes == i)
                recall = recall_score(y_true == i, predicted_classes == i)
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
            f1 = sum(f1_scores) / len(f1_scores)

        self.acc = accuracy
        self.spec = specificity
        self.sens = sensitivity
        self.f1 = f1

        return accuracy
    
    def run_one_round(self):
        print(f'Starting round Client {self.client_index} Replica {self.replica_index} Depth {self.replica_depth}')

        global_loss_train = self.train()
        self.no_epochs_trained += constants.NUM_LOCAL_EPOCHS

        print(f'Epochs trained: {self.no_epochs_trained}')

        # Run one round replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].run_one_round()

        # Aggregate the replicas
        if constants.AGGREGATION == "diversity" and len(self.replicas_list) > 0:
            self.aggregate_diversity()

            # Update the new model parameters
            model_state = self.model.state_dict()

            for name, param in self.model.named_parameters():
                if "conv" in name:
                    model_updates = [copy.deepcopy(self.model.state_dict()[name].data), copy.deepcopy(self.replica_aggregated_model.state_dict()[name].data)]
                    averaged_update = torch.stack(model_updates).mean(dim=0)
                    model_state[name] = averaged_update

            self.model.load_state_dict(model_state)
        else:
            self.aggregate_replicas()

        return global_loss_train
    
    def aggregate_replicas(self):
        model_state = self.model.state_dict()

        weight = 1. / (1 + constants.NUMBER_REPLICAS)
        for name, param in self.model.named_parameters():
            if "conv" in name:
                replicas_updates = [weight * copy.deepcopy(replica.get_replica_aggregated_model().state_dict()[name].data) for replica in self.replicas_list]
                replicas_updates.append(weight * copy.deepcopy(self.model.state_dict()[name].data))
                averaged_update = torch.stack(replicas_updates).sum(dim=0)
                model_state[name] = averaged_update

        # Update the replica aggregated model with the aggregated parameters
        self.model.load_state_dict(model_state)

    def aggregate_diversity(self):
        normalized_weights = []
        if constants.DIV_METRIC == "l2":
            l2_norms_replicas = [l2_norm_models(replica.get_generator(), self.model) for replica in self.replicas_list]
            normalized_weights = compute_weight_from_l2_norm(l2_norms_replicas)
        elif constants.DIV_METRIC == "kl":
            kl_div_replicas = [kl_divergence_models(replica.get_generator(), self.model) for replica in self.replicas_list]
            normalized_weights = compute_weights_from_kl(kl_div_replicas)

        # Aggregate model parameters based on the weights
        model_state = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if "conv" in name:
                client_updates = [normalized_weights[i] * copy.deepcopy(replica.get_generator().state_dict()[name].data) for i, replica in enumerate(self.replicas_list)]
                averaged_update = torch.stack(client_updates).sum(dim=0)
                model_state[name] = averaged_update

        # Update the replica aggregated model with the aggregated parameters
        self.replica_aggregated_model.load_state_dict(model_state)

    def save_model(self):
        torch.save(self.model.state_dict(), self.generator_save_path)

    def set_generator_saving_path(self, new_path):
        self.generator_save_path = new_path
    
    def get_source_target_prediction(self):
        return self.source_test, self.best_predicted_test, self.best_target
    
    def get_accuracy(self):
        return self.acc
    
    def get_sensitivity(self):
        return self.sens
    
    def get_specificity(self):
        return self.spec
    
    def get_f1(self):
        return self.f1
    
    def get_generator(self):
        return self.model
    
    def get_replica_aggregated_model(self):
        return self.replica_aggregated_model
    
    def get_replicas_list(self):
        return self.replicas_list

    def get_old_grad(self):
        return self.old_grad
    
    def initialize_model_parameters(self, global_model_state_dict):
        self.model.load_state_dict(global_model_state_dict)
        self.replica_aggregated_model.load_state_dict(global_model_state_dict)

        # Set training dataset for replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].initialize_model_parameters(global_model_state_dict)

    def update_parameters(self, global_model):
        generator_new_state_dict = copy.deepcopy(self.model.state_dict())
        for param_name in self.model.state_dict():
            if "conv" in param_name:
                generator_new_state_dict[param_name].data = copy.deepcopy(global_model.state_dict()[param_name])
                
        self.model.load_state_dict(generator_new_state_dict)

        for params in self.model.parameters():
            params.requires_grad = True

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].update_parameters(global_model)

    def set_global_model_params(self, global_model_param):
        self.global_model_param = global_model_param

        # Set global model params for replicas
        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                self.replicas_list[replica].set_global_model_params(global_model_param)

    def set_training_data(self, train_dataset, no_perturbed_samples = 0, offset_perturbed_samples = 0, stratified = False):
        train_dataset = copy.deepcopy(train_dataset)

        if self.is_replica:
            if offset_perturbed_samples >= len(train_dataset):
                offset_perturbed_samples = offset_perturbed_samples % len(train_dataset)

            if stratified:
                # Iterate over the dataset and check if the label matches the target label
                label_indices_dict = {}
                for i in range(len(train_dataset)):
                    _, label = train_dataset[i]  # Assuming each item in the dataset is a tuple of (data, label)
                    label = label[0]
                    if label not in label_indices_dict:
                        label_indices_dict[label] = []
                    label_indices_dict[label].append(i)

                # bincount
                max_label = max(label_indices_dict.keys())  # Find the maximum label
                label_counts = [0] * (max_label + 1)  # Initialize a list of counts
                for label, indices in label_indices_dict.items():
                    label_counts[label] = len(indices)  # Set the count for the corresponding label
                label_counts = np.array(label_counts)
                print(f"Label counts: {label_counts}")
                label_distribution = label_counts / len(train_dataset)
                # offset_per_label = [int(offset_perturbed_samples * distribution) for distribution in label_distribution]

                # no.of removed samples for each class
                class_subset_sizes = (label_distribution * no_perturbed_samples).astype(int)

                offset_per_label = [int(offset_perturbed_samples * distribution) for distribution in label_distribution]
                for i in range(len(label_counts)):
                    if i < no_perturbed_samples % len(label_counts):
                        class_subset_sizes[i] = class_subset_sizes[i] + 1
                        offset_per_label[i] = offset_per_label[i] + 1

                replica_indices = []
                for label in range(len(label_counts)):
                    replica_indices_one_label = []
                    replica_indices_1 = [label_indices_dict[label][i] for i in range(0, offset_per_label[label])]
                    replica_indices_2 = [label_indices_dict[label][i] for i in range(offset_per_label[label] + class_subset_sizes[label], len(label_indices_dict[label]))]
                    replica_indices_one_label = replica_indices_1 + replica_indices_2
                    replica_indices = replica_indices + replica_indices_one_label
            else:
                replica_indices_1 = [i for i in range(0, offset_perturbed_samples)]
                replica_indices_2 = [i for i in range(offset_perturbed_samples + no_perturbed_samples, len(train_dataset))]
                replica_indices = replica_indices_1 + replica_indices_2

            train_dataset = Subset(train_dataset, replica_indices)

            # Iterate over the dataset and check if the label matches the target label
            label_indices_dict = {}
            for i in range(len(train_dataset)):
                _, label = train_dataset[i]
                label = label[0]
                if label not in label_indices_dict:
                    label_indices_dict[label] = []
                label_indices_dict[label].append(i)

            # bincount
            max_label = max(label_indices_dict.keys())
            label_counts = [0] * (max_label + 1)
            for label, indices in label_indices_dict.items():
                label_counts[label] = len(indices)
            label_counts = np.array(label_counts)
        
        self.data_loader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)

        if len(self.replicas_list) > 0:
            for replica in range(constants.NUMBER_REPLICAS):
                no_perturbed_samples_next_depth = int((constants.NUMBER_PERTURBED_SAMPLES / 100.0) * len(train_dataset))
                self.replicas_list[replica].set_training_data(train_dataset, no_perturbed_samples = no_perturbed_samples_next_depth, 
                                                                offset_perturbed_samples = replica * no_perturbed_samples_next_depth, 
                                                                stratified=stratified)
    
    def get_next_data_loader_batch(self):
        return next(iter(self.data_loader))