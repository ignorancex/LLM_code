import numpy as np, copy
import torch, random
from collections import Counter, defaultdict
from sklearn.utils import resample
from torch.utils.data import DataLoader
import sys, os, json
base_path = os.path.dirname(os.path.realpath(__file__)).split('helper_classes')[0]
sys.path.append(base_path)
from util.weightedloss import WeightedMSELoss
from helper_classes.dataloader import HeadAndRelationBatchLoader, CLPDataLoader
from concept_length_predictors.helper_classes import ConceptLengthPredictor
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
from sklearn.metrics import f1_score, accuracy_score
import time


class Experiment:
    
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.decay_rate = kwargs['decay_rate']
        self.clp = ConceptLengthPredictor(kwargs)
    
    def get_data_idxs(self, data):
        data_idxs = [(self.clp.dataloader.entity_to_idx[t[0]], self.clp.dataloader.relation_to_idx[t[1]], self.clp.dataloader.entity_to_idx[t[2]]) for t in data]
        return data_idxs
    
    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    def get_batch(self, x, y, batch_size, shuffle=True):
        if shuffle:
            indx = list(range(x.shape[0]))
            random.shuffle(indx)
            x, y = x[indx], y[indx]
        if len(x) >= batch_size:
            for i in range(0, x.shape[0]-batch_size+1, batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size]
        else:
            yield x, y
            
    def get_optimizer(self, length_predictor, optimizer='Adam', embedding_model=None):
        if embedding_model is not None:
            if optimizer == 'Adam':
                return torch.optim.Adam(list(length_predictor.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            elif optimizer == 'SGD':
                return torch.optim.SGD(list(length_predictor.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            elif optimizer == 'RMSprop':
                return torch.optim.RMSprop(list(length_predictor.parameters())+list(embedding_model.parameters()), lr=self.kwargs['learning_rate'])
            else:
                raise ValueError
                print('Unsupported optimizer')
        else:
            if optimizer == 'Adam':
                return torch.optim.Adam(length_predictor.parameters(), lr=self.kwargs['learning_rate'])
            elif optimizer == 'SGD':
                return torch.optim.SGD(length_predictor.parameters(), lr=self.kwargs['learning_rate'])
            elif optimizer == 'RMSprop':
                return torch.optim.RMSprop(length_predictor.parameters(), lr=self.kwargs['learning_rate'])
            else:
                raise ValueError
                print('Unsupported optimizer')

    @staticmethod
    def remove_minority_problem_types(data:list, label='target concept length'):
        """
        Function for removing class expressions whose lengths are under-represented
        """
        length_counts = Counter(v[label] for _, v in data)
        mean_length_count = sum(length_counts.values()) // len(length_counts)
        return list(filter(lambda item: length_counts[item[1][label]] >= mean_length_count/5, data))
    
    
    def show_num_learnable_params(self):
        print("*"*20+"Trainable model size"+"*"*20)
        size = sum([p.numel() for p in self.clp.length_predictor.parameters()])
        size_ = 0
        print("Length predictor: ", size)
        if self.clp.embedding_model is not None:
            size_ += sum([p.numel() for p in self.clp.embedding_model.parameters()])
            size += size_
        print("Embedding model: ", size_)
        print("Total: ", size)
        print("*"*20+"Trainable model size"+"*"*20)
        print()
    
    def train(self, data_train, data_test, epochs=200, clp_batch_size=64, tc_batch_size=512, kf_n_splits=10, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False):
        if self.clp.embedding_model is not None:
            Weights = {int(l): 1./torch.sqrt(torch.tensor(c)) for l,c in Counter([v["target concept length"] for _,v in data_train]).items()}
        else:
            Weights = {int(l): 1./torch.sqrt(torch.tensor(c)) for l,c in Counter(data_train[1].tolist()).items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loss weights: ", Weights)
        if self.clp.length_predictor.as_classification:
            W = []
            for l in range(max(Weights.keys())+1):
                if not l in Weights:
                    W.append(torch.tensor(0.))
                else:
                    W.append(Weights[l])
            self.loss = CrossEntropyLoss(weight=torch.Tensor(W).to(device))
        else:
            self.loss = WeightedMSELoss(Weights)
        self.show_num_learnable_params()
        if self.clp.embedding_model is not None and include_embedding_loss:
            triple_data_idxs = self.get_data_idxs(self.clp.dataloader.data)
            head_to_relation_batch = list(DataLoader(
                HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.clp.dataloader.entities)),
                batch_size=tc_batch_size, num_workers=12, shuffle=True))
        
        embeddings = None
        if self.clp.embedding_model is None:
            embeddings = self.clp.get_embedding(embedding_model=None)
            
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")   
        best_performance = 0.
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.clp.length_predictor.name, self.kwargs['path_to_triples'].split("/")[-3]))
        print("#"*50, "\n")
        
        length_predictor = copy.deepcopy(self.clp.length_predictor)
        
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/")
            
        if self.clp.embedding_model is not None:
                embedding_model = copy.deepcopy(self.clp.embedding_model)
        if train_on_gpu:
            length_predictor.cuda()
            if embeddings is None:
                embedding_model.cuda()
        if embeddings is None: 
            opt = self.get_optimizer(length_predictor=length_predictor, optimizer=optimizer, embedding_model=embedding_model)
        else:
            opt = self.get_optimizer(length_predictor=length_predictor, optimizer=optimizer)
        if self.decay_rate:
            self.scheduler = ExponentialLR(opt, self.decay_rate)
        train_losses = []
        Train_loss = []
        Train_acc = []
        if include_embedding_loss:
            tc_iterator = 0
        if record_runtime:
            t0 = time.time()
        Emb = embeddings if embeddings is not None else self.clp.get_embedding(embedding_model)
        if self.clp.embedding_model is None:
            for e in range(epochs):
                tr_preds, tr_targets = [], []
                for x, y in self.get_batch(data_train[0], data_train[1], batch_size=clp_batch_size):
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    if length_predictor.as_classification:
                        y = y.to(torch.long)
                    tr_targets.extend(y.tolist())
                    if(train_on_gpu):
                        x, y = x.cuda(), y.cuda()
                    #ipdb.set_trace()
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        tr_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                        clp_loss = self.loss(y_hat, y)
                    else:
                        tr_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                        clp_loss = self.loss(y_hat.squeeze(), y)
                    # calculate the loss and perform backprop
                    train_losses.append(clp_loss.item())
                    opt.zero_grad()
                    clp_loss.backward()
                    opt.step()
                    if self.decay_rate:
                        self.scheduler.step()
                tr_acc = 100*accuracy_score(tr_preds, tr_targets)
                Train_loss.append(np.mean(train_losses))
                Train_acc.append(tr_acc)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Train loss: {:.4f}...".format(np.mean(train_losses)),
                      "Train acc: {:.2f}%...".format(tr_acc))
                train_losses = []
                weights_clp = copy.deepcopy(length_predictor.state_dict())
                if embeddings is None:
                    weights_emb = copy.deepcopy(embedding_model.state_dict())
                if Train_acc and Train_acc[-1] > best_performance:
                    best_performance = Train_acc[-1]
                    best_weights_clp = weights_clp
                    if embeddings is None:
                        best_weights_emb = weights_emb
        else:
            for e in range(epochs):
                tr_preds, tr_targets = [], []
                for x, y in self.clp.dataloader.load(Emb, data=data_train, batch_size=clp_batch_size, shuffle=True):
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    if include_embedding_loss:
                        head_batch = head_to_relation_batch[tc_iterator%len(head_to_relation_batch)]
                        tc_iterator += 1
                        e1_idx, r_idx, tc_targets = head_batch
                        if train_on_gpu:
                            tc_targets = tc_targets.cuda()
                            r_idx = r_idx.cuda()
                            e1_idx = e1_idx.cuda()
                        if tc_label_smoothing:
                            tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))
                        tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                    if length_predictor.as_classification:
                        y = y.to(torch.long)
                    tr_targets.extend(y.tolist())
                    if(train_on_gpu):
                        x, y = x.cuda(), y.cuda()
                    #ipdb.set_trace()
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        tr_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                        clp_loss = self.loss(y_hat, y)
                    else:
                        tr_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                        clp_loss = self.loss(y_hat.squeeze(), y)
                    # calculate the loss and perform backprop
                    if include_embedding_loss:
                        tclp_loss = 0.5*clp_loss + 0.5*tc_loss
                    else:
                        tclp_loss = clp_loss
                    train_losses.append(tclp_loss.item())
                    opt.zero_grad()
                    tclp_loss.backward()
                    opt.step()
                    if self.decay_rate:
                        self.scheduler.step()
                    Emb = self.clp.get_embedding(embedding_model)
                    tr_acc = 100*accuracy_score(tr_preds, tr_targets)
                    Train_loss.append(np.mean(train_losses))
                    Train_acc.append(tr_acc)
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train loss: {:.4f}...".format(np.mean(train_losses)),
                          "Train acc: {:.2f}%...".format(tr_acc))
                train_losses = []
                weights_clp = copy.deepcopy(length_predictor.state_dict())
                if embeddings is None:
                    weights_emb = copy.deepcopy(embedding_model.state_dict())
                if Train_acc and Train_acc[-1] > best_performance:
                    best_performance = Train_acc[-1]
                    best_weights_clp = weights_clp
                    if embeddings is None:
                        best_weights_emb = weights_emb
        length_predictor.load_state_dict(best_weights_clp)
        if embeddings is None:
            embedding_model.load_state_dict(best_weights_emb)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            length_predictor.eval()
            if embeddings is None:
                embedding_model.eval()
            preds, targets = [],[]
            if self.clp.embedding_model is None:
                for x, y in self.get_batch(data_test[0], data_test[1], batch_size=clp_batch_size, shuffle=False):
                    if length_predictor.as_classification:
                        y = y.to(torch.long)
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    targets.extend(y.tolist())
                    if train_on_gpu:
                        x, y = x.cuda(), y.cuda()
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                    else:
                        preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
            else:
                for x, y in self.clp.dataloader.load(Emb, data=data_test, batch_size=clp_batch_size, shuffle=False):
                    if length_predictor.as_classification:
                        y = y.to(torch.long)
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    targets.extend(y.tolist())
                    if train_on_gpu:
                        x, y = x.cuda(), y.cuda()
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                    else:
                        preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
            test_acc = 100.*accuracy_score(preds, targets)
            print("Test for {}:".format(length_predictor.name))
            print("Test accuracy: ", test_acc)
            f1 = 100.*f1_score(preds, targets, average='macro')
            results_dict.update({"Test acc":test_acc, "Test f1": f1})
            print("Test f1 score: ", f1)
        print("Train accuracy: {}".format(max(Train_acc)))
        print()
        results_dict.update({"Train Max Acc": max(Train_acc), "Train Min Loss": min(Train_loss)})
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/")
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/")
        if embeddings is None:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+embedding_model.name+'_'+length_predictor.name+"_final.json", "w") as file:
                json.dump(results_dict, file, indent=3)
        else:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+length_predictor.name+"_final.json", "w") as file:
                json.dump(results_dict, file, indent=3)
        if save_model:
            torch.save(length_predictor, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+length_predictor.name+"_final.pt")
            if embeddings is None:
                torch.save(embedding_model, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+embedding_model.name+'_'+length_predictor.name+"_final.pt")
            print("{} saved".format(length_predictor.name))
            print()
        if record_runtime:
            duration = time.time()-t0
            runtime_info = {"Length Learner": length_predictor.name,
                           "Number of Epochs": epochs, "Runtime (s)": duration}
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime")
            if embeddings is None:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime/"+"Runtime_"+embedding_model.name+'_'+length_predictor.name+".json", "w") as file:
                    json.dump(runtime_info, file, indent=3)
            else:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Runtime/"+"Runtime_"+length_predictor.name+".json", "w") as file:
                    json.dump(runtime_info, file, indent=3)
        return Train_acc, Train_loss
        
    
    def cross_validate(self, data_train, data_test, epochs=200, clp_batch_size=64, tc_batch_size=512, kf_n_splits=10, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9):
        if self.clp.embedding_model is not None:
            Weights = {int(l): 1./torch.sqrt(torch.tensor(c)) for l,c in Counter([v["target concept length"] for _,v in data_train]).items()}
        else:
            Weights = {int(l): 1./torch.sqrt(torch.tensor(c)) for l,c in Counter(data_train[1].tolist()).items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loss weights: ", Weights)
        if self.clp.length_predictor.as_classification:
            W = []
            for l in range(max(Weights.keys())+1):
                if not l in Weights:
                    W.append(torch.tensor(0.))
                else:
                    W.append(Weights[l])
            self.loss = CrossEntropyLoss(weight=torch.Tensor(W).to(device))
        else:
            self.loss = WeightedMSELoss(Weights)
        if self.clp.embedding_model is not None and include_embedding_loss:
            triple_data_idxs = self.get_data_idxs(self.clp.dataloader.data)
            head_to_relation_batch = list(DataLoader(
                HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(triple_data_idxs), num_e=len(self.clp.dataloader.entities)),batch_size=tc_batch_size, num_workers=12, shuffle=True))
        embeddings = None
        if self.clp.embedding_model is None:
            embeddings = self.clp.get_embedding(embedding_model=None)
            
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print("Training on CPU, it may take long...")
        else:
            print("GPU available !")   
        best_performance = 0.
        print()
        print("#"*50)
        print()
        print("{} starts training on {} data set \n".format(self.clp.length_predictor.name, self.kwargs['path_to_triples'].split("/")[-3]))
        print("#"*50, "\n")
        from sklearn.model_selection import KFold
        Kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=142)
        fold = 0
        All_losses = defaultdict(lambda: [])
        All_acc = defaultdict(lambda: [])
        iterable = data_train if self.clp.embedding_model is not None else list(range(len(data_train[1])))
        for train_index, valid_index in Kf.split(iterable):
            self.show_num_learnable_params()
            length_predictor = copy.deepcopy(self.clp.length_predictor)
            embedding_model = None
            if self.clp.embedding_model is not None:
                embedding_model = copy.deepcopy(self.clp.embedding_model)
            if train_on_gpu:
                length_predictor.cuda()
                if embeddings is None:
                    embedding_model.cuda()
            if embeddings is None: 
                opt = self.get_optimizer(length_predictor=length_predictor, optimizer=optimizer, embedding_model=embedding_model)
            else:
                opt = self.get_optimizer(length_predictor=length_predictor, optimizer=optimizer)
            if self.decay_rate:
                self.scheduler = ExponentialLR(opt, self.decay_rate)
            if self.clp.embedding_model is None:
                x_train, x_valid = data_train[0][train_index], data_train[0][valid_index]
                y_train, y_valid = data_train[1][train_index], data_train[1][valid_index]
            else:
                d_train, d_valid = np.array(data_train,dtype=object)[train_index], np.array(data_train,dtype=object)[valid_index]
            fold += 1
            print("*"*50)
            print("Fold {}/{}:\n".format(fold, kf_n_splits))
            print("*"*50, "\n")
            train_losses = []
            Train_losses = []
            Val_losses = []
            Train_acc = []
            Val_acc = []
            
            if self.clp.embedding_model is not None and include_embedding_loss:
                tc_iterator = 0
            Emb = embeddings if embeddings is not None else self.clp.get_embedding(embedding_model)
            if self.clp.embedding_model is None:
                for e in range(epochs):
                    tr_preds, tr_targets = [], []
                    for x, y in self.get_batch(x_train, y_train, batch_size=clp_batch_size):
                        if self.clp.learner_name == "MLP":
                            x = x.mean(1)
                        if length_predictor.as_classification:
                            y = y.to(torch.long)
                        tr_targets.extend(y.tolist())
                        if(train_on_gpu):
                            x, y = x.cuda(), y.cuda()
                        y_hat = length_predictor(x)
                        if length_predictor.as_classification:
                            tr_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                            clp_loss = self.loss(y_hat, y)
                        else:
                            tr_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                            clp_loss = self.loss(y_hat.squeeze(), y)
                        # calculate the loss and perform backprop
                        
                        train_losses.append(clp_loss.item())
                        opt.zero_grad()
                        clp_loss.backward()
                        opt.step()
                        if self.decay_rate:
                            self.scheduler.step()
                    tr_acc = 100*accuracy_score(tr_preds, tr_targets)
                    # Get validation loss
                    val_losses = []
                    length_predictor.eval()
                    if embeddings is None:
                        embedding_model.eval()
                    val_targets, val_preds = [], []
                    for x, y in self.get_batch(x_valid, y_valid, batch_size=clp_batch_size):
                        if length_predictor.as_classification:
                            y = y.to(torch.long)
                        val_targets.extend(y.tolist())
                        if self.clp.learner_name == "MLP":
                            x = x.mean(1)
                        if(train_on_gpu):
                            x, y = x.cuda(), y.cuda()
                        y_hat = length_predictor(x)
                        if length_predictor.as_classification:
                            val_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                            val_loss = self.loss(y_hat, y)
                        else:
                            val_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                            val_loss = self.loss(y_hat.squeeze(), y)
                        val_losses.append(val_loss.item())
                    length_predictor.train() # reset to train mode after iterationg through validation data
                    if embeddings is None:
                        embedding_model.train()
                    Train_losses.append(np.mean(train_losses))
                    Val_losses.append(np.mean(val_losses))
                    val_acc = 100*accuracy_score(val_preds, val_targets)
                    Val_acc.append(val_acc)
                    Train_acc.append(tr_acc)
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train loss: {:.4f}...".format(np.mean(train_losses)),
                          "Val loss: {:.4f}...".format(np.mean(val_losses)),
                          "Train acc: {:.2f}%...".format(tr_acc),
                          "Val acc: {:.2f}%".format(val_acc))
                    train_losses = []
                weights_clp = copy.deepcopy(length_predictor.state_dict())
                if embeddings is None:
                    weights_emb = copy.deepcopy(embedding_model.state_dict())
                if Val_acc and Val_acc[-1] > best_performance:
                    best_performance = Val_acc[-1]
                    best_weights_clp = weights_clp
                    if embeddings is None:
                        best_weights_emb = weights_emb
                All_losses["train"].append(Train_losses)
                All_losses["val"].append(Val_losses)
                All_acc["train"].append(Train_acc)
                All_acc["val"].append(Val_acc)
            else:
                for e in range(epochs):
                    tr_targets, tr_preds = [], []
                    for x, y in self.clp.dataloader.load(Emb, data=d_train, batch_size=clp_batch_size, shuffle=True):
                        if self.clp.learner_name == "MLP":
                            x = x.mean(1)
                        if self.clp.embedding_model is not None and include_embedding_loss:
                            head_batch = head_to_relation_batch[tc_iterator%len(head_to_relation_batch)]
                            tc_iterator += 1
                            e1_idx, r_idx, tc_targets = head_batch
                            if train_on_gpu:
                                tc_targets = tc_targets.cuda()
                                r_idx = r_idx.cuda()
                                e1_idx = e1_idx.cuda()

                            if tc_label_smoothing:
                                tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))

                            tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                        if length_predictor.as_classification:
                            y = y.to(torch.long)
                        tr_targets.extend(y.tolist())
                        if(train_on_gpu):
                            x, y = x.cuda(), y.cuda()
                        y_hat = length_predictor(x)
                        tr_total_dpoints_before_eval += len(y)
                        if length_predictor.as_classification:
                            tr_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                            clp_loss = self.loss(y_hat, y)
                        else:
                            tr_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                            clp_loss = self.loss(y_hat.squeeze(), y)
                        # calculate the loss and perform backprop
                        if self.clp.embedding_model is not None and include_embedding_loss:
                            tclp_loss = 0.5*clp_loss + 0.5*tc_loss
                        else:
                            tclp_loss = clp_loss
                        train_losses.append(tclp_loss.item())
                        opt.zero_grad()
                        tclp_loss.backward()
                        opt.step()
                        if self.decay_rate:
                            self.scheduler.step()
                        if self.clp.embedding_model is not None:
                            Emb = embeddings if embeddings is not None else self.clp.get_embedding(embedding_model)
                    tr_acc = 100*accuracy_score(tr_preds, tr_targets)
                    # Get validation loss
                    val_losses = []
                    length_predictor.eval()
                    if embeddings is None:
                        embedding_model.eval()
                    val_preds, val_targets = [], []
                    for x, y in self.clp.dataloader.load(Emb, data=d_valid, batch_size=clp_batch_size, shuffle=False):
                        if length_predictor.as_classification:
                            y = y.to(torch.long)
                        if self.clp.learner_name == "MLP":
                            x = x.mean(1)
                        val_targets.extend(y.tolist())
                        if(train_on_gpu):
                            x, y = x.cuda(), y.cuda()
                        if self.clp.embedding_model is not None and include_embedding_loss:
                            head_batch = random.choice(head_to_relation_batch)
                            e1_idx, r_idx, tc_targets = head_batch
                            if train_on_gpu:
                                tc_targets = tc_targets.cuda()
                                r_idx = r_idx.cuda()
                                e1_idx = e1_idx.cuda()
                            if tc_label_smoothing:
                                tc_targets = ((1.0 - tc_label_smoothing) * tc_targets) + (1.0 / tc_targets.size(1))
                            tc_loss = embedding_model.forward_head_and_loss(e1_idx, r_idx, tc_targets)
                        y_hat = length_predictor(x)
                        if length_predictor.as_classification:
                            val_preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                            val_loss = self.loss(y_hat, y)
                        else:
                            val_preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
                            val_loss = self.loss(y_hat.squeeze(), y)

                        if self.clp.embedding_model is not None and include_embedding_loss:
                            tclp_loss = 0.5*val_loss + 0.5*tc_loss
                        else:
                            tclp_loss = val_loss
                        val_losses.append(tclp_loss.item())
                    length_predictor.train() # reset to train mode after iterationg through validation data
                    if embeddings is None:
                        embedding_model.train()
                    Train_losses.append(np.mean(train_losses))
                    Val_losses.append(np.mean(val_losses))
                    val_acc = 100.*accuracy_score(val_preds, val_targets)
                    Val_acc.append(val_acc)
                    Train_acc.append(tr_acc)
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Train loss: {:.4f}...".format(np.mean(train_losses)),
                          "Val loss: {:.4f}...".format(np.mean(val_losses)),
                          "Train acc: {:.2f}%...".format(tr_acc),
                          "Val acc: {:.2f}%".format(val_acc))
                    train_losses = []
                    tr_total_dpoints_before_eval = 0.
                weights_clp = copy.deepcopy(length_predictor.state_dict())
                if embeddings is None:
                    weights_emb = copy.deepcopy(embedding_model.state_dict())
                if Val_acc and Val_acc[-1] > best_performance:
                    best_performance = Val_acc[-1]
                    best_weights_clp = weights_clp
                    if embeddings is None:
                        best_weights_emb = weights_emb
                All_losses["train"].append(Train_losses)
                All_losses["val"].append(Val_losses)
                All_acc["train"].append(Train_acc)
                All_acc["val"].append(Val_acc)
                
        min_num_steps = min(min([len(l) for l in All_losses['train']]), min([len(l) for l in All_losses['val']]))

        train_folds_loss = np.array([l[:min_num_steps] for l in All_losses["train"]])
        val_folds_loss = np.array([l[:min_num_steps] for l in All_losses["val"]])
        train_folds_acc = np.array([l[:min_num_steps] for l in All_acc["train"]])
        val_folds_acc = np.array([l[:min_num_steps] for l in All_acc["val"]])
        
        train_l = train_folds_loss.mean(0)
        val_l = val_folds_loss.mean(0)
        t_acc = train_folds_acc.mean(0)
        v_acc = val_folds_acc.mean(0)
        
        del All_losses, All_acc        
        length_predictor.load_state_dict(best_weights_clp)
        if embeddings is None:
            embedding_model.load_state_dict(best_weights_emb)
        results_dict = dict()
        if test:
            print()
            print("#"*50)
            print("Testing the model ....................")
            print()
            length_predictor.eval()
            if embeddings is None:
                embedding_model.eval()
            preds, targets = [], []
            if self.clp.embedding_model is None:
                for x, y in self.get_batch(data_test[0], data_test[1], batch_size=clp_batch_size, shuffle=False):
                    if train_on_gpu:
                        x = x.cuda()
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    targets.extend(y.tolist())
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                    else:
                        preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
            else:
                for x, y in self.clp.dataloader.load(Emb, data=data_test, batch_size=clp_batch_size, shuffle=False):
                    if length_predictor.as_classification:
                        y = y.to(torch.long)
                    if train_on_gpu:
                        x, y = x.cuda(), y.cuda()
                    if self.clp.learner_name == "MLP":
                        x = x.mean(1)
                    y_hat = length_predictor(x)
                    if length_predictor.as_classification:
                        preds.extend(y_hat.cpu().detach().argmax(1).tolist())
                    else:
                        preds.extend(torch.round(y_hat.cpu().detach()).squeeze().tolist())
            test_acc = 100.*accuracy_score(preds, targets)
            print("Test for {}:".format(length_predictor.name))
            print("Test accuracy: ", test_acc)
            f1 = 100*f1_score(preds, targets, average='macro')
            results_dict.update({"Test acc":test_acc, "Test f1": f1})
            print("Test f1 score: ", f1)

        print("Train avg acc: {}, Val avg acc: {}".format(max(t_acc), max(v_acc)))
        print()
        results_dict.update({"Train Avg Acc": max(t_acc), "Train Avg Loss": min(train_l), "Val Avg Acc": max(v_acc), "Val Avg Loss": min(val_l)})
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/")
        if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"):
            os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/")
        if embeddings is None:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+embedding_model.name+'_'+length_predictor.name+".json", "w") as file:
                json.dump(results_dict, file, indent=3)
        else:
            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Results/"+"Train_Results_"+length_predictor.name+".json", "w") as file:
                json.dump(results_dict, file, indent=3)

        if save_model:
            torch.save(length_predictor, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+length_predictor.name+".pt")
            if embeddings is None:
                torch.save(embedding_model, self.kwargs['path_to_triples'].split("Triples")[0]+"Model_weights/"+embedding_model.name+'_'+length_predictor.name+".pt")
            print("{} saved".format(length_predictor.name))
        return t_acc, v_acc, train_l, val_l, train_folds_loss, val_folds_loss, train_folds_acc, val_folds_acc
    
    
    def train_and_eval(self, data_train, data_test, epochs=200, clp_batch_size=64, tc_batch_size=512, kf_n_splits=10, cross_validate=False, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False):
        
        """
        function for training a concept length learner in DL KGs
        
        key args
        -> cll_batch_size: batch_size for the concept learner training (clp: concept length predictor)
        -> tc_batch_size: batch_size for the training the embedding model (tc: triple classification)
        key args
        """
        if cross_validate:
            return self.cross_validate(data_train, data_test, epochs, clp_batch_size, tc_batch_size,
                                       kf_n_splits, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing)

        else:
            return self.train(data_train, data_test, epochs, clp_batch_size, tc_batch_size,
                    kf_n_splits, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime)
            
            
    def train_all_nets(self, List_nets, data_train, data_test, epochs=200, clp_batch_size=64, tc_batch_size=512, kf_n_splits=10, cross_validate=False, test=False, save_model = False, include_embedding_loss=False, optimizer = 'Adam', tc_label_smoothing=0.9, record_runtime=False):
        if self.clp.embedding_model is None:
            embeddings = self.clp.get_embedding(embedding_model=None)
            print("Loading train and validate data\n")
            data_train = self.clp.dataloader.load(embeddings, data=data_train, shuffle=True)
            data_test = self.clp.dataloader.load(embeddings, data=data_test, shuffle=False)
            print("Done loading train and validate data\n")
        Training_data = dict()
        Validation_data = dict()
        kf_data = {"train": defaultdict(dict), "val": defaultdict(dict)}
        if cross_validate:
            for net in List_nets:
                self.clp.learner_name = net
                self.clp.refresh()
                t_acc, v_acc, train_l, val_l, train_folds_loss, val_folds_loss, train_folds_acc, val_folds_acc = self.train_and_eval(data_train, data_test, epochs, clp_batch_size, tc_batch_size, kf_n_splits, cross_validate, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime)
                Training_data.setdefault("acc", []).append(list(t_acc))
                Training_data.setdefault("loss", []).append(list(train_l))
                Validation_data.setdefault("acc", []).append(list(v_acc))
                Validation_data.setdefault("loss", []).append(list(val_l))
                
                kf_data["train"][net]["loss"] = train_folds_loss.tolist()
                kf_data["train"][net]["acc"] = train_folds_acc.tolist()
                kf_data["val"][net]["loss"] = val_folds_loss.tolist()
                kf_data["val"][net]["acc"] = val_folds_acc.tolist()
                
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/")

            with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_kf.json", "w") as plot_file:
                    json.dump(kf_data, plot_file, indent=3)
                
            #if len(List_nets) > 1:
            #    with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_with_val.json", "w") as plot_file:
            #        json.dump(Training_data, plot_file, indent=3)
            #else:
            #    with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_with_val_single.json", "w") as plot_file:
            #        json.dump(Training_data, plot_file, indent=3)
            
        else:
            for net in List_nets:
                self.clp.learner_name = net
                self.clp.refresh()
                t_acc, train_l = self.train_and_eval(data_train, data_test, epochs, clp_batch_size, tc_batch_size, kf_n_splits, cross_validate, test, save_model, include_embedding_loss, optimizer, tc_label_smoothing, record_runtime)
                Training_data.setdefault("acc", []).append(t_acc)
                Training_data.setdefault("loss", []).append(train_l)
                
            if not os.path.exists(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/"):
                os.mkdir(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/")
            if len(List_nets) > 1:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_no_val.json", "w") as plot_file:
                    json.dump(Training_data, plot_file, indent=3)
            else:
                with open(self.kwargs['path_to_triples'].split("Triples")[0]+"Plot_data/plot_data_no_val_single.json", "w") as plot_file:
                    json.dump(Training_data, plot_file, indent=3)
