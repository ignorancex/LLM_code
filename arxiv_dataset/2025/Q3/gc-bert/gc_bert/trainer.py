from functools import partial
import time
import os
import pickle

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, NLLLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gc_bert.utils import to_torch_sparse, vectorize_texts
from gc_bert.monitor import Monitor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL_NAME = "bert-base-uncased"


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=5e-5,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
        save_dir='saved_models',
        run_name='training',
    ):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fun = loss_fun
        self.logger = logger
        self.save_dir = save_dir
        self.run_name = run_name

        self.train_monitor = Monitor(logger, dataset.num_labels, "train")
        self.valid_monitor = Monitor(logger, dataset.num_labels, "valid")
        self.test_monitor = Monitor(logger, dataset.num_labels, "test")
        self.optimizer = None
        self.lr_scheduler = None

    def prepare(self):
        if self.optimizer is None or self.lr_scheduler is None:
            self.create_optimizer()

    def create_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # 'min' for loss; 'max' for acc
            patience=40,
            factor=0.1,  # new_lr = lr * factor
        )

    def get_dataloader(self, mode="train"):
        self.dataset.change_mode(mode=mode)
        return DataLoader(
            self.dataset, batch_size=32, shuffle=True if mode == "train" else False,
        )

    def validate(self, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for X, y in loader:
                output = self.model(X)
                preds = torch.argmax(output, dim=-1)
                loss = self.loss_fun(preds, y)
                monitor.update(preds, y, loss)
        return monitor.emit(time=round(time.time() - t, 2))

    def train(self, epochs):
        self.prepare()
        
        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for X, y in self.get_dataloader(mode="train"):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss_fun(output, y)
                loss.mean().backward()
                self.optimizer.step()

                preds = torch.argmax(output, dim=-1)
                self.train_monitor.update(preds, y, loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=round(time.time() - t, 2),
            )
            valid_res = self.validate(mode="valid")
            self.lr_scheduler.step(valid_res["loss"])
        
        self.validate(mode='test', epoch=epoch)
        return self.model

    def save(self, suffix = ''):
        with open(os.path.join(self.save_dir, self.run_name, 'model_' + suffix + '.pt'), 'wb') as f:
            torch.save(self.model.state_dict(), f)


class GNNTrainer(Trainer):
    def __init__(
        self, model, dataset, logger, loss_fun=NLLLoss(reduction="none"), **kwargs
    ):
        super().__init__(model, dataset, logger, **kwargs)
        self.loss_fun = loss_fun

    def prepare(self):
        super().prepare()
        if self.dataset.G is None:
            self.dataset.create_graph()
        if self.dataset.adj is None:
            self.dataset.create_adj_matrix(to_sparse=True)
        self.X = vectorize_texts(
            self.dataset.df.text.tolist(), to_sparse=False
        ).to(DEVICE)
        self.adj = to_torch_sparse(self.dataset.adj).to(DEVICE)
        self.edge_idx = self.dataset.edge_idx.to(DEVICE)

    def validate(self, mode="valid", epoch=None):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            y = self.dataset.labels
            output = self.model(self.X, self.edge_idx)
            output = output[self.dataset.mask]
            loss = self.loss_fun(output, y.to(DEVICE))
            preds = torch.argmax(output, dim=-1)
            monitor.update(preds, y, loss)
        return monitor.emit(epoch=epoch, time=round(time.time() - t, 3),)

    def train(self, epochs):

        self.prepare()
        best_val_acc = 0
        
        for epoch in range(epochs):
            t = time.time()

            self.model.train()
            self.dataset.change_mode("train")
            y = self.dataset.labels
            self.optimizer.zero_grad()

            output = self.model(self.X, self.edge_idx)
            output = output[self.dataset.mask]
            loss = self.loss_fun(output, y.to(DEVICE))
            loss.mean().backward()
            self.optimizer.step()

            preds = torch.argmax(output, dim=-1)
            self.train_monitor.update(preds, y, loss)

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=round(time.time() - t, 3),
            )

            valid_res = self.validate(epoch=epoch, mode="valid")
            self.lr_scheduler.step(valid_res["loss"])
            
            if valid_res['acc'] > best_val_acc:
                best_val_acc = valid_res['acc']
                self.save(suffix=f'_epoch={epoch}')
                self.validate(mode='test', epoch=epoch)


        self.validate(mode='test', epoch=epoch)
        return self.model


class BERTTrainer(Trainer):
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=0.00005,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
        **kwargs
    ):
        super().__init__(model, dataset, logger, lr, weight_decay, loss_fun, **kwargs)
        

    def prepare(self):
        super().prepare()
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.tokenizer = partial(
            tokenizer,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        self.dataset.transform = self.tokenizer

    def validate(self, epoch=-1, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for X, y in loader:
                output = self.model(
                    input_ids=X["input_ids"].squeeze().to(DEVICE),
                    token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                    attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    labels=y.to(DEVICE),
                )
                preds = torch.argmax(output['logits'], dim=-1)
                monitor.update(preds, y, output['loss'])
        return monitor.emit(time=round(time.time() - t, 2), epoch=epoch)

    def train(self, epochs):
        self.prepare()
        best_val_acc = 0
        
        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for i, (X, y) in enumerate(self.get_dataloader(mode="train")):
                if i == 150:
                    break
                self.optimizer.zero_grad()
                output = self.model(
                    input_ids=X["input_ids"].squeeze().to(DEVICE),
                    token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                    attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    labels=y.to(DEVICE),
                )
                output['loss'].backward()
                preds = torch.argmax(output['logits'], dim=-1)
                self.optimizer.step()
                self.train_monitor.update(preds, y, output['loss'])

            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=round(time.time() - t, 2),
            )
            torch.cuda.empty_cache()

            valid_res = self.validate(epoch=epoch, mode="valid")
            self.lr_scheduler.step(valid_res["loss"])
            
            if valid_res['acc'] > best_val_acc:
                best_val_acc = valid_res['acc']
                self.save(suffix=f'_epoch={epoch}')
                self.validate(mode='test', epoch=epoch)
        
        self.validate(mode='test', epoch=epoch)
        return self.model


class GraphBERTTrainer(Trainer):
    def __init__(
        self,
        model,
        dataset,
        logger,
        lr=0.00005,
        weight_decay=0.01,
        loss_fun=BCEWithLogitsLoss(reduction="none"),
        freeze_bert=False,
        **kwargs
    ):
        super().__init__(model, dataset, logger, lr, weight_decay, loss_fun, **kwargs)
        self.adj = None
        self.X = None
        self.tokenizer = None
        self.freeze_bert = freeze_bert

    def fill_text_repr(self):
         with torch.no_grad():
            for mode in ['train', 'valid', 'test']:
                for X, _, idx in self.get_dataloader(mode=mode):
                    output = self.model.bert(
                        input_ids=X["input_ids"].squeeze().to(DEVICE),
                        token_type_ids=X["token_type_ids"].squeeze().to(DEVICE),
                        attention_mask=X["attention_mask"].squeeze().to(DEVICE),
                    )
                    self.model.T[idx, :] = output['last_hidden_state']

    def fill_node_repr(self):
        # precalculate N using GNN
        with torch.no_grad():
            self.model.N = nn.parameter.Parameter(self.model.gnn(self.model.T, self.edge_idx), requires_grad=False)

        # OR if you have precalculated graph_embeddings you can load it here
        # with open('graph_embeddings.pkl', 'rb') as f:
        #     N = torch.tensor(pickle.load(f)).to(DEVICE)

        # OR just define N
        # self.model.N = nn.parameter.Parameter(N, requires_grad=False)

        # OR create random N
        # self.model.N = nn.parameter.Parameter(
        #     torch.rand_like(self.model.N, device='cuda'), requires_grad=False
        # )

    def prepare(self):
        super().prepare()
        if self.dataset.G is None:
            self.dataset.create_graph()
        if self.dataset.adj is None:
            self.dataset.create_adj_matrix(to_sparse=True)
        self.X = vectorize_texts(
            self.dataset.df.text.tolist(), to_sparse=False
        ).to(DEVICE)
        self.adj = to_torch_sparse(self.dataset.adj).to(DEVICE)
        self.edge_idx = self.dataset.edge_idx.to(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<GC>']})
        self.model.bert.resize_token_embeddings(len(tokenizer))

        self.tokenizer = partial(
            tokenizer,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        self.dataset.transform = self.tokenizer
        self.dataset.return_idx = True
        self.dataset.add_gc_token = True

        self.fill_text_repr()
        self.fill_node_repr()

    def validate(self, epoch=-1, mode="valid"):
        self.model.eval()
        self.dataset.change_mode(mode=mode)
        loader = self.get_dataloader(mode=mode)
        monitor = self.valid_monitor if mode == "valid" else self.test_monitor
        t = time.time()
        with torch.no_grad():
            for E, y, idx in loader:
                logits = self.model(E, self.edge_idx, idx)
                loss = self.loss_fun(
                    logits,
                    F.one_hot(y.to(DEVICE), num_classes=self.dataset.num_labels
                ).to(torch.float32))
                preds = torch.argmax(logits, dim=-1)
                monitor.update(preds, y, loss)
        return monitor.emit(time=round(time.time() - t, 2), epoch=epoch)

    def train(self, epochs):
        self.prepare()
        best_val_acc = 0

        if self.freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            self.dataset.change_mode("train")
            for i, (E, y, idx) in tqdm(enumerate(self.get_dataloader(mode="train"))):
                self.optimizer.zero_grad()
                logits = self.model(E, self.edge_idx, idx)
                loss = self.loss_fun(
                    logits,
                    F.one_hot(y.to(DEVICE), num_classes=self.dataset.num_labels
                ).to(torch.float32))
                loss.mean().backward()
                preds = torch.argmax(logits, dim=-1)
                self.optimizer.step()
                self.train_monitor.update(preds, y, loss)
                self.model.T = nn.parameter.Parameter(self.model.T.clone().detach(), requires_grad=False)
                self.model.N = nn.parameter.Parameter(self.model.N.clone().detach(), requires_grad=False)
            self.train_monitor.emit(
                epoch=epoch,
                lr=self.optimizer.param_groups[0]["lr"],
                time=round(time.time() - t, 2),
            )

            valid_res = self.validate(epoch=epoch, mode="valid")
            self.lr_scheduler.step(valid_res["loss"])
            
            if valid_res['acc'] > best_val_acc:
                best_val_acc = valid_res['acc']
                self.save(suffix=f'_epoch={epoch}')
                self.validate(mode='test', epoch=epoch)
        
        self.validate(mode='test', epoch=epoch)
        return self.model
