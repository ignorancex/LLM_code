import gc
import time
import argparse
import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
import sys
from normco.model.scoring import *
from normco.model.phrase_model import *
from evaluator import *
from normco.utils.text_processing import *
import logging
th.manual_seed(7)
np.random.seed(7)

class NormCoTrainer:
    def __init__(self,args,logger):
        self.args = args
        self.logger = logger
        self.model,self.optimizer,self.loss = None,None,None

    def _build_model(self,id_size,vocab_size):
        args = self.args
        output_dim = args.output_dim
        sparse = True
        if args.optimizer in 'adam':
            sparse = False

        if args.model in "GRU":
            rnn = nn.GRU
        elif args.model in "LSTM":
            rnn = nn.LSTM

        # Pick the distance function
        margin = np.sqrt(output_dim)
        if args.scoring_type in "euclidean":
            distance_fn = EuclideanDistance()
        if args.scoring_type in "cosine":
            distance_fn = CosineSimilarity(dim=-1)
            margin = args.num_neg - 1
        elif args.scoring_type in "bilinear":
            distance_fn = BilinearMap(output_dim)
            margin = 1.0

        # Create the normalization model
        model = NormalizationModel(id_size,
                                     disease_embeddings_init=None,
                                     phrase_embeddings_init=None,
                                     vocab_size = vocab_size,
                                     distfn=distance_fn,
                                     rnn=rnn, embedding_dim=args.embedding_dim, output_dim=output_dim,
                                     dropout_prob=args.dropout_prob, sparse=sparse,
                                     use_features=args.use_features)

        # Choose the optimizer
        parameters = []
        default_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in 'feature_layer.weight' or name in 'score_layer.weight':
                    default_params.append(param)
                else:
                    parameters.append({'params': param, 'weight_decay': 0.0})
        parameters.append({'params': default_params})

        if args.optimizer in 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2reg, momentum=0.9)
        elif args.optimizer in 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adagrad':
            optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adadelta':
            optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=args.l2reg)
        elif args.optimizer in 'adam':
            optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2reg)

        # Pick the loss function
        if args.loss in 'maxmargin':
            loss = MaxMarginLoss(margin=margin)
        elif args.loss in 'xent':
            loss = CrossEntropyDistanceLoss()

        return model,optimizer,loss

    def train(self,mention_train,coherence_train,mention_valid,coherence_valid,id_size,vocab_size,logger):
        self.model, self.optimizer, self.loss = self._build_model(id_size, vocab_size)

        return self._train(mention_train,coherence_train,mention_valid,coherence_valid,logger)

    def _train(self,mention_train,coherence_train,mention_valid,coherence_valid,logger,
              log_dir='./tb',eval_data=None,
              logfile=None):

        if len(mention_train)>0:
            mention_train_loader = DataLoader(
                mention_train,
                batch_size=self.args.test_bsz,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=mention_train.collate
            )
        else:mention_train_loader=None
        if len(coherence_train)>0:
            coherence_train_loader = DataLoader(
                coherence_train,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=coherence_train.collate
            )
        else:coherence_train_loader=None
        if len(mention_valid)>0:
            mention_valid_loader = DataLoader(
                mention_valid,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=mention_train.collate
            )
        else: mention_valid_loader = None
        if len(coherence_valid)>0:
            coherence_valid_loader = DataLoader(
                coherence_valid,
                batch_size=self.args.test_bsz,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=coherence_train.collate
            )
        else: coherence_valid_loader = None

        '''
        Main training loop
        '''
        n_epochs_mention = self.args.num_epochs_mention_only
        n_epochs_coherence = self.args.num_epochs_with_coherence
        save_every = self.args.save_every
        save_file_name = self.args.save_file_name
        eval_every = self.args.eval_every
        log_train_loss_every = self.args.train_loss_log_interval
        use_coherence = not(self.args.mention_only)
        self.model.cuda()
        # Set mode to training
        self.model.train()
        step = 0
        # Keep track of best results for far
        acc_best = (-1, 0.0)
        patience = 15
        # Training loop

        for e in range(n_epochs_mention):

            self.model.train()
            if mention_train_loader:
                for mb in tqdm(mention_train_loader, desc='Epoch %d' % e):
                    mb['words'] = Variable(mb['words'])
                    mb['lens'] = Variable(mb['lens'])
                    mb['ids'] = Variable(mb['ids'])
                    mb['seq_lens'] = Variable(mb['seq_lens'])
                    if self.model.use_features:
                        mb['features'] = Variable(mb['features'])
                    # Mention step
                    self.optimizer.zero_grad()
                    # Pass through the model
                    mention_scores = self.model(mb, False)
                    # Get sequence length and number of negatives
                    nneg = mention_scores.size()[2]

                    # Get the loss
                    loss = self.loss(mention_scores.view(-1, nneg))

                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    step += 1
                gc.collect()
                if e%log_train_loss_every == 0 or e==n_epochs_coherence-1:
                    logger.info("loss at epoch {} is: {}".format(e,loss.data))

        for e in range(n_epochs_coherence):

            self.model.train()
            if coherence_train_loader:
                # Coherence data
                for cb in tqdm(coherence_train_loader, desc='Epoch %d' % e):
                    cb['words'] = Variable(cb['words'])
                    cb['lens'] = Variable(cb['lens'])
                    cb['ids'] = Variable(cb['ids'])
                    cb['seq_lens'] = Variable(cb['seq_lens'])
                    if self.model.use_features:
                        cb['features'] = Variable(cb['features'])
                    # Coherence step
                    self.optimizer.zero_grad()
                    # Pass through the model
                    scores = self.model(cb, use_coherence)
                    # Get sequence length and number of negatives
                    nneg = scores.size()[2]
                    # Get the loss
                    loss = self.loss(scores.view(-1, nneg))
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    step += 1

                gc.collect()
                if e%log_train_loss_every == 0 or e==n_epochs_coherence-1:
                    logger.info("loss at epoch {} is: {}".format(e,loss.data))
        res = self._evaluate(mention_valid_loader, coherence_valid_loader, logger, data_lab="valid")
        for i in res:
            logger.info("Valid result on {} is {:6f},{:6f}".format(i, res[i][0], res[i][1]))
        return res
        # Log final best values
        # if logfile:
        #     with open(logfile, 'a') as f:
        #         f.write("Best accuracy: %f in epoch %d\n" % (acc_best[1], acc_best[0]))
    def evaluate(self,mention_test_data,coherence_test_data,logger):
        if len(mention_test_data)>0:
            mention_test_loader = DataLoader(
                mention_test_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=mention_test_data.collate
            )
        else:
            mention_test_loader = None
        if len(coherence_test_data)>0:
            coherence_test_loader = DataLoader(
                coherence_test_data,
                batch_size=self.args.test_bsz,
                shuffle=True,
                num_workers=self.args.threads,
                collate_fn=coherence_test_data.collate
            )
        else:
            coherence_test_loader = None
        return self._evaluate(mention_test_loader,coherence_test_loader,logger)
    def _evaluate(self,mention_test_loader,coherence_test_loader,logger,data_lab="test"):

        predictions = {'without_coherence':[],'with_coherence':[]}
        true_labels = {'without_coherence':[],'with_coherence':[]}
        res = {}
        self.model.eval()
        loss = 0
        if mention_test_loader:
            for mb in tqdm(mention_test_loader):
                mb['words'] = Variable(mb['words'])
                mb['lens'] = Variable(mb['lens'])
                mb['ids'] = Variable(mb['ids'])

                mb['seq_lens'] = Variable(mb['seq_lens'])
                if self.model.use_features:
                    mb['features'] = Variable(mb['features'])
                # print(mb)
                # Mention step
                # Pass through the model
                with th.no_grad():
                    scores = self.model(mb, False)
                # Get sequence length and number of negatives
                nneg = scores.size()[2]
                # Get the loss
                loss += self.loss(scores.view(-1, nneg))
                predictions['without_coherence'].append(scores.view(-1,nneg).cpu().data.numpy())
                true_labels['without_coherence'].append(mb['ids'].cpu().data.numpy())
        if coherence_test_loader:
            for cb in tqdm(coherence_test_loader):
                cb['words'] = Variable(cb['words'])
                cb['lens'] = Variable(cb['lens'])
                cb['ids'] = Variable( cb['ids'])

                cb['seq_lens'] = Variable(cb['seq_lens'])
                if self.model.use_features:
                    cb['features'] = Variable(cb['features'])
                # Coherence step
                # Pass through the model
                with th.no_grad():
                    scores = self.model(cb, True)
                    nneg = scores.size()[2]
                    # print(scores.view(-1,nneg).cpu().data.numpy().shape)
                predictions['with_coherence'].append(scores.view(-1,nneg).cpu().data.numpy())
                true_labels['with_coherence'].append(cb['ids'].cpu().data.numpy())
                # Get sequence length and number of negatives
                nneg = scores.size()[2]
                # Get the loss
                loss += self.loss(scores.view(-1, nneg))
        # print("test_loss", loss.data)
        # logger.info("test loss: {}".format(loss.data))
        evaluator = Evaluator()
        correct_total1,correct_total5, nsamples = 0,0,0

        for type in predictions:
            if len(predictions[type])==0:
                res[type]=(0,0)
                logger.info("{} | data on {} type is empty".format(data_lab,type))
            else:
                result = np.concatenate(predictions[type], 0)
                true_labs = np.vstack(true_labels[type])
                true_labs = true_labs.reshape(-1)
                acc1,correct1,ntot = evaluator.accu(result,1)
                correct_total1+=correct1
                nsamples+=ntot
                logger.info("{} | acc1 on {} data is {:6f}".format(data_lab,type,acc1))
                acc5, correct5, ntot = evaluator.accu(result, 10)
                correct_total5+=correct5
                logger.info("{} | acc10 on {} data is {:6f}".format(data_lab,type, acc5))
                res[type] = (acc1,acc5)
        print("{} | acc1 on {} data is {:6f}".format(data_lab,"all", correct_total1/nsamples))
        print("{} | acc10 on {} data is {:6f}".format(data_lab,"all", correct_total5/ nsamples))
        res['all']=(correct_total1/nsamples,correct_total5/nsamples)
        return res