import torch
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from Qmod import *
# from baseline_methods.nlp.transtee import TransTEE_nlp
from sklearn.tree import DecisionTreeRegressor

class QNet_baseline:
    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, train_dataset, valid_dataset, test_dataset, model_name="causalqnet",
                 a_weight = 1.0, y_weight=1.0, mlm_weight=1.0,
                 batch_size = 64,
                 modeldir = None, classification=False):
        # df['text'], df['T'], df['C'], df['Y']
        # if model_name == "causalqnet":
        self.model = CausalQNet.from_pretrained(
            'distilbert-base-uncased',
            num_labels = 2,
            output_attentions=False,
            output_hidden_states=False)
        # elif model_name == "TransTEE":
        #     self.model = TransTEE_nlp()
        
            

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_weights = {
            'a': a_weight,
            'y': y_weight,
            'mlm': mlm_weight}

        self.batch_size = batch_size
        self.modeldir = modeldir

        # idx = list(range(len(texts)))
        # random.shuffle(idx) # shuffle the index
        # n_train = int(len(texts)*0.8) 
        # n_val = len(texts)-n_train
        # idx_train = idx[0:n_train]
        # idx_val = idx[n_train:]

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.classification = classification


    def build_dataloader(self, texts, treatments, confounders=None, outcomes=None, counter_outcomes=None, tokenizer=None, sampler='random'):

        # fill with dummy values
        if outcomes is None:
            outcomes = [-1 for _ in range(len(treatments))]
        if counter_outcomes is None:
            counter_outcomes = [-1 for _ in range(len(treatments))]
        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

        out = defaultdict(list)
        for i, (W, A, C, Y, count_Y) in enumerate(zip(texts, treatments, confounders, outcomes, counter_outcomes)):
            encoded_sent = tokenizer.encode_plus(W, add_special_tokens=True,
                                                max_length=128,
                                                truncation=True,
                                                pad_to_max_length=True)

            out['text_id'].append(encoded_sent['input_ids'])
            out['text_mask'].append(encoded_sent['attention_mask'])
            out['text_len'].append(sum(encoded_sent['attention_mask']))
            out['A'].append(A)
            out['C'].append(C)
            out['Y'].append(Y)
            out["count_Y"].append(count_Y)

        self.y_scaler = StandardScaler().fit(np.array(out["Y"] + out['count_Y']).reshape(-1,1))
        y = self.y_scaler.transform(np.array(out["Y"]).reshape(-1,1))
        out["Y"] = y.reshape(-1).tolist()

        count_y = self.y_scaler.transform(np.array(out["count_Y"]).reshape(-1,1))
        out["count_Y"] = count_y.reshape(-1).tolist()

        data = (torch.tensor(out[x]) for x in ['text_id', 'text_len', 'text_mask', 'A', 'C','Y', 'count_Y'])
        # if y_scaler is None:
        

        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader
    
    def test(self,val_dataloader):
        # evaluate validation set
        self.model.eval()
        pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
        a_val_losses, y_val_losses, a_val_accs = [], [], []
    
        all_gt_outcome_ls = []
        all_gt_count_outcome_ls = []
        all_pos_pred_outcome_ls = []
        all_neg_pred_outcome_ls = []
        all_gt_treatment_ls = []


        for batch in pbar:
            # if torch.cuda.is_available():
                # batch = (x.cuda() for x in batch)
            # text_id, text_len, text_mask, A, _, Y, count_Y = batch
            _, _, A, Y, count_Y, _, _, (text_id, text_mask, text_len) = batch
            # text_id, text_len, text_mask, A, _, Y, count_Y = batch
            # text_id = text_id.unsqueeze(1)
            if torch.cuda.is_available():
                text_id = text_id.cuda() 
                text_mask = text_mask.cuda() 
                text_len = text_len.cuda() 
                A= A.cuda() 
                Y = Y.cuda() 

            Q0, Q1, _, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y, use_mlm=False)
            
            a_val_losses.append(a_loss.item())
            y_val_losses.append(y_loss.item())

            # A accuracy
            a_acc = torch.round(a_acc*len(A))
            a_val_accs.append(a_acc.item())

            all_gt_outcome_ls.append(Y.detach().cpu())
            if count_Y is not None:
                all_gt_count_outcome_ls.append(count_Y.detach().cpu())
            all_pos_pred_outcome_ls.append(Q1.detach().cpu())
            all_neg_pred_outcome_ls.append(Q0.detach().cpu())
            all_gt_treatment_ls.append(A.detach().cpu())
        
        all_gt_outcome_ls = torch.cat(all_gt_outcome_ls, dim=0)
        # all_gt_count_outcome_ls = None
        if len(all_gt_count_outcome_ls) > 0:
            all_gt_count_outcome_ls = torch.cat(all_gt_count_outcome_ls, dim=0)
        all_pos_pred_outcome_ls = torch.cat(all_pos_pred_outcome_ls, dim=0)
        all_neg_pred_outcome_ls = torch.cat(all_neg_pred_outcome_ls, dim=0)
        all_gt_treatment_ls = torch.cat(all_gt_treatment_ls, dim=0)

        if not self.classification:
            if self.train_dataset.y_scaler is not None:
                all_pos_pred_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_pos_pred_outcome_ls)
                all_neg_pred_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_neg_pred_outcome_ls)
                all_gt_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_gt_outcome_ls)
                if len(all_gt_count_outcome_ls) > 0:
                    all_gt_count_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_gt_count_outcome_ls)

        if all_gt_count_outcome_ls is not None and len(all_gt_count_outcome_ls) > 0:
            gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.stack([all_gt_outcome_ls.view(-1), all_gt_treatment_ls.view(-1)], dim=1), all_gt_count_outcome_ls.reshape(-1,1))
            avg_treatment_effect = evaluate_treatment_effect_core(all_pos_pred_outcome_ls, all_neg_pred_outcome_ls, gt_treatment_outcome, gt_control_outcome)
            print("average treatment effect::%f"%(avg_treatment_effect))
        
        else:
            if self.classification:
                avg_treatment_effect = torch.mean(torch.norm(all_pos_pred_outcome_ls - all_neg_pred_outcome_ls, dim=-1))
                print("average treatment effect::%f"%(avg_treatment_effect))
            else:
                avg_treatment_effect = torch.mean(torch.abs(all_pos_pred_outcome_ls - all_neg_pred_outcome_ls))
                print("average treatment effect::%f"%(avg_treatment_effect))
        
        n_val = len(val_dataloader.dataset)
        a_val_loss = sum(a_val_losses)/n_val
        print('A Validation loss:',a_val_loss)

        y_val_loss = sum(y_val_losses)/n_val
        print('Y Validation loss:',y_val_loss)

        val_loss = self.loss_weights['a'] * a_val_loss + self.loss_weights['y'] * y_val_loss
        print('Validation loss:',val_loss)

        a_val_acc = sum(a_val_accs)/n_val
        print('A accuracy:',a_val_acc)
        return avg_treatment_effect
    

    def train(self, learning_rate=2e-5, epochs=1, patience=5):
        ''' Train the model'''

        # split data into two parts: one for training and the other for validation
        
        # list of data
        # train_dataloader = self.build_dataloader(self.train_text, 
        #     self.train_treatment, self.train_confounder, self.train_outcome, self.train_counter_outcome)
        # val_dataloader = self.build_dataloader(self.val_text, 
        #     self.val_treatment, self.val_confounder, self.val_outcome, self.val_counter_outcome, sampler='sequential')
        
        # test_dataloader = self.build_dataloader(self.test_text,
        #     self.test_treatment, self.test_confounder, self.test_outcome, self.test_counter_outcome, sampler='sequential')

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=NLP_Dataset.collate_fn, drop_last=True)
        val_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=NLP_Dataset.collate_fn)
        self.train_dataloader = train_dataloader

        self.model.train() 
        optimizer = AdamW(self.model.parameters(), lr = learning_rate, eps=1e-8)

        best_val_loss = 1e6
        best_test_loss = 1e6
        epochs_no_improve = 0
        self.test(test_dataloader)
        for epoch in range(epochs):
            losses = []
            self.model.train()
            all_pred_outcome_ls = []
            all_gt_outcome_ls = []
        
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc='Training')
            for step, batch in pbar:
                # if torch.cuda.is_available():
                #     batch = (x.cuda() for x in batch)
                _, _, A, Y, count_Y, _, _, (text_id, text_mask, text_len) = batch
                # text_id, text_len, text_mask, A, _, Y, count_Y = batch
                # text_id = text_id.unsqueeze(1)
                if torch.cuda.is_available():
                    text_id = text_id.cuda() 
                    text_mask = text_mask.cuda() 
                    text_len = text_len.cuda() 
                    A= A.cuda() 
                    Y = Y.cuda() 


                self.model.zero_grad()
                Q0, Q1, mlm_loss, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y)
                all_gt_outcome_ls.append(Y.detach().cpu())
                all_pred_outcome_ls.append((Q0.view(-1)*(Y==0).type(torch.float).view(-1) + Q1.view(-1)*(Y==1).type(torch.float).view(-1)).detach().cpu())
                # compute loss
                loss = self.loss_weights['a'] * a_loss + self.loss_weights['y'] * y_loss + self.loss_weights['mlm'] * mlm_loss
                
                       
                pbar.set_postfix({'Y loss': y_loss.item(),
                  'A loss': a_loss.item(), 'A accuracy': a_acc.item(), 
                  'mlm loss': mlm_loss.item()})

                # optimizaion for the baseline
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            all_gt_outcome_ls = torch.cat(all_gt_outcome_ls, dim=0)
            all_pred_outcome_ls = torch.cat(all_pred_outcome_ls, dim=0)

            outcome_error = (torch.abs(all_pred_outcome_ls - all_gt_outcome_ls)).mean().item()
            print("training outcome error before rescaling back:", outcome_error)

            if not self.classification:
                if self.train_dataset.y_scaler is not None:
                    all_pred_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_pred_outcome_ls)
                    all_gt_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_gt_outcome_ls)
            outcome_error = (torch.abs(all_pred_outcome_ls - all_gt_outcome_ls)).mean().item()
            print("training outcome error:", outcome_error)
            val_loss = self.test(val_dataloader)
            test_loss = self.test(test_dataloader)

            # early stop
            if val_loss < best_val_loss:
                torch.save(self.model.state_dict(), self.modeldir+'_bestmod.pt') # save the best model
                best_val_loss = val_loss
                best_test_loss = test_loss
                epochs_no_improve = 0              
            else:
                epochs_no_improve += 1
            
            print("best validation loss:", best_val_loss)
            print("best test loss:", best_test_loss)
            

            # if epoch >= 5 and epochs_no_improve >= patience:              
            #     print('Early stopping!' )
            #     print('The number of epochs is:', epoch)
            #     break

        # load the best model as the model after training
        self.model.load_state_dict(torch.load(self.modeldir+'_bestmod.pt'))

        return self.model


    def get_Q(self, texts, treatments, confounders=None, outcomes=None, model_dir=None):
        '''Get conditional expected outcomes Q0 and Q1 based on the training model'''
        dataloader = self.build_dataloader(texts, treatments, confounders, outcomes, sampler='sequential')
        As, Cs, Ys = [], [], []
        Q0s = []  # E[Y|A=0, text]
        Q1s = []  # E[Y|A=1, text]
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Statistics computing")

        if not model_dir:
            self.model.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = self.model(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()
        else:
            Qmodel = torch.load(model_dir)
            Qmodel.eval()
            for step, batch in pbar:
                if torch.cuda.is_available():
                    batch = (x.cuda() for x in batch)
                text_id, text_len, text_mask, A, C, Y = batch
                Q0, Q1, _, _, _, _ = Qmodel(text_id, text_len, text_mask, A, Y, use_mlm = False)
                As += A.detach().cpu().numpy().tolist()
                Cs += C.detach().cpu().numpy().tolist()
                Ys += Y.detach().cpu().numpy().tolist()
                Q0s += Q0.detach().cpu().numpy().tolist()
                Q1s += Q1.detach().cpu().numpy().tolist()

        Q0s = [item for sublist in Q0s for item in sublist]
        Q1s = [item for sublist in Q1s for item in sublist]
        As = np.array(As)
        Ys = np.array(Ys)
        Cs = np.array(Cs)

        return Q0s, Q1s, As, Ys, Cs

