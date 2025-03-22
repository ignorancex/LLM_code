import torch
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from Qmod import *
# from baseline_methods.nlp.transtee import TransTEE_nlp
from sklearn.tree import DecisionTreeRegressor
from transformers import CLIPProcessor, CLIPModel
from baseline_methods.TransTEE.TransTEE import TransTEE_image, TransTEE
from baseline_methods.TransTEE.DRNet import Drnet_image, Vcnet_image
from baseline_methods.self_interpretable_models.prototype import ProtoVAE
from baseline_methods.dragonnet import *
from baseline_methods.self_interpretable_models.ENRL import ENRL

class QNet_model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



def process_images(image_ls, processor):
    
    return processor(images=image_ls.astype(np.uint8), return_tensors="pt", padding=False)

class QNet_baseline:
    """Model wrapper for training Qnet and get Q's for new data"""
    def __init__(self, train_dataset, valid_dataset, test_dataset, input_size, args, model_configs, backbone_model_config
                 ):
        # df['text'], df['T'], df['C'], df['Y']
        # if model_name == "causalqnet":
        args.num_treatments=2
        if args.method == "TransTEE":
            params = {'num_features': input_size, 'num_treatments': 2,
            'h_dim': model_configs["hidden_size"], 'cov_dim':model_configs["cov_dim"]}
            self.model = TransTEE_image(params)
        elif args.method == "TransTEE_tab":
            params = {'num_features': input_size, 'num_treatments': 2,
            'h_dim': model_configs["hidden_size"], 'cov_dim':model_configs["cov_dim"]}
            self.model = TransTEE(params)
        elif args.method == "drnet":
            cfg_density = [[input_size, 100, 1, 'relu'], [100, 64, 1, 'relu']]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            isenhance = 1
            self.model = Drnet_image(cfg_density, num_grid, cfg, isenhance=isenhance, h=model_configs["h"], num_t=args.num_treatments)
        elif args.method == "tarnet":
            cfg_density = [[input_size, 100, 1, 'relu'], [100, 64, 1, 'relu']]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            isenhance = 0
            self.model = Drnet_image(cfg_density, num_grid, cfg, isenhance=isenhance, h=model_configs["h"], num_t=args.num_treatments)
        elif args.method == "vcnet":
            cfg_density = [[input_size, 100, 1, 'relu'], [100, 64, 1, 'relu']]
            num_grid = 10
            cfg = [[64, 64, 1, 'relu'], [64, 1, 1, 'id']]
            degree = 2
            knots = [0.33, 0.66]
            self.model = Vcnet_image(cfg_density, num_grid, cfg, degree, knots, num_t=args.num_treatments)
        elif args.method == "prototype":
            img_size = train_dataset.image_array[0].shape[0]
            self.model = ProtoVAE(img_size, input_size, model_configs["hidden_size"], model_configs["num_prototypes"], args.num_treatments)
            
        elif args.method == "nam":
            self.model = dragonet_model_nam(input_size, model_configs["hidden_size"])
            self.shared_hidden_dim = model_configs["hidden_size"]
            
        elif args.method == "ENRL":
            if train_dataset.feat_range_mappings is not None:
                feat_range_mappings = [train_dataset.feat_range_mappings[num_feat] for num_feat in train_dataset.num_cols]
            else:
                feat_range_mappings = [[0,1] for num_feat in train_dataset.num_cols]
            multihot_f_unique_num_ls = [len(train_dataset.cat_unique_count_mappings[cat_feat]) for cat_feat in train_dataset.cat_cols]
            
            self.model = ENRL(rule_len = args.program_max_len, rule_n = args.topk_act, numeric_f_num=len(train_dataset.num_cols), multihot_f_num=len(train_dataset.cat_cols), multihot_f_dim=sum(multihot_f_unique_num_ls), feat_range_mappings=feat_range_mappings, num_treatment=args.num_treatments, cont_treatment=False, has_dose=False)
        elif args.method == "dragonnet":
            self.model = dragonet_model_image(input_size, model_configs["hidden_size"])
            self.shared_hidden_dim = model_configs["hidden_size"]
        
        self.backbone_full_model = None
        self.backbone = args.backbone
        if args.backbone is not None:
            if args.backbone.lower() == "transtee":
                print("start loading transtee backbone")
                if not args.cont_treatment and args.has_dose:
                    cov_dim = backbone_model_config["cov_dim"]
                else:
                    cov_dim = input_size
                
                params = {'num_features': input_size, 'num_treatments': args.num_treatments,
                'h_dim': backbone_model_config["hidden_size"], 'cov_dim':cov_dim}
                # self.model = TransTEE(params, has_dose=args.has_dose, cont_treatment = args.cont_treatment)
                self.shared_hidden_dim = backbone_model_config["hidden_size"]
                self.backbone_full_model = TransTEE_image(params)
                # self.backbone_model = self.backbone_full_model.encoding_features
        
        if args.cached_backbone is not None and os.path.exists(args.cached_backbone):
            cached_backbone_state_dict = torch.load(args.cached_backbone)
            self.backbone_full_model.load_state_dict(cached_backbone_state_dict)
            if args.fix_backbone:
                for p in self.backbone_full_model.parameters():
                    p.requires_grad = False
        if self.backbone_full_model is not None:
            self.backbone_full_model = self.backbone_full_model.cuda()

        self.regression_ratio = args.regression_ratio


        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.model = CausalQNet.from_pretrained(
        #     'distilbert-base-uncased',
        #     num_labels = 2,
        #     output_attentions=False,
        #     output_hidden_states=False)
        # elif model_name == "TransTEE":
        #     self.model = TransTEE_nlp()
        
            

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.batch_size = args.batch_size
        self.modeldir = args.log_folder
        self.method = args.method
        # idx = list(range(len(texts)))
        # random.shuffle(idx) # shuffle the index
        # n_train = int(len(texts)*0.8) 
        # n_val = len(texts)-n_train
        # idx_train = idx[0:n_train]
        # idx_val = idx[n_train:]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_folder = args.log_folder
        self.has_dose = False
        self.cont_treatment = False
        self.classification = False
        self.outcome_regression=True
        self.num_treatments = 2

    def eval_self_interpretable_models(self, test_dataset, train_dataset, max_depth, subset_count = 10, all_selected_feat_ls = None, explanation_type="decision_tree"):
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=Image_Dataset.collate_fn)
        full_base_X = train_dataset.transformed_features.mean(dim=0)
        full_base_X = full_base_X.to(self.device)
        # all_idx_set_ls = generate_all_subsets([k for k in range(self.program_max_len)])
        # random.shuffle(all_idx_set_ls)
        
        # selected_idx_set_ls = all_idx_set_ls[:subset_count]
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        # all_tree_by_treatment_id = self.fit_decision_tree(train_dataset, max_depth)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),desc='Testing')

            compute_ate = False
            
            all_outcome_diff_ls = []
            all_ate_diff_ls = []

            all_explanation_perturbation_ls = []
            all_feat_ls = []
            all_outcome_ls_by_treatment = dict()
            all_treatment_ls = []
            all_gt_outcome_ls= []
            all_orig_outcome_ls = []
            all_topk_features_by_treatment = {k:[] for k in range(self.num_treatments)}
            for step, batch in pbar:
                
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
                # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
                # idx_ls, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
                ids, origin_X, A, Y, count_Y, X, _, _ = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                A = A.to(self.device)
                if count_Y is not None:
                    compute_ate = True
                
                # if D is not None:
                #     D = D.to(self.device)

                # X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                # X_num = X_num.to(self.device)
                # X_cat = X_cat.to(self.device)
                # pred, _ = self.model.forward(X_num, X_cat, A, d=D)
                
                _, pred = self.model.forward(X, A)
                for k in range(self.num_treatments):
                    topk_feature_ids = self.model.get_topk_features(X, A, k=max_depth)
                    all_topk_features_by_treatment[k].append(topk_feature_ids.cpu())

                if test_loader.dataset.y_scaler is not None:
                    pred = transform_outcome_by_rescale_back(test_loader.dataset, pred.cpu())

                all_orig_outcome_ls.append(pred.cpu().view(-1))

                for treatment_id in range(self.num_treatments):
                    # _, curr_outcome_pred = self.model.forward(X, A, d=D)
                    _, curr_outcome_pred = self.model.forward(X, torch.ones_like(A)*treatment_id)
                    if test_loader.dataset.y_scaler is not None:
                        curr_outcome_pred = transform_outcome_by_rescale_back(test_loader.dataset, curr_outcome_pred)

                    if treatment_id not in all_outcome_ls_by_treatment:
                        all_outcome_ls_by_treatment[treatment_id] = []
                    all_outcome_ls_by_treatment[treatment_id].append(curr_outcome_pred.cpu().view(-1))
                all_treatment_ls.append(A.cpu().view(-1))
                all_feat_ls.append(X.cpu())
                if test_loader.dataset.y_scaler is not None:
                    Y = transform_outcome_by_rescale_back(test_loader.dataset, Y.cpu())

                all_gt_outcome_ls.append(Y.cpu().view(-1))
            
            all_feat_tensor = torch.cat(all_feat_ls, dim=0)
            all_outcome_tensor_by_treatment = {treatmet_id: torch.cat(all_outcome_ls_by_treatment[treatmet_id], dim=0) for treatmet_id in all_outcome_ls_by_treatment}
            all_topk_features_by_treatment = {k:torch.cat(all_topk_features_by_treatment[k], dim=0) for k in range(self.num_treatments)}
            all_treatment_tensor = torch.cat(all_treatment_ls, dim=0)
            all_origin_outcome_tensor = torch.cat(all_orig_outcome_ls)
            # for treatment_id in range(self.num_treatments):
            #     tree = DecisionTreeRegressor(max_depth=max_depth)
            #     tree.fit(all_feat_tensor.cpu().numpy(), all_outcome_tensor_by_treatment[treatment_id].cpu().numpy())
            #     all_tree_by_treatment_id[treatment_id] = tree
            return all_topk_features_by_treatment, all_outcome_tensor_by_treatment, all_origin_outcome_tensor

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
    
    def test(self,val_dataloader, return_pred=False):
        # evaluate validation set
        self.model.eval()
        pbar = tqdm(val_dataloader, total=len(val_dataloader), desc='Validating')
        a_val_losses, y_val_losses, a_val_accs = [], [], []
    
        all_gt_outcome_ls = []
        all_gt_count_outcome_ls = []
        all_pos_pred_outcome_ls = []
        all_neg_pred_outcome_ls = []
        all_gt_treatment_ls = []
        all_pred_outcome_ls= []
        all_full_pred_outcome=[]
        with torch.no_grad():
            for batch in pbar:
                # if torch.cuda.is_available():
                    # batch = (x.cuda() for x in batch)
                # text_id, text_len, text_mask, A, _, Y, count_Y = batch
                ids, origin_X, A, Y, count_Y, X, _, image_ls = batch
                image = process_images(image_ls, self.processor)

                # text_id, text_len, text_mask, A, _, Y, count_Y = batch
                # text_id = text_id.unsqueeze(1)
                if torch.cuda.is_available():
                    A= A.cuda() 
                    Y = Y.cuda()
                    image.to(self.device)
                    X = X.to(self.device)
                if not self.method == "prototype":
                    # pred, full_pred = self.model.forward(image, X, A, test=True) 
                    if self.method == "nam" or self.method == "TransTEE_tab":
                        _, pred, full_pred = self.model.forward(X, A, test=True)
                    elif self.method == "ENRL":
                        X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                        X_num = X_num.to(self.device)
                        X_cat = X_cat.to(self.device)
                        pred, full_pred = self.model(X_num, X_cat, A)
                    else:
                        pred, full_pred = self.model.forward(image, X, A, test=True) 
                        
                else:
                    image_ls = [torch.from_numpy(image_ls[i]).float().permute(2,0,1) for i in range(len(image_ls))]
                    image_tensor = torch.stack(image_ls,dim=0).float().to(self.device)
                    pred, decoded, kl_loss, ortho_loss = self.model.forward(image_tensor, X, A) 
                    pos_pred, _,_,_ = self.model.forward(image_tensor, X, torch.ones_like(A))
                    neg_pred, _,_,_ = self.model.forward(image_tensor, X, torch.zeros_like(A))
                    full_pred = torch.cat([neg_pred.view(-1,1), pos_pred.view(-1,1)], dim=-1)
                    # recons = torch.nn.functional.mse_loss(decoded, image_tensor, reduction="mean")
                if self.backbone is not None:
                    pred, full_pred = reg_model_forward_image(self, image, X, A, None, origin_X, pred, full_out=full_pred, test = True)
                all_gt_treatment_ls.append(A)
                all_pred_outcome_ls.append(pred.view(len(Y), -1))
                all_gt_outcome_ls.append(Y)
                all_full_pred_outcome.append(full_pred.view(len(Y), -1))
                # all_count_gt_outcome.append(count_Y)
            

        
        # if not self.cont_treatment:
        all_full_pred_outcome_tensor = torch.cat(all_full_pred_outcome).cpu()
        all_pred_outcome_tensor = torch.cat(all_pred_outcome_ls).cpu()
        all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls).cpu()
        # if self.tr:
        # all_pred_treatment_tensor = torch.cat(all_pred_treatment).cpu()
        all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls).cpu()
        
        if self.test_dataset.y_scaler is not None:
            all_gt_outcome_tensor = transform_outcome_by_rescale_back(self.test_dataset, all_gt_outcome_tensor)
            all_pred_outcome_tensor = transform_outcome_by_rescale_back(self.test_dataset, all_pred_outcome_tensor)
            # if not self.cont_treatment:
            all_full_pred_outcome_tensor = torch.cat([transform_outcome_by_rescale_back(self.test_dataset, all_full_pred_outcome_tensor[:,k]) for k in range(all_full_pred_outcome_tensor.shape[1])], dim=-1)
        
        # all_concat_count_Y_tensor = None
        # if all_count_gt_outcome[0] is not None:
        #     all_concat_count_Y_tensor = torch.cat(all_count_gt_outcome)
        #     if self.test_dataset.y_scaler is not None:
        #         all_concat_count_Y_tensor = transform_outcome_by_rescale_back(self.test_dataset, all_concat_count_Y_tensor)
        # regression_loss = self.regression_loss(all_concat_true_tensor, all_concat_pred_tensor)
            
        # mise = romb(np.square(all_gt_outcome_tensor.cpu().numpy() - all_pred_outcome_tensor.cpu().numpy()), dx=step_size)
        # inter_r = np.array(all_gt_outcome_tensor.numpy()) - all_pred_outcome_tensor.squeeze().numpy()
        # ite = np.mean(inter_r ** 2)
        # if self.outcome_regression:
        outcome_error = F.mse_loss(all_gt_outcome_tensor.view(-1, 1).cpu(), all_pred_outcome_tensor.view(-1, 1).cpu()).item()
        # else:
        #     outcome_error = F.binary_cross_entropy_with_logits(all_gt_outcome_tensor.view(-1, 1).cpu(), all_pred_outcome_tensor.view(-1, 1).cpu()).item()
        #     all_pred_outcome_tensor_probs = torch.sigmoid(all_pred_outcome_tensor)
        #     outcome_acc = np.mean((all_pred_outcome_tensor_probs > 0.5).view(-1).numpy() == all_gt_outcome_tensor.view(-1).numpy())
        #     outcome_auc = roc_auc_score(all_gt_outcome_tensor.view(-1).numpy(), all_pred_outcome_tensor_probs.view(-1).numpy())
        #     print("outcome accuracy::%f, outcome auc score::%f"%(outcome_acc, outcome_auc))
            
        # if self.tr:
        #     treatment_acc, treatment_auc = self.treatment_pred_gt_comparison(all_gt_treatment_tensor, all_pred_treatment_tensor)
        
        
        # if all_concat_count_Y_tensor is not None and self.num_treatments == 2 and not self.cont_treatment:
        #     gt_treatment_outcome, gt_control_outcome = split_treatment_control_gt_outcome(torch.cat([all_gt_outcome_tensor, all_gt_treatment_tensor], dim=1), all_concat_count_Y_tensor)
            
        #     avg_ite, avg_ate = evaluate_treatment_effect_core(all_full_pred_outcome_tensor[:,1], all_full_pred_outcome_tensor[:,0], gt_treatment_outcome, gt_control_outcome)
        #     print("average individual treatment effect::%f"%avg_ite)
        #     print("average treatment effect::%f"%avg_ate)
        #     return outcome_error, avg_ite, avg_ate
        # if self.tr:
        #     print("treatment accuracy::%f, treatment auc score::%f, outcome error::%f"%(treatment_acc, treatment_auc, outcome_error))
        # else:
        if not return_pred:
            print("outcome error::%f"%outcome_error)
            return outcome_error
        else:
            return all_pred_outcome_tensor, all_full_pred_outcome_tensor[:, 1], all_full_pred_outcome_tensor[:,0]
    def posthoc_explain(self, test_dataset, tree_depth=2, explanation_type="decision_tree"):
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=Image_Dataset.collate_fn)
        self.model.load_state_dict(torch.load(os.path.join(self.log_folder, 'bestmod.pt')))
        with torch.no_grad():
            
            self.model.eval()
            pbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Validating')
            

            feature_ls = []
            origin_feature_ls = []
            unique_treatment_set = set()
            unique_dose_set = set()
            unique_treatment_dose_set = set()


            for batch in pbar:
                ids, origin_X, A, Y, count_Y, X, _, image_ls = batch
                # idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls,_ = batch
                
                feature_ls.append(X.detach().cpu())
                origin_feature_ls.append(origin_X.detach().cpu())
                curr_treatment_ls = A.detach().cpu().view(-1).tolist()
                curr_dose_ls = None

                
                unique_treatment_set.update(curr_treatment_ls)
                if curr_dose_ls is not None:
                    unique_dose_set.update(curr_dose_ls)
                    unique_treatment_dose_set.update((zip(curr_treatment_ls, curr_dose_ls)))
            
            feature_tensor = torch.cat(feature_ls, dim=0)
            origin_feature_tensor = torch.cat(origin_feature_ls, dim=0)

        return obtain_post_hoc_explanatios_main(self, test_dataset, test_dataloader, unique_treatment_set, feature_tensor, origin_feature_tensor, tree_depth, explanation_type)



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

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=Image_Dataset.collate_fn, drop_last=True)
        val_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=Image_Dataset.collate_fn)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=Image_Dataset.collate_fn)
        self.train_dataloader = train_dataloader

        self.model.train() 

        best_val_outcome_error = 1e6
        best_test_outcome_error = 1e6
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
                # _, _, A, Y, count_Y, _, _, (text_id, text_mask, text_len) = batch
                ids, origin_X, A, Y, count_Y, X, _, image_ls = batch
                images = process_images(image_ls, self.processor)

                # text_id, text_len, text_mask, A, _, Y, count_Y = batch
                # text_id = text_id.unsqueeze(1)
                if torch.cuda.is_available():
                    A= A.cuda() 
                    Y = Y.cuda()
                    images.to(self.device)
                    X = X.to(self.device)
                self.optimizer.zero_grad()
                other_loss = 0
                if not self.method == "prototype":
                    if self.method == "nam" or self.method == "TransTEE_tab":
                         _, out = self.model.forward(X, A)
                    # elif self.method == "TransTEE_tab":
                    #     out = self.model.forward(X, A, Y) 
                        
                    elif self.method == "ENRL":
                        X_num, X_cat = origin_X[:,0:self.model.numeric_f_num], origin_X[:,self.model.numeric_f_num:]
                        X_num = X_num.to(self.device)
                        X_cat = X_cat.to(self.device)
                        out, other_loss = self.model(X_num, X_cat, A)
                    else:
                        out = self.model.forward(images, X, A, Y) 
                else:
                    image_ls = [torch.from_numpy(image_ls[i]).float().permute(2,0,1) for i in range(len(image_ls))]
                    image_tensor = torch.stack(image_ls,dim=0).float().to(self.device)
                    out, decoded, kl_loss, ortho_loss = self.model.forward(image_tensor, X, A) 
                    recons = torch.nn.functional.mse_loss(decoded, image_tensor, reduction="mean")
                    other_loss += recons + kl_loss + ortho_loss
                
                if self.backbone_full_model is not None:
                    out = reg_model_forward_image(self, images, X, A, None, origin_X, out)

                loss = F.mse_loss(out.view(-1,1), Y) + other_loss
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            print("Training loss::", np.mean(np.array(losses)))

            print("evaluation on validation set::")
            val_outcome_error = self.test(val_dataloader)
            
            print("evaluation on test set::")
            test_outcome_error = self.test(test_dataloader)

            if val_outcome_error < best_val_outcome_error:
                best_val_outcome_error = val_outcome_error
                best_test_outcome_error = test_outcome_error
                torch.save(self.model.state_dict(), os.path.join(self.log_folder, 'bestmod.pt'))
            print("Performance at Epoch %d: validation outcome error::%f, test outcome error::%f"%(epoch, best_val_outcome_error, best_test_outcome_error))
            print("best validation outcome error::%f, best test outcome error::%f"%(best_val_outcome_error, best_test_outcome_error))


                # self.model.zero_grad()
                # Q0, Q1, mlm_loss, y_loss, a_loss, a_acc = self.model(text_id, text_len, text_mask, A, Y)
                # all_gt_outcome_ls.append(Y.detach().cpu())
                # all_pred_outcome_ls.append((Q0.view(-1)*(Y==0).type(torch.float).view(-1) + Q1.view(-1)*(Y==1).type(torch.float).view(-1)).detach().cpu())
                # # compute loss
                # loss = self.loss_weights['a'] * a_loss + self.loss_weights['y'] * y_loss + self.loss_weights['mlm'] * mlm_loss
                
                       
                # pbar.set_postfix({'Y loss': y_loss.item(),
                #   'A loss': a_loss.item(), 'A accuracy': a_acc.item(), 
                #   'mlm loss': mlm_loss.item()})

                # optimizaion for the baseline
            #     loss.backward()
            #     optimizer.step()
            #     losses.append(loss.item())
            # all_gt_outcome_ls = torch.cat(all_gt_outcome_ls, dim=0)
            # all_pred_outcome_ls = torch.cat(all_pred_outcome_ls, dim=0)

            # outcome_error = (torch.abs(all_pred_outcome_ls - all_gt_outcome_ls)).mean().item()
            # print("training outcome error before rescaling back:", outcome_error)

            # if not self.classification:
            #     if self.train_dataset.y_scaler is not None:
            #         all_pred_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_pred_outcome_ls)
            #         all_gt_outcome_ls = transform_outcome_by_rescale_back(self.train_dataset, all_gt_outcome_ls)
            # outcome_error = (torch.abs(all_pred_outcome_ls - all_gt_outcome_ls)).mean().item()
            # print("training outcome error:", outcome_error)
            # val_loss = self.test(val_dataloader)
            # test_loss = self.test(test_dataloader)

            # # early stop
            # if val_loss < best_val_loss:
            #     torch.save(self.model.state_dict(), self.modeldir+'_bestmod.pt') # save the best model
            #     best_val_loss = val_loss
            #     best_test_loss = test_loss
            #     epochs_no_improve = 0              
            # else:
            #     epochs_no_improve += 1
            
            # print("best validation loss:", best_val_loss)
            # print("best test loss:", best_test_loss)
            

            # if epoch >= 5 and epochs_no_improve >= patience:              
            #     print('Early stopping!' )
            #     print('The number of epochs is:', epoch)
            #     break

        # load the best model as the model after training
        # self.model.load_state_dict(torch.load(self.modeldir+'_bestmod.pt'))

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

