import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


from parse_args_tabular import parse_args
from tabular_data_utils.tabular_dataset import *
import synthetic_lang
from create_language import Language
from utils_treatment import load_configs 
import torch
from tab_models.tab_model import *
from data_generations.TCGA import *
from utils_treatment_tab import load_tab_data

from baseline_methods.baseline import *
from baseline_main import classical_baseline_main, classical_baseline_ls
import json

rule_based_explanations=["decision_tree", "anchor", "lore"]

if __name__ == "__main__":
    args, parser = parse_args()   
    
    args = parser.parse_args()
    if args.seed is None and args.dataset_id is not None:
        args.seed = args.dataset_id
    if args.seed is not None:
        print("configure random seed as {}".format(args.seed))
        random_seed = args.seed
        random.seed(random_seed)

        # Set a random seed for NumPy
        np.random.seed(random_seed)

        # Set a random seed for PyTorch
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(all_data, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, synthetic_lang.DROP_FEATS)
    all_data, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr, normalize_y, extra_info = load_tab_data(args)
    train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(args.dataset_name, all_data, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, synthetic_lang.DROP_FEATS, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, normalize_y=normalize_y, extra_info=extra_info)

    # if args.cat_and_cont_treatment or args.subset_num is not None:
    #     args.seed = args.dataset_id
    # random_seed = args.seed
    # random.seed(random_seed)

    # # Set a random seed for NumPy
    # np.random.seed(random_seed)

    # # Set a random seed for PyTorch
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # if not args.dataset_name == "sw" and not args.dataset_name == "tcga2":
    #     all_data, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr, normalize_y, extra_info = load_tab_data(args)
    #     numeric_count  = len(train_dataset.num_cols)
    #     # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
    #     category_count = list(train_dataset.cat_cols) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
    #     category_sum_count = train_dataset.cat_sum_count
    #     input_size = numeric_count + category_sum_count
    # else:
    #     if args.dataset_name == "sw":
    #         sw.add_params(parser)
    #     elif args.dataset_name == "tcga2":
    #         tcga.add_params(parser)
    #     args = parser.parse_args()
            
    #     id_attr = "id"
    #     outcome_attr = "outcome"
    #     treatment_attr = None
    #     dose_attr = None
    #     count_outcome_attr = None
    #     args.has_dose= False
    #     all_data = None
    #     normalize_y = False
    #     args.cont_treatment = False
    #     # train_dataset.num_cols = ["x_" + str(k) for k in range(train_dataset["units"].shape[-1])]
    #     # train_dataset.cat_cols = []
    #     # train_dataset.cat_sum_count = 0
    #     input_size = train_dataset[0].covariates.shape[-1]


        # all_data, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr, normalize_y, extra_info = load_tab_data(args)
    
    numeric_count  = len(train_dataset.num_cols)
        # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
    category_count = list(train_dataset.cat_cols) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
    category_sum_count = train_dataset.cat_sum_count

    input_size = numeric_count + category_sum_count
        
    
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(classical_baseline_ls)
    print("method::", args.method)
    if args.method in classical_baseline_ls:
        classical_baseline_main(args, args.method, train_dataset.transformed_features, train_dataset.treatment_array,train_dataset.dose_array, train_dataset.outcome_array,
                                 valid_dataset.transformed_features, valid_dataset.treatment_array,valid_dataset.dose_array, valid_dataset.outcome_array, 
                                 test_dataset.transformed_features, test_dataset.treatment_array, test_dataset.dose_array, test_dataset.outcome_array, 
                                 train_dataset, test_dataset, train_dataset.count_outcome_array, valid_dataset.count_outcome_array, test_dataset.count_outcome_array)
        exit(1)

    if not args.method == "causal_forest":
        rl_config, model_config,backbone_model_config = load_configs(args,root_dir=root_dir)
        # args.embed_size = model_config["hidden_size"]
        if not args.dataset_name == "six":
            trainer = baseline_trainer(args, train_dataset, input_size, model_config, backbone_model_config, device)
        else:
            trainer = baseline_trainer(args, train_dataset, input_size, model_config, backbone_model_config, device, outcome_regression=False)
        if args.is_log:
            trainer.model.load_state_dict(torch.load(os.path.join(trainer.log_folder, 'bestmod.pt')))
            test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
            trainer.test(test_dataloader)
            if args.method == "nam":

                explanations_by_treatment, all_outcome_tensor_by_treatment, all_pred_outcome_tensor = trainer.eval_self_interpretable_models(test_dataset, train_dataset, max_depth=args.program_max_len, explanation_type=args.explanation_type)

                if train_dataset.count_outcome_attr is not None:
                    eval_consistency(trainer, test_dataloader, all_outcome_tensor_by_treatment[1] - all_outcome_tensor_by_treatment[0], None, explanations_by_treatment, explanation_type=args.explanation_type, out_folder = args.log_folder)
                else:
                    eval_consistency(trainer, test_dataloader, all_outcome_tensor_by_treatment, None, explanations_by_treatment, explanation_type=args.explanation_type, out_folder = args.log_folder)

            else:
                subset_ids = torch.arange(len(test_dataset)).tolist()
                if args.dataset_name == "news_cont" or args.dataset_name == "tcga":
                    if args.explanation_type == "anchor":
                        subset_size = 20#max(len(test_dataset)*0.05, 20)
                    else:
                        subset_size = 50
                    subset_ids = np.random.choice(len(test_dataset), int(subset_size), replace=False).tolist()
                    # test_dataset = torch.utils.data.Subset(test_dataset, np.random.choice(len(test_dataset), int(subset_size), replace=False))
                    # test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=tabular_Dataset.collate_fn)
                    # subset_test_dataqet = torch.utils.data.Subset(test_dataset, )
                
                explanations_by_treatment, explainer_by_treatment = trainer.posthoc_explain(test_dataset, tree_depth=args.program_max_len, explanation_type=args.explanation_type, subset_ids=subset_ids)
                
                if args.dataset_name == "six":
                    all_feat_ls = torch.load(os.path.join(args.log_folder, "all_feat_ls"))
                    res_explanation_json = dict()
                    for treat in explanations_by_treatment:
                        for sample_id in range(len(explanations_by_treatment[treat])):
                            local_explanation = explanations_by_treatment[treat][sample_id]
                            selected_feats = [all_feat_ls[feat_id] for feat_id in local_explanation]
                            
                            print("explanation of sample {} in treatment {}: ".format(sample_id, treat))
                            print(selected_feats)
                            if sample_id not in res_explanation_json:
                                res_explanation_json[sample_id] = dict()
                            res_explanation_json[sample_id][treat] = selected_feats
                    
                    with open(os.path.join(args.log_folder, "explanation.json"), "w") as f:
                        json.dump(res_explanation_json, f, indent=4)
                    
                    exit(1)
                
                
                if train_dataset.count_outcome_attr is not None:
                    all_pred_outcome_tensor, all_pos_pred_tensor, all_neg_pred_tensor, all_gt_pos_tensor, all_gt_neg_tensor = trainer.test(test_dataloader, return_pred=True)
                    all_pred_outcome_tensor = all_pred_outcome_tensor[subset_ids]
                    all_pos_pred_tensor = all_pos_pred_tensor[subset_ids]
                    all_neg_pred_tensor = all_neg_pred_tensor[subset_ids]
                    eval_consistency(trainer, test_dataloader, all_pos_pred_tensor - all_neg_pred_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, sub_sample_ids=subset_ids, out_folder=args.log_folder)
                    with open(os.path.join(args.log_folder, "pred_treatment_effect.json"), "w") as f:
                        pred_treatment_effect = (all_pos_pred_tensor - all_neg_pred_tensor).tolist()
                        pred_treatment_effect_maps = {idx: pred_treatment_effect[idx] for idx in range(len(pred_treatment_effect))}
                        json.dump(pred_treatment_effect_maps, f, indent=4)
                    with open(os.path.join(args.log_folder, "gt_treatment_effect.json"), "w") as f:
                        gt_treatment_effect = (all_gt_pos_tensor - all_gt_neg_tensor).tolist()
                        gt_treatment_effect_maps = {idx: gt_treatment_effect[idx] for idx in range(len(gt_treatment_effect))}
                        json.dump(gt_treatment_effect_maps, f, indent=4)

                    if args.explanation_type in rule_based_explanations:
                        eval_sufficiency(trainer, test_dataloader, all_pos_pred_tensor - all_neg_pred_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, sub_sample_ids=subset_ids)
                else:
                    all_pred_outcome_tensor = trainer.test(test_dataloader, return_pred=True)
                    # all_pred_outcome_tensor = all_pred_outcome_tensor[subset_ids]
                    # all_pos_pred_tensor = all_pos_pred_tensor[subset_ids]
                    # all_neg_pred_tensor = all_neg_pred_tensor[subset_ids]
                    eval_consistency(trainer, test_dataloader, all_pred_outcome_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, sub_sample_ids=subset_ids)
                    if args.explanation_type  in rule_based_explanations:
                        eval_sufficiency(trainer, test_dataloader, all_pred_outcome_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, sub_sample_ids=subset_ids)

                
                # trainer.eval_aopc(test_dataset, train_dataset, explainer_by_treatment, explanations_by_treatment, max_depth=args.program_max_len, explanation_type=args.explanation_type)
                # trainer.eval_faithfulness(test_dataset, train_dataset, max_depth=args.program_max_len, explanation_type=args.explanation_type)
                # all_selected_feats_ours = torch.load(os.path.join(args.log_folder, "selected_feat_ours"), map_location="cpu")
                # trainer.eval_aopc(test_dataset, train_dataset, max_depth=args.program_max_len, all_selected_feat_ls=all_selected_feats_ours)
                # trainer.eval_stability(test_dataset, explainer_by_treatment, tree_depth=args.program_max_len, explanation_type=args.explanation_type)
            exit(1)
        else:
            trainer.run(train_dataset, valid_dataset, test_dataset)
        
    else:
        # model = causal_forest(train_dataset.features.numpy(), train_dataset.treatment_array.view(-1).numpy(), train_dataset.outcome_array.view(-1).numpy())
        # # interpret_tree(model, torch.cat([test_X, single_test_X.view(1,-1)]).numpy(), adapted_feat_name_ls, max_depth=model.max_depth, min_samples_leaf=model.min_samples_leaf)
        # treatment_effect = model.effect(test_dataset.features.numpy())
        est = causal_forest_main(train_dataset, valid_dataset, test_dataset)
        interpret_causal_forest_main(est, test_dataset)

