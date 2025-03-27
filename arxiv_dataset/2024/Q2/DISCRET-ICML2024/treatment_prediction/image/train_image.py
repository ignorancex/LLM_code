import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from baseline_methods.shapley_value import collect_shap_deep_explanations
# from treatment_prediction.nlp.nlp_data_utils.simulation import run_simulation
from models.Qmod_baseline import *
from parse_args_image import parse_args
from image_data_utils.image_dataset import *
from utils_image import load_pretrained_backbone_models
from image_data_utils.image_dataset import split_train_valid_test_df
from utils_treatment import load_configs, transform_treatment_ids, set_lang_data

rule_based_explanations=["decision_tree", "anchor", "lore"]

if __name__ == "__main__":
        args = parse_args()

        args.cont_treatment = False
        args.has_dose = False

        
        random_seed = args.seed
        random.seed(random_seed)

        # Set a random seed for NumPy
        np.random.seed(random_seed)

        # Set a random seed for PyTorch
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, count_outcome_attr, treatment_attr, lang  = construct_dataset_main(args)

        # device = ("cuda" if torch.cuda.is_available() else "cpu")

        # if args.dataset_name == "music":
        #         raw_df = pd.read_csv(os.path.join(args.data_folder, 'music.csv'))
        #         df, offset =run_simulation(raw_df, propensities=[0.8, 0.6], 
        #                                         beta_t=1.0, 
        #                                         beta_c=50.0,
        #                                         gamma=1.0   ,
        #                                         cts=True)


        # train_df, valid_df, test_df = split_train_valid_test_df(df)
        
        numeric_count  = len(train_dataset.num_cols)
                # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
        category_count = list(train_dataset.cat_cols) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
        category_sum_count = train_dataset.cat_sum_count

        input_size = numeric_count + category_sum_count
        root_dir = os.path.dirname(os.path.realpath(__file__))
        rl_config, model_config, backbone_model_config = load_configs(args,root_dir=root_dir)

        mod = QNet_baseline(train_dataset, valid_dataset, test_dataset, input_size, args, model_config, backbone_model_config) # directory for saving the best model

        # load_pretrained_backbone_models(mod, os.path.join(args.log_folder, "_bestmod.pt"))

        # for p in mod.model.distilbert.parameters():
        #         p.requires_grad = False

        if args.is_log:
                thres = 0.02
                mod.model.load_state_dict(torch.load(os.path.join(mod.log_folder, 'bestmod.pt')))
                test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=Image_Dataset.collate_fn)
                if args.method == "nam":

                    explanations_by_treatment, all_outcome_tensor_by_treatment, all_pred_outcome_tensor = mod.eval_self_interpretable_models(test_dataset, train_dataset, max_depth=args.program_max_len, explanation_type=args.explanation_type)

                    eval_consistency(mod, test_dataloader, all_outcome_tensor_by_treatment[1] - all_outcome_tensor_by_treatment[0], None, explanations_by_treatment, explanation_type=args.explanation_type, fp=thres)

                else:
                        explanations_by_treatment, explainer_by_treatment = mod.posthoc_explain(test_dataset, tree_depth=args.program_max_len, explanation_type=args.explanation_type)
                    
                #     if train_dataset.count_outcome_attr is not None:
                        all_pred_outcome_tensor, all_pos_pred_tensor, all_neg_pred_tensor = mod.test(test_dataloader, return_pred=True)
                        eval_consistency(mod, test_dataloader, all_pos_pred_tensor - all_neg_pred_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, fp=thres)
                        if args.explanation_type in rule_based_explanations:
                                eval_sufficiency(mod, test_dataloader, all_pos_pred_tensor - all_neg_pred_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type, fp=thres)

                exit(1)
                #     else:
                #         all_pred_outcome_tensor = mod.test(test_dataloader, return_pred=True)
                #         eval_consistency(mod, test_dataloader, all_pred_outcome_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type)
                #         if args.explanation_type  in rule_based_explanations:
                #             eval_sufficiency(mod, test_dataloader, all_pred_outcome_tensor, explainer_by_treatment, explanations_by_treatment, explanation_type=args.explanation_type)



        # if not args.posthoc:
        mod.train(epochs=args.epochs,   # the maximum number of training epochs
                learning_rate = args.lr)  # learning rate for the training

        
                # Q0, Q1, A, Y, _ = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'])
        # else:
        #         mod.model = torch.load(args.log_folder+'_bestmod.pt')
        #         collect_shap_deep_explanations(mod.model, (mod.train_text[0:100], mod.train_confounder[]), test_data)

