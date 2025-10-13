import torch
import os
#from core.config import OUTPUT_PATH

import matplotlib.pyplot as plt


OUTPUT_PATH = ... # set path from config here

#model_name = "resnet50_robust"
model_name = "grid_search_resnet50"
layer_name = "_layer4[2]_1_comps"


out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", model_name + layer_name)
#class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
#matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))
class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

"""
print(torch.mean(class_preds))
print(torch.std(class_preds))
print(torch.mean(matches_desired_pred))

above_point_one_pred = torch.where(class_preds > 0.1, 1.0, 0.0)
print(torch.mean(above_point_one_pred))

above_point_two_pred = torch.where(class_preds > 0.2, 1.0, 0.0)
print(torch.mean(above_point_two_pred))

above_point_five_pred = torch.where(class_preds > 0.5, 1.0, 0.0)
print(torch.mean(above_point_five_pred))

"""


class_preds_list = []
matches_class_preds_list = []


for i in range(10):
    layer_name = "_layer3[5]_"+str(i+1)+"_comps"# + "_craft"


    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", model_name + layer_name)
    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))
    #class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
    #matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))


    print(torch.mean(class_preds))
    #print(torch.std(class_preds))
    #print(torch.mean(matches_desired_pred))

    #above_point_one_pred = torch.where(class_preds > 0.1, 1.0, 0.0)
    #print(torch.mean(above_point_one_pred))

    #above_point_two_pred = torch.where(class_preds > 0.2, 1.0, 0.0)
    #print(torch.mean(above_point_two_pred))

    #above_point_five_pred = torch.where(class_preds > 0.5, 1.0, 0.0)
    #print(torch.mean(above_point_five_pred))

    class_preds_list.append(torch.mean(class_preds).cpu())
    matches_class_preds_list.append(torch.mean(matches_desired_pred).cpu())

"""
plt.figure()
plt.plot([_+1 for _ in range(10)], class_preds_list, "--bo")
plt.xlabel("Number of Concepts")
plt.ylabel("Average Prediction")
plt.title("ResNet50 Layer4.2")
plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "layer_4_2_include_target_class_pred.png"))




plt.figure()
plt.plot([_+1 for _ in range(10)], matches_class_preds_list, "--bo")
plt.xlabel("Number of Concepts")
plt.ylabel("Average Matches Prediction")
plt.title("ResNet50 Layer4.2")
plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "layer_4_2_include_target_class_matches_pred.png"))
"""


