import torch
import os
#from core.config import OUTPUT_PATH

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = ... # set path from config here

model_name = "resnet50"


def print_model_results(out_save_path):
    #out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search" + model_name+"_"+layer_name+"_"+str(n_components)+"_comps_50_samples")

    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))
    #class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
    #matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

    print("class pred: {}".format(torch.mean(class_preds).cpu()))
    print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))



def print_results_compare_attribution_methods():
    model_name = "resnet50"
    layer_name = "layer4[2]"
    n_components = 4

    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search" + model_name+"_"+layer_name+"_"+str(n_components)+"_comps_50_samples")
    print_model_results(out_save_path)

    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+"_"+layer_name+"_gradientXinput_"+str(n_components)+"_comps")
    print_model_results(out_save_path)

    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+"_"+layer_name+"_noise_tunnel_"+str(n_components)+"_comps")
    print_model_results(out_save_path)



    model_name = "resnet50"
    layer_name = "layer3[5]"
    n_components = 4

    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+"_"+layer_name+"_gradientXinput_"+str(n_components)+"_comps")
    print_model_results(out_save_path)


    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+"_"+layer_name+"_"+str(n_components)+"_comps")
    print_model_results(out_save_path)




def print_full_model_results():
    model_name = "resnet50"
    layer_name = "layer4[2]"
    n_components = 4
    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", model_name+"_"+layer_name+"_"+str(n_components)+"_comps")
    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

    print(matches_desired_pred.shape)
    print("class pred: {}".format(torch.mean(class_preds).cpu()))
    print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))





    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

    print(matches_desired_pred.shape)
    print("class pred: {}".format(torch.mean(class_preds).cpu()))
    print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))





def print_full_model_results_validation_vis():
    model_name = "resnet50"
    layer_name = "layer4[2]"
    n_components = 4
    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", model_name+"_"+layer_name+"_"+str(n_components)+"_comps_val_data")
    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

    print(matches_desired_pred.shape)
    print("class pred: {}".format(torch.mean(class_preds).cpu()))
    print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))





    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

    print(matches_desired_pred.shape)
    print("class pred: {}".format(torch.mean(class_preds).cpu()))
    print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))





def print_get_failure_cases():
    model_name = "resnet50"
    layer_name = "layer4[2]"
    n_components = 4
    out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", model_name+"_"+layer_name+"_"+str(n_components)+"_comps")
    class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
    matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

    #print(matches_desired_pred.shape)
    #print("class pred: {}".format(torch.mean(class_preds).cpu()))
    #print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))

    print(class_preds.shape)
    #print(matches_desired_pred)




def print_other_models():
    print_model_results("resnet50_robust", "layer4[2]", 4)
    print_model_results("resnet50_robust", "layer3[5]", 6)


    print_model_results("resnet34", "layer4[2]", 4)
    print_model_results("resnet34", "layer3[5]", 6)


    #print_model_results("resnet50", "layer4[2]", 4)
    #print_model_results("resnet50", "layer3[5]", 6)


def print_var_run():

    class_preds_list = []
    matches_class_preds_list = []

    for i in range(5):
        num_comp = 4
        model_name = "resnet50"
        layer_name = "_layer4[2]"
        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+layer_name+"_4_comps_run_" + str(i))
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        print("class pred: {}".format(torch.mean(class_preds).cpu()))
        print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))

        class_preds_list.append(torch.mean(class_preds).cpu())
        matches_class_preds_list.append(torch.mean(matches_desired_pred).cpu())


    print("class_preds: {}".format(class_preds_list))
    print("matches_desired_pred: {}".format(matches_class_preds_list))

    print(torch.mean(torch.from_numpy(np.array(class_preds_list))))
    print(torch.std(torch.from_numpy(np.array(class_preds_list))))



    print(torch.mean(torch.from_numpy(np.array(matches_class_preds_list))))
    print(torch.std(torch.from_numpy(np.array(matches_class_preds_list))))




def plot_var_samples():

    class_preds_list = []
    matches_class_preds_list = []

    number_of_samples = [50] + [(i+1)*100 for i in range(9)]

    for num_samples in number_of_samples:
        num_comp = 4
        model_name = "resnet50"
        layer_name = "_layer4[2]"
        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name+layer_name+"_4_comps_"+str(num_samples)+"_samples")
        try:
            class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
            matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

            print("class pred: {}".format(torch.mean(class_preds).cpu()))
            print("matches class pred: {}".format(torch.mean(matches_desired_pred).cpu()))

            class_preds_list.append(torch.mean(class_preds).cpu())
            matches_class_preds_list.append(torch.mean(matches_desired_pred).cpu())
        except:
            class_preds_list.append(1.0)
            matches_class_preds_list.append(1.0)


    plt.figure()
    plt.plot(number_of_samples, class_preds_list, "--bo")
    plt.xlabel("Number of Samples")
    plt.ylabel("Average Prediction")
    plt.title("ResNet50; Layer4.2; 4 Components")
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "layer4_2_var_sample_count_avg_pred.png"))


    plt.figure()
    plt.plot(number_of_samples, matches_class_preds_list, "--bo")
    plt.xlabel("Number of Samples")
    plt.ylabel("Average Matches Prediction")
    plt.title("ResNet50; Layer4.2; 4 Components")
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "layer4_2_var_sample_count_matches_pred.png"))




def plot_results():
    class_preds_list_craft = []
    matches_class_preds_list_craft = []


    craft_highest_matching_pred = -1.0
    craft_highest_avg_pred = -1.0


    for i in range(10):
        layer_name = "_layer4[2]_"+str(i+1)+"_comps" + "_craft"

        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        class_preds_list_craft.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_craft.append(torch.mean(matches_desired_pred).cpu())

    #print("class preds craft: {}".format(class_preds_list))


    #craft_highest_avg_pred = torch.max(torch.tensor(class_preds_list))
    #craft_highest_matching_pred = torch.max(torch.tensor(matches_class_preds_list))



    class_preds_list = []
    matches_class_preds_list = []


    for i in range(10):
        layer_name = "_layer4[2]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        class_preds_list.append(torch.mean(class_preds).cpu())
        matches_class_preds_list.append(torch.mean(matches_desired_pred).cpu())


    #print(class_preds_list)
    #print(matches_class_preds_list)

    class_preds_list_layer3 = []
    matches_class_preds_list_layer3 = []


    for i in range(10):
        layer_name = "_layer3[5]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer3.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer3.append(torch.mean(matches_desired_pred).cpu())


    class_preds_list_layer2 = []
    matches_class_preds_list_layer2 = []


    for i in range(10):
        layer_name = "_layer2[3]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer2.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer2.append(torch.mean(matches_desired_pred).cpu())



    class_preds_list_layer1 = []
    matches_class_preds_list_layer1 = []


    for i in range(10):
        layer_name = "_layer1[2]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_exclude_target_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer1.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer1.append(torch.mean(matches_desired_pred).cpu())

    #print("class preds layer3: {}".format(class_preds_list_layer3))


    plt.figure()
    plt.plot([_+1 for _ in range(10)], class_preds_list, "--bo",
            [_+1 for _ in range(10)], class_preds_list_layer3, "--ys",
            [_+1 for _ in range(10)], class_preds_list_craft, "--g^",
            [_+1 for _ in range(10)], class_preds_list_layer2, "--rp",
            [_+1 for _ in range(10)], class_preds_list_layer1, "--mv",
            #[_+1 for _ in range(10)], [craft_highest_avg_pred for _ in range(10)], "--g",
            )
    plt.xlabel("Number of Components")
    plt.ylabel("Average Prediction")
    plt.title("ResNet50")
    plt.legend(["Layer4.2", "Layer3.5", "CRAFT (Layer4.2)", "Layer2.3", "Layer1.2"], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "average_pred.png"))



    plt.figure()
    plt.plot([_+1 for _ in range(10)], matches_class_preds_list, "--bo",
            [_+1 for _ in range(10)], matches_class_preds_list_layer3, "--ys",
            [_+1 for _ in range(10)], matches_class_preds_list_craft, "--g^",
            [_+1 for _ in range(10)], matches_class_preds_list_layer2, "--rp",
            [_+1 for _ in range(10)], matches_class_preds_list_layer1, "--mv",
            #[_+1 for _ in range(10)], [craft_highest_avg_pred for _ in range(10)], "--g",
            )
    plt.xlabel("Number of Components")
    plt.ylabel("Matches Desired Prediction")
    plt.title("ResNet50")
    plt.legend(["Layer4.2", "Layer3.5", "CRAFT (Layer4.2)", "Layer2.3", "Layer1.2"], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "average_matches_pred.png"))


    """
    plt.figure()
    plt.plot([_+1 for _ in range(10)], matches_class_preds_list, "--bo")
    plt.xlabel("Number of Concepts")
    plt.ylabel("Average Matches Prediction")
    plt.title("ResNet50 Layer4.2")
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "layer_4_2_include_target_class_matches_pred.png"))
    """






    class_preds_list_craft = []
    matches_class_preds_list_craft = []


    craft_highest_matching_pred = -1.0
    craft_highest_avg_pred = -1.0


    for i in range(10):
        layer_name = "_layer4[2]_"+str(i+1)+"_comps" + "_craft"

        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

        class_preds_list_craft.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_craft.append(torch.mean(matches_desired_pred).cpu())

    #print("class preds craft: {}".format(class_preds_list))


    #craft_highest_avg_pred = torch.max(torch.tensor(class_preds_list))
    #craft_highest_matching_pred = torch.max(torch.tensor(matches_class_preds_list))



    class_preds_list = []
    matches_class_preds_list = []


    for i in range(10):
        layer_name = "_layer4[2]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

        class_preds_list.append(torch.mean(class_preds).cpu())
        matches_class_preds_list.append(torch.mean(matches_desired_pred).cpu())


    #print(class_preds_list)
    #print(matches_class_preds_list)

    class_preds_list_layer3 = []
    matches_class_preds_list_layer3 = []


    for i in range(10):
        layer_name = "_layer3[5]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer3.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer3.append(torch.mean(matches_desired_pred).cpu())


    class_preds_list_layer2 = []
    matches_class_preds_list_layer2 = []


    for i in range(10):
        layer_name = "_layer2[3]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer2.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer2.append(torch.mean(matches_desired_pred).cpu())



    class_preds_list_layer1 = []
    matches_class_preds_list_layer1 = []


    for i in range(10):
        layer_name = "_layer1[2]_"+str(i+1)+"_comps"# + "_craft"


        out_save_path = os.path.join(OUTPUT_PATH, "sanity_check_patch_based", "grid_search_" + model_name + layer_name)
        class_preds = torch.load(os.path.join(out_save_path, "nmf_comp_check_predictions_for_class_avg_pool.pt"))
        matches_desired_pred = torch.load(os.path.join(out_save_path, "nmf_comp_check_matches_desired_pred_avg_pool.pt"))

        class_preds_list_layer1.append(torch.mean(class_preds).cpu())
        matches_class_preds_list_layer1.append(torch.mean(matches_desired_pred).cpu())

    #print("class preds layer3: {}".format(class_preds_list_layer3))


    plt.figure()
    plt.plot([_+1 for _ in range(10)], class_preds_list, "--bo",
            [_+1 for _ in range(10)], class_preds_list_layer3, "--ys",
            [_+1 for _ in range(10)], class_preds_list_craft, "--g^",
            [_+1 for _ in range(10)], class_preds_list_layer2, "--rp",
            [_+1 for _ in range(10)], class_preds_list_layer1, "--mv",
            #[_+1 for _ in range(10)], [craft_highest_avg_pred for _ in range(10)], "--g",
            )
    plt.xlabel("Number of Components")
    plt.ylabel("Average Prediction")
    plt.title("ResNet50")
    plt.legend(["Layer4.2", "Layer3.5", "CRAFT (Layer4.2)", "Layer2.3", "Layer1.2"], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "average_pred_allow_target.png"))



    plt.figure()
    plt.plot([_+1 for _ in range(10)], matches_class_preds_list, "--bo",
            [_+1 for _ in range(10)], matches_class_preds_list_layer3, "--ys",
            [_+1 for _ in range(10)], matches_class_preds_list_craft, "--g^",
            [_+1 for _ in range(10)], matches_class_preds_list_layer2, "--rp",
            [_+1 for _ in range(10)], matches_class_preds_list_layer1, "--mv",
            #[_+1 for _ in range(10)], [craft_highest_avg_pred for _ in range(10)], "--g",
            )
    plt.xlabel("Number of Components")
    plt.ylabel("Matches Desired Prediction")
    plt.title("ResNet50")
    plt.legend(["Layer4.2", "Layer3.5", "CRAFT (Layer4.2)", "Layer2.3", "Layer1.2"], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "average_matches_pred_allow_target.png"))
















print_full_model_results_validation_vis()
