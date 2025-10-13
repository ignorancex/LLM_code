import numpy as np
import pickle
import torch
import os

import matplotlib.pyplot as plt

OUTPUT_PATH = ... # set path from config here


def print_shifting_results():
    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/resnet50_robust/adjusted_preds.npy"
    data = np.load(data_path, allow_pickle=True)#.item()

    print(data.shape)

    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))


    print(np.max(data, axis=0))


    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/resnet50/adjusted_preds.npy"
    data = np.load(data_path, allow_pickle=True)#.item()

    print(data.shape)

    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))


    print(np.max(data, axis=0))

    """
    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/adjusted_preds_tensor_list.npy"
    data = np.load(data_path, allow_pickle=True)#.item()

    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/adjusted_preds_max_tensor_list.npy"
    data_max = np.load(data_path, allow_pickle=True)#.item()





    print(data.shape)

    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))

    print(np.mean(data_max, axis=0))
    print(np.std(data_max, axis=0))
    """
    #print(data[5:22].shape)

    #data = data[5:22]

    #for i in range(len(data)):
    #    print(i, data[i])

    #print(np.mean(data, axis=1))
    #print(np.std(data, axis=1))



    """
    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/adjusted_preds.npy"
    data = np.load(data_path, allow_pickle=True).item()

    print(data["1_32"])
    print(data["max_1_32"])

    #print(data["32_1"])
    #print(data["max_32_1"])


    print(data["1_5"])
    print(data["5_1"])


    print(data["13_21"])
    print(data["21_13"])

    print(data["max_13_21"])
    print(data["max_21_13"])


    print(data["14_22"])
    print(data["22_14"])
    """


def print_masking_results():
    #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/masking_input_adjusted_preds_max_tensor_list.npy"


    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/resnet50_robust/adjusted_preds.pickle"

    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    data, from_class_towards_class = data

    #data = np.load(data_path, allow_pickle=True)#.item()
    #print(data.shape)

    increase_factors = []
    max_preds = []
    original_preds = []
    overrule_original_pred = []


    #i = 0
    for i, results in enumerate(data):
        #print(results)
        from_class, towards_class = from_class_towards_class[i]
        #print(results[:, towards_class])

        original_pred_original_class = results[0, from_class]

        original_max_pred = results[0, towards_class]
        original_preds.append(original_max_pred)
        new_max_pred = np.max(results[1:, towards_class])
        idx_new_max_pred = np.argmax(results[1:, towards_class])

        original_class_pred = results[idx_new_max_pred, from_class]


        if (new_max_pred >= original_class_pred) and (new_max_pred > 0.5):
            print(from_class, towards_class, new_max_pred, original_max_pred)

            overrule_original_pred.append(1.0)
        else:
            overrule_original_pred.append(0.0)

        #print(new_max_pred)
        increase_factors.append(new_max_pred/original_max_pred)

        max_preds.append(new_max_pred)
        #i += 1
        #if i > 30:
        #    break
        #raise RuntimeError
    #print(max_preds)


    increase_factors = np.array(increase_factors)
    print(np.mean(increase_factors))

    original_preds = np.array(original_preds)
    print(np.mean(original_preds))
    print(np.std(original_preds))

    max_preds = np.array(max_preds)
    print(np.mean(max_preds))
    print(np.std(max_preds))



    print(100.0*np.mean(overrule_original_pred))





def print_masking_results_list():
    #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/masking_input_adjusted_preds_max_tensor_list.npy"

    mean_default_preds = []
    mean_shifted_preds = []

    std_default_preds = []
    std_shifted_preds = []

    mean_pred_swap = []
    mean_increase_factors = []

    average_pred_by_shift_factor = []


    for run_index in range(5):
        #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/resnet50_robust_layer3[5]/adjusted_preds_"+str(run_index)+".pickle"
        #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/resnet50_robust/adjusted_preds_"+str(run_index)+".pickle"
        data_path = os.path.join(OUTPUT_PATH, "sanity_check/shifting_test/resnet50_layer3[5]/adjusted_preds_"+str(run_index)+"_more_steps.pickle")

        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)

        data, from_class_towards_class = data

        average_pred_by_shift_factor.append(np.mean(data, axis=0))
        #data = np.load(data_path, allow_pickle=True)#.item()
        #print(data.shape)

        increase_factors = []
        max_preds = []
        original_preds = []
        overrule_original_pred = []


        #i = 0
        for i, results in enumerate(data):
            #print(results)
            from_class, towards_class = from_class_towards_class[i]
            #print(results[:, towards_class])

            #original_pred_original_class = results[0, from_class]

            #original_max_pred = results[0, towards_class]
            #original_preds.append(original_max_pred)
            #new_max_pred = np.max(results[1:, towards_class])
            #idx_new_max_pred = np.argmax(results[1:, towards_class])

            #original_class_pred = results[idx_new_max_pred, from_class]

            #original_pred_original_class = results[0, from_class]

            original_max_pred = results[0]
            original_preds.append(original_max_pred)
            new_max_pred = np.max(results[1:])
            idx_new_max_pred = np.argmax(results[1:])

            #original_class_pred = results[idx_new_max_pred, from_class]

            """
            if (new_max_pred >= original_class_pred) and (new_max_pred > 0.5):
                #print(from_class, towards_class, new_max_pred, original_max_pred)

                overrule_original_pred.append(1.0)
            else:
                overrule_original_pred.append(0.0)
            """
            #print(new_max_pred)
            increase_factors.append(new_max_pred/original_max_pred)

            max_preds.append(new_max_pred)
            #i += 1
            #if i > 30:
            #    break
            #raise RuntimeError
        #print(max_preds)

        mean_default_preds.append(np.mean(original_preds))
        std_default_preds.append(np.std(original_preds))

        mean_shifted_preds.append(np.mean(max_preds))
        std_shifted_preds.append(np.std(max_preds))

        mean_pred_swap.append(100.0*np.mean(overrule_original_pred))

        mean_increase_factors.append(np.mean(increase_factors))

        """
        increase_factors = np.array(increase_factors)
        print(np.mean(increase_factors))

        original_preds = np.array(original_preds)
        print(np.mean(original_preds))
        print(np.std(original_preds))

        max_preds = np.array(max_preds)
        print(np.mean(max_preds))
        print(np.std(max_preds))



        print(100.0*np.mean(overrule_original_pred))
        """

    print(np.mean(mean_default_preds))
    print(np.std(mean_default_preds))
    print(np.mean(std_default_preds))

    print(np.mean(mean_shifted_preds))
    print(np.std(mean_shifted_preds))
    print(np.mean(std_shifted_preds))

    print(np.mean(mean_pred_swap))
    print(np.std(mean_pred_swap))

    print(np.mean(mean_increase_factors))
    print(np.mean(mean_shifted_preds) / np.mean(mean_default_preds))


    average_pred_by_shift_factor = np.array(average_pred_by_shift_factor)
    print(average_pred_by_shift_factor.shape)
    average_pred_by_shift_factor = np.mean(average_pred_by_shift_factor, axis=0)

    offsets = [0.0, 0.5, 1.0, 2.0, 10.0, 15.0, 20.0, 35.0, 50.0, 200.0]
    offsets = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 35.0, 50.0, 200.0]

    print(average_pred_by_shift_factor)
    plt.figure()
    plt.plot(offsets, average_pred_by_shift_factor, "--bo")
    plt.xlabel("Offset")
    plt.ylabel("Average Prediction")
    plt.title("ResNet50; Layer3.5")
    plt.savefig(os.path.join(OUTPUT_PATH, "tmp", "sanity_check_layer_3_5_more_offsets.png"))



def print_masking_results_digipath():
    #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/masking_input_adjusted_preds_max_tensor_list.npy"
    #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/masking_input_adjusted_preds_max_tensor_list.pt"

    data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/masking_input_adjusted_preds_max_tensor_list.pt"
    #data_path = "/home/digipath2/projects/xai/contrast_diagnoses/out/sanity_check/version_299/adjusted_preds_max_tensor_list.pt"

    data = torch.load(data_path)#.cpu().numpy()

    #print(data[1, 32])
    #raise RuntimeError

    """
    overrule_original_pred = []

    for from_class in range(data.shape[0]):
        for towards_class in range(data.shape[1]):
            towards_pred_max = np.max(data[from_class, towards_class, :, towards_class])
            from_class_original_pred = data[from_class, towards_class, 0, from_class]

            if towards_pred_max > 0.7:#(towards_pred_max > from_class_original_pred) and (towards_pred_max > 0.8):
                print("{}, {}, {}".format(from_class, towards_class, towards_pred_max))
                overrule_original_pred.append(1.0)
            else:
                overrule_original_pred.append(0.0)


    #print(data[32, 1])
    #print(np.max(data[32, 1, :, 1]))


    """
    #print(data.shape)

    #print(data[1, 32])

    #raise RuntimeError


    valid_data = []
    wanted_indices = []
    original_class_indices = []
    towards_class_indices = []

    for from_class in range(data.shape[0]):
        for towards_class in range(data.shape[1]):
            if torch.mean(data[from_class, towards_class]) > - 0.9:
                valid_data.append(data[from_class, towards_class])
                wanted_indices.append(towards_class)
                original_class_indices.append(from_class)
                towards_class_indices.append(towards_class)

    valid_data = torch.stack(valid_data, dim=0).numpy()

    #print(valid_data.shape)
    #raise RuntimeError

    increase_factors = []
    max_preds = []
    original_preds = []
    overrule_original_pred = []
    max_other_preds = []
    overrule_other_preds = []
    original_pred_other_class = []

    for i, results in enumerate(valid_data):
        #print(results)
        idx = wanted_indices[i]
        original_max_pred = results[0, idx]
        original_pred_other_class.append(original_max_pred)
        new_max_pred = np.max(results[:, idx])
        max_idx = np.argmax(results[:, idx])

        max_other_preds_at_idx = np.max(results[max_idx, 1:])

        valid_other_indices = [_+1 for _ in range(45)]
        valid_other_indices.remove(idx)
        valid_other_indices.remove(original_class_indices[i])
        max_other_pred = np.max(results[0, valid_other_indices])

        max_other_preds.append(max_other_pred)

        original_class_pred = results[max_idx, original_class_indices[i]]
        original_preds.append(original_class_pred)

        #if new_max_pred >= max_other_preds_at_idx:

        #if (new_max_pred > original_class_pred) and (new_max_pred > 0.2):
        if (new_max_pred > 0.2):
            print("{}, {}, {}".format(original_class_indices[i], towards_class_indices[i], new_max_pred))
            overrule_original_pred.append(1.0)
        else:
            overrule_original_pred.append(0.0)

        #print(new_max_pred)
        increase_factors.append(new_max_pred/original_max_pred)

        max_preds.append(new_max_pred)

    #print(max_preds)
    increase_factors = np.array(increase_factors)
    print(np.mean(increase_factors))

    original_pred_other_class = np.array(original_pred_other_class)
    print(np.mean(original_pred_other_class))
    print(np.std(original_pred_other_class))

    max_preds = np.array(max_preds)
    print(np.mean(max_preds))

    print(np.std(max_preds))

    max_other_preds = np.array(max_other_preds)
    print(np.mean(max_other_preds))


    original_preds = np.array(original_preds)
    print(np.mean(original_preds))

    overrule_original_pred = np.array(overrule_original_pred)
    print(np.mean(overrule_original_pred))

    print(np.mean(max_preds)/np.mean(original_pred_other_class))



print_masking_results_list()

#print_masking_results_digipath()

