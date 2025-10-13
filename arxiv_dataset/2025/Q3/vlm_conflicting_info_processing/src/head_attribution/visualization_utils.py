import os

import matplotlib.pyplot as plt
import pickle as pkl


def get_intervened_result(output_dir, layer_id, head_id):
    output_path = os.path.join(output_dir, f"L{layer_id}_h{head_id}.pkl")
    with open(output_path, "rb") as f:
        res = pkl.load(f)
    return res


def get_accuracy(preds, labels, processor):
    correct = 0
    for pred_id, label in zip(preds, labels):
        pred = processor.tokenizer.decode(pred_id).lower().strip()
        if pred != "" and label.lower().startswith(pred):
            correct += 1
    return correct / len(labels)


def get_intervened_plot(args, layer_id, head_id, im_preds, cap_preds, processor):
    # Load intervened results
    res = get_intervened_result(args.output_dir, layer_id, head_id)

    all_labels = res["all_labels"]
    all_misled_labels = res["all_misled_labels"]
    all_total_logit_diffs_after_intervention = res["all_total_logit_diffs_after_intervention"]
    all_predictions_intervened_dict = res["all_predictions_intervened_dict"]

    # Compute before and after intervention performance
    acc_clean = 0.5 * get_accuracy(im_preds, all_labels[::2], processor) + 0.5 * get_accuracy(cap_preds, all_labels[1::2], processor)
    # print(f"L{layer_id} H{head_id}")
    # print(f"acc_clean: {acc_clean}")

    acc_intervened_dict, acc_intervened_dict_misled = {},{}
    for alpha, all_predictions_intervened in all_predictions_intervened_dict.items():
        acc_intervened_dict[alpha] = get_accuracy(all_predictions_intervened, all_labels, processor)
        acc_intervened_dict_misled[alpha] = get_accuracy(all_predictions_intervened, all_misled_labels, processor)
    max_acc_intervened = max(acc_intervened_dict.values())
    # print(f"best acc_intervened: {max_acc_intervened}")

    # Visualization
    plt.plot(acc_intervened_dict.keys(), acc_intervened_dict.values(), label="correct")
    plt.plot(acc_intervened_dict_misled.keys(), acc_intervened_dict_misled.values(), label="misled")
    plt.hlines(acc_clean, args.alpha_lower_bound, args.alpha_upper_bound, alpha=0.6, linestyle="--", color="red")
    plt.title(f"| Layer {layer_id} | Head {head_id} |")
    plt.legend(loc="best")

    max_acc_diff = max_acc_intervened - acc_clean
    image_output_path = os.path.join(args.output_dir, f"acc_diff{max_acc_diff*100:.2f}_L{layer_id}_h{head_id}.png")

    plt.savefig(image_output_path, bbox_inches="tight")
    # print()
    plt.close()


def get_unimodal_intervened_plot(args, layer_id, head_id, im_preds, cap_preds, processor):
    # Load intervened results
    res = get_intervened_result(args.output_dir, layer_id, head_id)

    all_labels = res["all_labels"]
    all_misled_labels = res["all_misled_labels"]
    all_total_logit_diffs_after_intervention = res["all_total_logit_diffs_after_intervention"]
    all_predictions_intervened_dict = res["all_predictions_intervened_dict"]

    # Compute before and after intervention performance
    acc_clean_image = get_accuracy(im_preds, all_labels[::2], processor)
    acc_clean_caption = get_accuracy(cap_preds, all_labels[1::2], processor)
    # print(f"L{layer_id} H{head_id}")
    # print(f"acc_clean: {acc_clean}")

    acc_intervened_dict_image, acc_intervened_dict_misled_image = {},{}
    acc_intervened_dict_caption, acc_intervened_dict_misled_caption = {},{}
    for alpha, all_predictions_intervened in all_predictions_intervened_dict.items():
        acc_intervened_dict_image[alpha] = get_accuracy(all_predictions_intervened[::2], all_labels[::2], processor)
        acc_intervened_dict_misled_image[alpha] = get_accuracy(all_predictions_intervened[::2], all_misled_labels[::2], processor)

        acc_intervened_dict_caption[alpha] = get_accuracy(all_predictions_intervened[1::2], all_labels[1::2], processor)
        acc_intervened_dict_misled_caption[alpha] = get_accuracy(all_predictions_intervened[1::2], all_misled_labels[1::2], processor)
    max_acc_intervened_image = max(acc_intervened_dict_image.values())
    max_acc_intervened_caption = max(acc_intervened_dict_caption.values())

    
    # print(f"best acc_intervened: {max_acc_intervened}")

    # Visualization
    make_plot(
        layer_id, 
        head_id, 
        acc_intervened_dict_image, 
        acc_intervened_dict_misled_image, 
        acc_clean_image, 
        args,
        sort_by_value=max_acc_intervened_image - acc_clean_image, 
        setup_name="unimodal_image_correct", 
    )

    make_plot(
        layer_id, 
        head_id, 
        acc_intervened_dict_caption, 
        acc_intervened_dict_misled_caption, 
        acc_clean_caption, 
        args,
        sort_by_value=max_acc_intervened_caption - acc_clean_caption, 
        setup_name="unimodal_caption_correct", 
    )


def make_plot(layer_id, head_id, acc_intervened_dict, acc_intervened_dict_misled, acc_clean, args, sort_by_value, setup_name):
    plt.plot(acc_intervened_dict.keys(), acc_intervened_dict.values(), label="correct")
    plt.plot(acc_intervened_dict_misled.keys(), acc_intervened_dict_misled.values(), label="misled")
    plt.hlines(acc_clean, args.alpha_lower_bound, args.alpha_upper_bound, alpha=0.6, linestyle="--", color="red")
    plt.title(f"| Layer {layer_id} | Head {head_id} |")
    plt.legend(loc="best")

    image_output_dir = os.path.join(args.output_dir, setup_name)
    image_output_path = os.path.join(image_output_dir, f"acc_diff{sort_by_value*100:.2f}_L{layer_id}_h{head_id}.png")
    
    os.makedirs(image_output_dir, exist_ok=True)
    plt.savefig(image_output_path, bbox_inches="tight")
    plt.close()


def get_precompute_logit_diff(output_dir_prefix, target_modality, num_samples=100):
    output_dir = os.path.join(output_dir_prefix, f"{target_modality}_n{num_samples}")
    output_path = os.path.join(output_dir, "precompute_logit_diff.pkl")
    with open(output_path, "rb") as f:
        all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list = pkl.load(f)
    return all_logit_diffs, all_total_logit_diffs, all_predictions_clean, sample_idx_list