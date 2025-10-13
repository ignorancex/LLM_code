def get_sampled_dataset(im_dataset, cap_dataset, im_sample_idx_list):
    sampled_dataset = []
    for sample_idx in im_sample_idx_list:
        sampled_dataset.append(["image", im_dataset[sample_idx]])
        sampled_dataset.append(["caption", cap_dataset[sample_idx]])
    return sampled_dataset


def get_accuracy(preds, labels, processor):
    correct = 0
    for pred_id, label in zip(preds, labels):
        pred = processor.tokenizer.decode(pred_id).lower().strip()
        if pred != "" and label.lower().startswith(pred):
            correct += 1
    return correct / len(labels)
