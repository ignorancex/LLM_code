import random

import numpy as np
import torch

import base64
from io import BytesIO


def get_caption_label(args, classes, image_label):
    caption_label = -1

    if args.caption_type == "consistent":
        caption_label = image_label

    elif args.caption_type == "inconsistent":
        caption_label = np.random.choice([
            i for i in range(len(classes))
            if i != image_label
        ]).item()
    elif args.caption_type == "text_only":
        caption_label = np.random.choice([
            i for i in range(len(classes))
        ]).item()
    elif args.caption_type == ["irrelevant", "no_caption"]:
        caption_label = -1
    
    return caption_label


def get_candidate_labels(args, classes, image_label, caption_label):
    """
    Generate candidate labels based on `args.caption_type`. 
    """
    candidate_labels = None

    if args.n_candidates == 0:
        return [], -1, -1

    if args.caption_type in ["consistent", "irrelevant", "no_caption"]:
        candidate_labels = np.random.choice(
            [
                i for i in range(len(classes))
                if i != image_label
            ],
            args.n_candidates - 1,
            replace=False,
        ).tolist()
        candidate_labels += [image_label]

    elif args.caption_type == "inconsistent":
        candidate_labels = np.random.choice(
            [
                i for i in range(len(classes))
                if (i != image_label) and (i != caption_label)
            ],
            args.n_candidates - 2,
            replace=False,
        ).tolist()
        candidate_labels += [image_label, caption_label]
    
    elif args.caption_type == "text_only":
        
        candidate_labels = np.random.choice(
            [
                i for i in range(len(classes))
                if i != caption_label
            ],
            args.n_candidates - 1,
            replace=False,
        ).tolist()
        candidate_labels += [caption_label]

    if candidate_labels:
        
        random.shuffle(candidate_labels)

        # get the relative position of the image option and the caption option
        relative_position_image = candidate_labels.index(image_label) if (image_label != -1 and (not args.caption_type == "text_only")) else -1
        relative_position_caption = candidate_labels.index(caption_label) if caption_label != -1 else -1

        candidate_labels = [classes[i] for i in candidate_labels]

        return candidate_labels, relative_position_image, relative_position_caption
    else:
        return [], -1, -1


def get_format_labels(args, candidate_labels, classes, caption_label):
    format_labels = None


    if args.caption_type in ["consistent", "inconsistent", "text_only"]:

        caption_label_str = classes[caption_label]

        format_labels = [caption_label_str] + candidate_labels

    elif args.caption_type in ["irrelevant", "no_caption"]:
        format_labels = candidate_labels

    return format_labels
    
    
def get_prompt(args, prompt_generator, format_labels, classes, caption_label, simple_caption=False):
    prompt = None

    if args.n_candidates > 0:
        prompt = prompt_generator(format_labels)
    else:
        if args.caption_type in ["consistent", "inconsistent", "text_only"]:
            if simple_caption:
                prompt = f"An image of a {classes[caption_label]}"
            else:
                prompt = prompt_generator([classes[caption_label]])
        elif args.caption_type == "no_caption":
            prompt = prompt_generator()

    return prompt


def get_responses(args, model, processor, batch):
    responses = None

    if args.caption_type == "text_only":
        with torch.no_grad():
            outputs = model.generate(
                **batch['inputs'], 
                do_sample=False,
                num_beams=1,
                max_new_tokens=16,
                min_length=1,
                pad_token_id = processor.tokenizer.pad_token_id,
            )
    else:


        with torch.no_grad():
            outputs = model.generate( # warning due to this generate function implemented in HF..., forward will not result in a warning
                **batch['inputs'], 
                do_sample=False,
                num_beams=1,
                max_new_tokens=16,
                min_length=1,
                pad_token_id = processor.tokenizer.pad_token_id,
            )
    responses = process_responses(outputs, processor, args, source_length=batch["inputs"]["input_ids"].size(1))
    
    return responses


def process_responses(responses, processor, args, source_length):

    print(processor.batch_decode(responses, skip_special_tokens=True))

    if args.model_family == "gpt":
        responses = [responses.choices[0].message.content]
    elif args.model_family in ["llava", "custom_llava", "instructblip", "qwen", "llava-onevision"] or args.caption_type == "text_only":
        responses = processor.batch_decode(responses[:, source_length:], skip_special_tokens=True)
    else:
        responses = processor.batch_decode(responses, skip_special_tokens=True)

    responses = [t.strip() for t in responses]
    for i, response in enumerate(responses):        
        responses[i] = response.strip(".").strip("'").strip(".").lower()

    print(responses)

    return responses


def get_target_and_misleading_labels(args, image_labels, caption_labels):
    if args.modality_to_report == "image":
        target_labels = image_labels
        misleading_labels = caption_labels
    if args.modality_to_report == "text":
        target_labels = caption_labels
        misleading_labels = image_labels
    return target_labels, misleading_labels


def get_choices(choice_ids, classes, args):

    return [
        [classes[int(c)] for c in choices]
        for choices in choice_ids
    ]


def get_confusion_matrix(responses, classes, image_labels, caption_labels, args, choice_ids=None):
    target_labels, misleading_labels = get_target_and_misleading_labels(
        args, image_labels, caption_labels
    )

    classes = [c.lower() for c in classes] # account for capitalized letters in class labels

    all_choices = get_choices(choice_ids, classes, args)

    all_predictions = []

    n_correct, n_misled, n_incorrect_in_choices, n_incorrect_out_choices = 0,0,0,0
    for i, response in enumerate(responses):
        target_label = target_labels[i]
        misleading_label = misleading_labels[i]
        
        choices = [c.lower() for c in all_choices[i]] # account for capitalized letters in class labels
        
        print(f"FOR DEBUG: prediction: [{response}], gt: [{classes[target_label]}]")

        if args.n_candidates == 0:
            response = classes.index(response)
            if response == target_label:
                    n_correct += 1
                    all_predictions.append("correct")
            elif response == misleading_label:
                n_misled += 1
                all_predictions.append("misled")
            else:
                n_incorrect_out_choices += 1
                all_predictions.append("incorrect-out")
        else:
            if response in choices:
                response = classes.index(response)
                if response == target_label:
                    n_correct += 1
                    all_predictions.append("correct")
                elif response == misleading_label:
                    n_misled += 1
                    all_predictions.append("misled")
                else:
                    n_incorrect_in_choices += 1
                    all_predictions.append("incorrect-in")
            else:
                n_incorrect_out_choices += 1
                all_predictions.append("incorrect-out")



    return {
        "n_correct": n_correct,
        "n_misled": n_misled,
        "n_incorrect_in_choices": n_incorrect_in_choices,
        "n_incorrect_out_choices": n_incorrect_out_choices,
    }, all_predictions
