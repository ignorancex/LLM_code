
import pandas as pd
from negator import Negator
from negator.generation_processing import get_prompts
from negator.PEFT.data_preprocess import process_and_blank_sentences
from negator.generation_processing import get_outputs,remove_blanks

def generate_and_clean_predictions(negator_aug, blanks, get_outputs):

    """
    Generates and cleans predictions by filling in blanks in the provided sentences.

    This function takes a list of blanked sentences, generates possible completions 
    using a given model, and then cleans the predictions by removing any residual 
    blanks or special tokens.

    Args:
        negator_aug: 
            The augmentation object containing the model and tokenizer.
        blanks (List[str]): 
            A list of sentences with blanks to be filled by the model.
        get_outputs (Callable): 
            A function that generates predictions from the model given the input prompts.

    Returns:
        None: 
            The function prints the cleaned predictions.
    """

    device = negator_aug.device
    input_prompt = negator_aug.tokenizer(list(blanks), return_tensors='pt', padding=True, truncation=True).to(device)
    outputs_original_model = get_outputs(negator_aug.model, input_prompt, num_beams=5, num_return_sequences=3)
    preds_list = negator_aug.tokenizer.batch_decode(outputs_original_model, skip_special_tokens=True)

    preds_list_cleaned = []

    for sequence in preds_list:
        normalized, _ = remove_blanks(sequence)
        preds_list_cleaned.append(normalized)


    print(preds_list_cleaned)


negator_aug = Negator()

text = "Everybody loves the coffee in London"
print(text)

blanks = negator_aug.get_random_blanked_sentences(text,max_blank_sent_count=6, max_blank_block = 2,is_token_only = True)
print(blanks)


print("############################# NegVerse ##############")
perturbations = negator_aug.perturb(text, blanked_sent = blanks, num_beams=5)
print("\n")
print(perturbations)


print("############################# Polyjuice #################")

generate_and_clean_predictions(negator_aug, blanks, get_outputs)