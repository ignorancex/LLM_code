from ..PEFT import config
from ..PEFT.inference import NegationModel
from ..negator_wrapper import Negator
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from ..generation_processing import remove_blanks
from ..PEFT.data_preprocess import Special_tokens
from .eval_metrics import compute_closeness,compute_self_bleu
from .perplex_filter import compute_sent_perplexity
from tqdm import tqdm
from .some import Scorer

import numpy as np

def save_sentences_to_file(sentences: List[str], file_path: str) -> None:
    """
    Save a list of sentences to a text file.
    
    Args:
        sentences (list of str): List of sentences to be written to file.
        file_path (str): Path to the file where sentences will be stored.
    """
    with open(file_path, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')  # Write each sentence followed by a newline


def query_model(test_data: List[str]) -> Tuple[List[str], List[str]]:

    """
    Query a trained negation model with a list of test data and return the generated outputs.

    Args:
        test_data (List[str]): A list of strings representing the test data to be fed into the model.

    Returns:
        Tuple[List[str], List[str]]:
            - A list of generated outputs from the PEFT (Parameter Efficient Fine-Tuning) model.
            - A list of generated outputs from the original model.
    """
        
    generations_original = []
    generations_peft = []

    print("\n Evaluating the model")
    directory = "negator/PEFT/peft_outputs"
    negation_model = NegationModel(directory)

    #create batches
    test_dataloader = DataLoader(test_data, batch_size=16, drop_last=False)

    # Query the model
    for batch in tqdm(test_dataloader, desc="Processing Batches"):
        outputs = negation_model.infer(batch, num_beams=5, num_return_sequences=3)
        generations_peft.extend(outputs["trained_model_outputs"])
        generations_original.extend(outputs["original_model_outputs"])

    return generations_peft,generations_original

def calculate_some(negations: List[str], file_path: str) -> None:
    """
    Evaluate a list of negations using a predefined scoring model fluency, grammaticality 
    and print the results.

    Args:
        negations (List[str]): A list of negation strings to be evaluated by the scoring model.
        file_path (str): The path to the file where evaluation results will be saved.

    Returns:
        None
    """
    path = "negator/Evaluation/gfm-models"
    scorer = Scorer(models_path=path, save_to_file=True,file_path=file_path)
    scorer.add(negations)
    scores = scorer.score()

    # Print each metric and its score
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")

def evaluate_model(test_data: List[str], file_name: str = "") -> Dict[str, float]:
    """
    Evaluate the performance of a negation model by comparing generated text from two models 
    (PEFT and original polyjuice) using various metrics. Results are saved to files and printed to the console.

    Args:
        test_data (List[str]): A list of text prompts to generate and evaluate.
        file_name (str, optional): A suffix to append to filenames where results are saved. Defaults to an empty string.

    Returns:
        Dict[str, float]: A dictionary containing average scores for different metrics:
            - 'avg_tree_peft': Average syntactic tree distance for the PEFT model.
            - 'avg_lev_peft': Average Levenshtein distance for the PEFT model.
            - 'avg_tree_orig': Average syntactic tree distance for the original model.
            - 'avg_lev_orig': Average Levenshtein distance for the original model.
            - 'avg_bleu_orig': Average BLEU score for the original model.
            - 'avg_bleu_peft': Average BLEU score for the PEFT model.
    """
    org_tree_dist, org_lev_dist = [], []
    peft_tree_dist, peft_lev_dist = [], []
    peft_bleu, org_bleu = [], []
    gen_org_clean, gen_peft_clean = [], []
    #poly_ppl, peft_ppl = [],[]

    previous_original = None
    grouped_gen_docs = []
    grouped_poly_docs = []

    generations_peft,generations_original = query_model(test_data)

    negator_object = Negator()

    # Process the generated text: remove blanks, lowercase, 
    for index in range(len(generations_original)):
        replaced_peft, _ = remove_blanks(generations_peft[index])
        gen_peft_clean.append(replaced_peft)
        replaced_original, _ = remove_blanks(generations_original[index])
        gen_org_clean.append(replaced_original)
        original = generations_peft[index].split(Special_tokens.PERETURB_TOK)[0]

        # Convert text to doc
        ref_doc = negator_object._process(original.lower())
        gen_doc = negator_object._process(replaced_peft.lower())
        org_doc = negator_object._process(replaced_original.lower())

        # Check if the original is the same as the previous one
        if original == previous_original:
            # Expand the last sublist by adding gen_doc to the last group
            grouped_gen_docs[-1].extend([gen_doc])
            grouped_poly_docs[-1].extend([org_doc])
        else:
            # Original is different; create a new sublist with the gen_doc
            grouped_gen_docs.append([gen_doc])
            grouped_poly_docs.append([org_doc])

        # Update previous_original for the next iteration
        previous_original = original

        # Compute perplexity
        #poly_ppl.append(negator_object.calculate_sent_perplexity(replaced_original))
        #peft_ppl.append(negator_object.calculate_sent_perplexity(replaced_peft))

        # Compute closeness and diversity metric with trained model
        closeness = compute_closeness([gen_doc],ref_doc)
        peft_tree_dist.append(closeness['tree_dist'])
        peft_lev_dist.append(closeness['edit_dist'])

        # Compute closeness and diversity metric with original model
        closeness = compute_closeness([org_doc],ref_doc)
        org_tree_dist.append(closeness['tree_dist'])
        org_lev_dist.append(closeness['edit_dist'])

    for index in range(len(grouped_gen_docs)):
        diversity = compute_self_bleu(grouped_gen_docs[index])
        peft_bleu.append(diversity['bleu4'])
        diversity = compute_self_bleu(grouped_poly_docs[index])
        org_bleu.append(diversity['bleu4'])

    poly_ppl = compute_sent_perplexity(gen_org_clean,batch_size = 1)
    peft_ppl = compute_sent_perplexity(gen_peft_clean,batch_size = 1)


    avg_tree_peft = np.mean(peft_tree_dist)
    avg_lev_peft = np.mean(peft_lev_dist)
    avg_tree_orig = np.mean(org_tree_dist)
    avg_lev_orig = np.mean(org_lev_dist)
    avg_bleu_orig = np.mean(org_bleu)
    avg_bleu_peft = np.mean(peft_bleu)
    avrg_ppl_orig = np.mean(poly_ppl)
    avrg_ppl_peft = np.mean(peft_ppl)


    folder = "reports"
    # Save cleaned sentences to a text file
    file_path =  f"{folder}/sentences_peft_{file_name}.txt"
    save_sentences_to_file(gen_peft_clean, file_path)

    # Save cleaned sentences to a text file
    file_path =  f"{folder}/sentences_poly_{file_name}.txt"
    save_sentences_to_file(gen_org_clean, file_path)

    print('### Polyjuice ########')
    print(f"Average Syntactic Tree Distance (Poly): {avg_tree_orig:.3f}")
    print(f"Average Levenshtein Distance (Poly): {avg_lev_orig:.3f}")
    print(f"Average Bleu Score (Poly): {avg_bleu_orig:.3f}")
    print(f"Average PPL Score (Poly): {avrg_ppl_orig:.3f}")
    calculate_some(gen_org_clean,file_path = f"{folder}/scores_poly_{file_name}.txt")

    print('\n### PEFT ########')
    print(f"Average Syntactic Tree Distance (PEFT): {avg_tree_peft:.3f}")
    print(f"Average Levenshtein Distance (PEFT): {avg_lev_peft:.3f}")
    print(f"Average Bleu Score (PEFT): {avg_bleu_peft:.3f}")
    print(f"Average PPL Score (PEFT): {avrg_ppl_peft:.3f}")
    calculate_some(gen_peft_clean,file_path =  f"{folder}/scores_peft_{file_name}.txt")
    

    return {
    'avg_tree_peft': avg_tree_peft,
    'avg_lev_peft': avg_lev_peft,
    'avg_tree_orig': avg_tree_orig,
    'avg_lev_orig': avg_lev_orig,
    'avg_bleu_orig': avg_bleu_orig,
    'avg_bleu_peft': avg_bleu_peft 
    }

    





    
