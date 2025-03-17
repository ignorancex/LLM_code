
import pandas as pd
from negator import Negator
from negator.PEFT.data_preprocess import process_and_blank_sentences
from negator.Evaluation.evaluate import evaluate_model

# Path to the dataset for evaluation
data_path = './negator/data/nli/SNLI.txt'

# Preprocess the data with token-only blanks for intrinsic evaluation
# This evaluates the model's performance on token-only blanked sentences
test_data  = process_and_blank_sentences(data_path, sample_size=300, max_blank=2,max_sent=4, is_token_only= True)
print("\nModel with token only")
evaluate_model(test_data, file_name="token")

# Preprocess the data with subtree-based blanks for intrinsic evaluation
# This evaluates the model's performance on subtree blanked sentences
test_data  = process_and_blank_sentences(data_path, sample_size=300, max_blank=2,max_sent=4, is_token_only= False)
print("\nModel with subtree")
evaluate_model(test_data, file_name="subtree")