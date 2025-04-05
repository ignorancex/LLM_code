from fuzzywuzzy import fuzz
def get_fuzzy_match(arg1, arg2):
    return fuzz.token_set_ratio(arg1.lower(), arg2.lower())

#cosine similarity check
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

model_args = ModelArgs(max_seq_length=156)

model = RepresentationModel(
    "bert",
    "bert-base-uncased",
    args=model_args,)

from sklearn.metrics.pairwise import cosine_similarity as cos

def get_cos(arg1, arg2):
    # Two lists of sentences
    
    sent_list = [arg1, arg2]
    embed = model.encode_sentences(sent_list, combine_strategy="mean")

    return cos(embed)[0][1]


def explicit(arg1, arg2):

    fuzz_ratio = get_fuzzy_match(arg1, arg2)
    cos_sim = get_cos(arg1, arg2)

    if fuzz_ratio >= 75.0:
        #print('Fuzzy....',arg1, arg2, fuzz_ratio)
        return True
    #elif cos_sim >= 0.70:
    #    print('Cos....', arg1, arg2, cos_sim)
    #    return True

    return False
