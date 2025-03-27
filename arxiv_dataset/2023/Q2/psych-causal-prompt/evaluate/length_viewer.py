from warping_dataset_like_GPT2 import GPT3TokenizerWarper, generate_prompts
import pandas as pd
from icecream import ic

if __name__=='__main__':

    random_tokenizer = GPT3TokenizerWarper(["", ""])
    df = pd.read_csv('finalized_short_prompt.csv')
    perfix_list, postfix_list = generate_prompts(df, 0)
    for i in range(3):
        ic(random_tokenizer.get_length(perfix_list[i])+random_tokenizer.get_length(postfix_list[i]))
    df = pd.read_csv('finalized_prompt.csv')
    perfix_list, postfix_list = generate_prompts(df, 0)
    for i in range(3):
        ic(random_tokenizer.get_length(perfix_list[i])+random_tokenizer.get_length(postfix_list[i]))
