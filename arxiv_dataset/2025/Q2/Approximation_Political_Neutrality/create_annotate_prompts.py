# This file creates the prompts for GPT4-o to annotate the approximation techniques used in the model response. 
import pandas as pd
from src.eval_templates import create_prompt, process_r1

def create_eval_prompts(data, model, output_file, n_evals = 1):
    """ 
    data(df): should include the following keys: input,label,task,original_input,format_number,response,model
    model (str): the model used to create the generations (note if R1, use "R1" specifically)
    output_file (str): path of where the final prompts list should be saved
    n_evals (int): number of evaluation prompts to create per input/output pair
    Return: None (saved to output_file)
    """
    # Replace Nan (only for Gemini not responding for safety)
    data['response'].fillna('No Response', inplace = True)

    # Download input_num to properly classify each input, give unique name
    data_input_num_df = pd.read_csv("~/Data/test_data_input_num_ls.csv")
    data['input_num'] = data_input_num_df['input_num']

    eval_prompts = []
    eval_num_ls = []
    data_copy = data.copy().reset_index(drop=True)
    # create repeats of data if more than 1 eval wanted
    data_repeat = data_copy.loc[data_copy.index.repeat(n_evals)].reset_index(drop=True)
    for i in range(data.shape[0]):
        if model == "R1":
            response = process_r1(data['response'][i]) #only R1
        else:
            response = data['response'][i]
        prompts = create_prompt(data['input'][i], response, n_evals) #only R1
        # prompts = create_prompt(data['input'][i], data['response'][i], n_evals)
        eval_prompts.extend(prompts)
        eval_num_ls.extend(range(n_evals))
    data_repeat['prompt'] = eval_prompts
    data_repeat['eval_num'] = eval_num_ls

    # Save final prompts
    data_repeat.to_csv(output_file, index=False)


