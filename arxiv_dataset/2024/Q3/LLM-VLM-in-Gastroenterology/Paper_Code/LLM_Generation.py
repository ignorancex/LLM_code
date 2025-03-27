# Context: I am using Windows 11, cuda-enabled Laptop GPU RTX 3080Ti (16GB memory) and 12gen Core i9-12900HX. I have used Python 3.6 and 3.8.
# Note for reuse: As the project took 5 months, I have used different versions of openai and Langchain (v0.1), you may face errors on the updated libraries. 
#    Sorry for that, but the libraries were under development. You can debug it using their documentation (hopefully).
# Table of Content:
# Micro functions --> Line 28
# OpenAI Open call  --> Line 238
# OpenAI Function Call  --> Line 453
# Old GPT --> Line 769
# Claude Structured Output --> Line 1191
# Groq API --> Line 1481
# Poe Wrapper --> Line 1663
# Local LLM with LM studio --> Line 1840
# Clean Errors --> Line 2023

#----------- Installing packages -------------
#[cmd] conda activate .conda
# %pip install --upgrade pip
# %pip install cloudinary -U
# %pip install pandas -U
# %pip install python-dotenv -U
# %pip install replicate -U
# %pip install langchain-community -U
# %pip install langchain -U
# %pip install openai -U
# %pip install -U langchain-openai


#----------- Micro-Functions -------------
import pandas as pd
import numpy as np
import os

def save_and_open_excel(df, excel_output_path, open_at_end):
    try:
        df.to_excel(excel_output_path, index=False)
        print(f'Saved the excel file at {excel_output_path}')
    except Exception as e:
        print(f"Error saving the file to excel: {e}")
        return df

    if os.path.exists(excel_output_path) and open_at_end is True:
        try:
            os.startfile(excel_output_path)
        except Exception as e:
            print(f"Error opening excel: {e}")
            return df

    return df


def analyze_model_accuracy(excel_path):
    # Load the Excel file
    df = pd.read_excel(excel_path)
    
    # Filter columns that end with '_correctness'
    correctness_columns = [col for col in df.columns if col.endswith('_correctness')]
    
    # Prepare a dictionary to hold results
    results = {
        'Model Name': [],
        'Oall_Accu': []
    }
    
    # Analyze each model's performance
    for col in correctness_columns:
        # Extract the model name (the first component of the column name)
        model_name = col.split('_')[0]
        
        # Count correct and incorrect answers
        correct_count = df[col].value_counts().get('correct', 0)
        incorrect_count = df[col].value_counts().get('incorrect', 0)
        total = df.shape[0]
        total_answered = correct_count + incorrect_count
        
        # Calculate Oall_Accu
        accuracy = (correct_count / total) * 100
        
        
        # Format the results
        accuracy_str = f'{accuracy:.2f}% ({correct_count}-of-{total_answered}; Error: {total-total_answered})'
        
        # Append the results
        results['Model Name'].append(model_name)
        results['Oall_Accu'].append(accuracy_str)

        
    # Convert results dictionary to a DataFrame for nicer display
    results_df = pd.DataFrame(results)
    
    # Display the results table
    #print(results_df)
    
    return results_df


def show_models_performance(excel_file_path):
    data = pd.read_excel(excel_file_path)
    data['Category'] = data['Category'].str.lower().str.strip()
    data_images = data.copy()
    data = data[data['Number of image'] == 0]
    
    # Identify model names and relevant columns
    model_columns = [col for col in data.columns if 'correctness' in col]
    models = [col.replace('_correctness', '') for col in model_columns]
    
    # Initialize dictionaries to hold accuracies
    overall_accuracy = {}
    accuracy_2022 = {}
    accuracy_2023 = {}
    
    # Calculate accuracies
    for model in models:
        correctness_col = f'{model}_correctness'
        correct_responses = data[correctness_col] == 'correct'
        
        
        # Overall accuracy
        overall_accuracy[model] = correct_responses.mean()
        
        # Yearly accuracies
        accuracy_2022[model] = data[data['Year'] == 2022][correctness_col].eq('correct').mean()
        accuracy_2023[model] = data[data['Year'] == 2023][correctness_col].eq('correct').mean()
    
    # Convert accuracies to DataFrame for Table 1
    table_1 = pd.DataFrame([overall_accuracy, accuracy_2022, accuracy_2023], 
                           index=['Overall Accuracy', '2022 Test Accuracy', '2023 Test Accuracy'])
    
    # Convert accuracy values to percentages with one decimal place for both tables
    table_1_percentage = table_1.map(lambda x: f'{x:.1%}')
    
    display(HTML(table_1_percentage.to_html()))
    return table_1_percentage

def check_answer_correctness(answer, truth, reporterror_index=None,reporterror_name=None):
    if reporterror_index is None:
        reporterror_index=r'(IDK the index)'
    if reporterror_name is None:
        reporterror_name =r'(IDK the llm name)'
        
    valid_choices = ['A', 'B', 'C', 'D', 'E']

    # Normalize inputs by removing the word 'option' and converting to upper case
    norm_answer = str(answer).replace('choice', '').strip().upper()[0] if answer else ''
    norm_truth = str(truth).replace('choice', '').strip().upper()[0] if truth else ''
    # Normalize inputs by getting the first string since mistral will add option string after the option
    
    # Check if both answer and truth are valid options
    if norm_answer not in valid_choices:
        return f'''WARNING: The answer "{norm_answer}" is not a valid option, even after normalization (removing word option, and getting the first string only). 
                    INDEX:{reporterror_index} , LLM:{reporterror_name}.
                    ANSWER::: {answer}
                    TRUTH::: {truth} '''
    if norm_truth not in valid_choices:
        return f'''WARNING: The truth "{norm_truth}" is not a valid option, even after normalization (removing word option, and getting the first string only).
                    INDEX:{reporterror_index} , LLM:{reporterror_name}
                    ANSWER::: {answer}
                    TRUTH::: {truth} '''

    # Check correctness
    if 'ERROR' in answer:
        # for cases that extraction of answer caused error
        return 'ERROR'
    
    return 'correct' if norm_answer == norm_truth else 'incorrect'


def show_models_performance(excel_file_path):
    data = pd.read_excel(excel_file_path)
    data['Category'] = data['Category'].str.lower().str.strip()
    data_images = data.copy()
    data = data[data['Number of image'] == 0]
    
    # Identify model names and relevant columns
    model_columns = [col for col in data.columns if 'correctness' in col]
    models = [col.replace('_correctness', '') for col in model_columns]
    
    # Initialize dictionaries to hold accuracies
    overall_accuracy = {}
    accuracy_2022 = {}
    accuracy_2023 = {}
    
    # Calculate accuracies
    for model in models:
        correctness_col = f'{model}_correctness'
        correct_responses = data[correctness_col] == 'correct'
        
        
        # Overall accuracy
        overall_accuracy[model] = correct_responses.mean()
        
        # Yearly accuracies
        accuracy_2022[model] = data[data['Year'] == 2022][correctness_col].eq('correct').mean()
        accuracy_2023[model] = data[data['Year'] == 2023][correctness_col].eq('correct').mean()
    
    # Convert accuracies to DataFrame for Table 1
    table_1 = pd.DataFrame([overall_accuracy, accuracy_2022, accuracy_2023], 
                           index=['Overall Accuracy', '2022 Test Accuracy', '2023 Test Accuracy'])
    
    # Convert accuracy values to percentages with one decimal place for both tables
    table_1_percentage = table_1.map(lambda x: f'{x:.1%}')
    
    display(HTML(table_1_percentage.to_html()))
    return table_1_percentage

def check_answer_correctness(answer, truth, reporterror_index=None,reporterror_name=None):
    if reporterror_index is None:
        reporterror_index=r'(IDK the index)'
    if reporterror_name is None:
        reporterror_name =r'(IDK the llm name)'
        
    valid_choices = ['A', 'B', 'C', 'D', 'E']

    # Normalize inputs by removing the word 'option' and converting to upper case
    norm_answer = str(answer).replace('choice', '').strip().upper()[0] if answer else ''
    norm_truth = str(truth).replace('choice', '').strip().upper()[0] if truth else ''
    # Normalize inputs by getting the first string since mistral will add option string after the option
    
    # Check if both answer and truth are valid options
    if norm_answer not in valid_choices:
        return f'''WARNING: The answer "{norm_answer}" is not a valid option, even after normalization (removing word option, and getting the first string only). 
                    INDEX:{reporterror_index} , LLM:{reporterror_name}.
                    ANSWER::: {answer}
                    TRUTH::: {truth} '''
    if norm_truth not in valid_choices:
        return f'''WARNING: The truth "{norm_truth}" is not a valid option, even after normalization (removing word option, and getting the first string only).
                    INDEX:{reporterror_index} , LLM:{reporterror_name}
                    ANSWER::: {answer}
                    TRUTH::: {truth} '''

    # Check correctness
    if 'ERROR' in answer:
        # for cases that extraction of answer caused error
        return 'ERROR'
    
    return 'correct' if norm_answer == norm_truth else 'incorrect'


#----------- LLM Generation: OpenAI Open call  -------------

from openai import OpenAI, AsyncOpenAI
import json
import os
import asyncio
from IPython.display import HTML, display #to display performance tables

async def OpenAI_QA_OpenandExtract(question: str, choices: str, model_name: str, openai_api, 
                                         overall_prompt:str,
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):
    """
    Answer a medical question using the OPENAI language model asynchronously.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    
    if not hasattr(openai_answer_medical_question_openthenextraact, 'has_run_before') or not openai_answer_medical_question_openthenextraact.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        openai_answer_medical_question_openthenextraact.has_run_before = True

    try:
        
        client = AsyncOpenAI(api_key=openai_api)
        
        response  = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", 
                "content": f"""{overall_prompt}
                
                Question:
                {question}
                Choices:
                {choices}
                """
                }
                ],
            max_tokens=model_max_tokens,
            temperature=model_tempreature,
            logprobs=False,
            timeout=request_timeout
            )

        correct_answer_schema = [
        {
            "type": "function",
            "function": {
                "name": "extract_true_choice",
                "description": f"""What is the selected option  
                Provided options were: 
                {question}""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected option": {
                            "type": "string",
                            "description": "The selected option from the list of provided choices.",
                            "enum": ["A", "B", "C", "D", "E"]
                        },
                    },
                    "required": ["selected option"],
                },
            }
        }]
        
        try:
            extraction_response = await client.chat.completions.create(
                model = 'gpt-3.5-turbo',
                messages = [{'role': 'user', 'content': response.choices[0].message.content}],
                tools= correct_answer_schema,
                tool_choice=  'auto',
                temperature=0
            )
            correct_answer_dic=json.loads(extraction_response.choices[0].message.tool_calls[0].function.arguments)
            structure_answer=correct_answer_dic['selected option']
        except:
            extraction_response=structure_answer='ERROR in structure extraction'
    except:
        Experiment_detail=response=extraction_response=structure_answer='ERROR in response generation'


        
    return Experiment_detail, response, extraction_response, structure_answer



async def handle_llm_response_openthenextract(excel_file_path: str, llm_list: list, openai_api,  overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,

                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512,
                              
                               ):
    
    list_of_vision_models=['gpt-4-0125-preview']
    list_of_text_models=['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview','gpt-3.5-turbo-0125'
                         "mistral8x7b",
                         "llama2-70b"]
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    async def process_row(row, llm_name, idx):
        answer = certainty = rationale = raw = correctness = None
        if llm_name in ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview','gpt-3.5-turbo-0125'):
            Experiment_detail, response, extraction_response, structure_answer = await OpenAI_QA_OpenandExtract(question=row['Question'], choices=row['Options'], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, 
                                                                                                overall_prompt=overall_prompt)
            correctness = check_answer_correctness(truth=row['Correct Answer'], answer=structure_answer)
            return idx, Experiment_detail, response, extraction_response, structure_answer, correctness
        else:
            return idx, 'ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname'
 
    # Loop through llm list
    for llm in llm_list:
        # Create a column for each llm (to store response) if it doesn't exist
        if llm not in df.columns:
            df[llm] = ''
        
        tasks = []
        
        i=0
        max_index = df.index.max()
        for index, row in df.iterrows():
            if row[llm] != 'EXTRACTED': 

                task = asyncio.create_task(process_row(row, llm, idx=index))
                tasks.append(task)
                
                #saving output after finishing number_of_task_to_save tasks    
                if i==number_of_task_to_save or index == max_index: 
                    results = await asyncio.gather(*tasks)
                    
                    for result in results:
                        if result:  # Ensure result is not None or handle as needed
                            
                            idx, Experiment_detail, response, extraction_response, structure_answer, correctness = result
                            # Update the DataFrame based on the result
                            df.at[idx, llm] = 'EXTRACTED'
                            df.at[idx, f'{llm}_rawoutput'] = str(response) 
                            df.at[idx, f'{llm}_extraction_response'] = str(extraction_response) 
                            df.at[idx, f'{llm}_answer'] = str(structure_answer)
                            df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                            df.at[idx, f'{llm}_correctness'] = correctness
                            
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    
                    if show_performance_during_loop:
                        try:
                            show_models_performance(excel_file_path)
                        except Exception as e:
                            print(f"Error in showing performance during loop. Error:   {e}")
                            continue
                        
                    
                    #reset for continue
                    i=0
                    tasks = []
                    
                        
    df = save_and_open_excel(df, excel_file_path, open_at_end)
    
    if show_performance_at_end:
        try:
            table_1_percentage= show_models_performance(excel_file_path)
            print(table_1_percentage)
        except Exception as e:
            print(f"Error in showing the performance. Error:   {e}")
    
    return df


# Example use
#overall_prompt='Answer this question and select one option.'
#model_tempreature=0
#model_max_tokens=512

#question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
#choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
#model_name='gpt-3.5-turbo'
#openai_api=os.getenv('OPENAI_API_KEY')
#request_timeout=60
#
#Experiment_detail, response, correct_answer, structure_answer= openai_answer_medical_question_openthenextraact(question=question, choices=choices, model_name=model_name, openai_api=openai_api, 
#                                                                                            overall_prompt=overall_prompt,
#                                                                                            model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                                                                            request_timeout=request_timeout)



#----------- LLM Generation: OpenAI Function Call  -------------
import shutil
import time
from openai import  AsyncOpenAI
import json
import os
import asyncio

async def OpenAI_QA_FunctionCall(question: str, choices: str, model_name: str, openai_api, 
                                         overall_prompt:str,
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30,
                                         use_seed: bool=False):
    """
    Answer a medical question using the OPENAI language model asynchronously.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens
    Experiment_detail['seed']= 123 if use_seed else 'None'

    
    if not hasattr(OpenAI_QA_FunctionCall, 'has_run_before') or not OpenAI_QA_FunctionCall.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        OpenAI_QA_FunctionCall.has_run_before = True

    try:
    

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "structuring_output",
                    "description": "You can generate your unstructured output and selected option. ",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unstructured output": {
                                "type": "string",
                                "description": 'Your unstructured output for the answer, if needed.',
                            },
                            "selected option": {"type": "string",  
                                                "description": "The selected option from the list of provided choices. (Example: 'F')",
                                                "enum": ["A", "B", "C", "D", "E"]},
                        },
                        "required": ["unstructured output","selected option"],
                    },
                },
            }
        ]
        
        messages=[
            {"role": "user", 
             "content": f"""
             {overall_prompt}
             
             Question:
             {question}
             Choices:
             {choices}
             """  }]
        client = AsyncOpenAI(api_key=openai_api)

        if use_seed:
            seed_value = 123
        else:
            seed_value = None
        
        response  = await client.chat.completions.create(
            model=model_name,
            messages=messages,  
            max_tokens=model_max_tokens,
            temperature=model_tempreature,
            tools= tools,
            logprobs=False,
            timeout=request_timeout, 
            tool_choice= {"type": "function", "function": {"name": "structuring_output"}},
            seed=seed_value
            )
        
        try:
            correct_answer_dic=json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            structure_answer=correct_answer_dic["selected option"]
        except Exception as ee:
            structure_answer=f'ERROR in extracting selected answer: {ee}'
    except Exception as e:
        Experiment_detail=response=structure_answer=f'ERROR in response generation: {e}'

    return Experiment_detail, response, structure_answer



def save_and_open_excel(df, excel_output_path, open_at_end):
    try:
        df.to_excel(excel_output_path, index=False)
        print(f'Saved the excel file at {excel_output_path}')
    except Exception as e:
        print(f"Error saving the file to excel: {e}")
        return df

    if os.path.exists(excel_output_path) and open_at_end is True:
        try:
            os.startfile(excel_output_path)
        except Exception as e:
            print(f"Error opening excel: {e}")
            return df

    return df


async def handle_llm_response_functioncall(excel_file_path: str, llm_list: list, openai_api,  overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, add_delay_sec:int=1,
                              request_timeout:int=15,

                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512, max_token_plus_input=False,
                              use_seed: bool=False,
                               ):
    

    
    try:
        
        list_of_vision_models=['gpt-4-0125-preview']
        list_of_text_models=['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview','gpt-3.5-turbo-0125'
                            "mistral8x7b",
                            "llama2-70b"]
        # Read excel
        df = pd.read_excel(excel_file_path,)
        
        async def process_row(row, llm_name, idx, model_max_tokens, max_token_plus_input,):
            if max_token_plus_input:
                input_token_count= row['input_token_count']
                
                max_token=input_token_count+model_max_tokens
                max_token=int(max_token)
            else:
                max_token=int(model_max_tokens)
            Experiment_detail= response=structure_answer= correctness = None
            
            if llm_name in ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview', 'gpt-4-1106-preview','gpt-3.5-turbo-0125', 'gpt-4o-mini-2024-07-18'):
                Experiment_detail, response, structure_answer = await OpenAI_QA_FunctionCall(question=row['Question'], choices=row['Options'], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
                                                                                                    model_tempreature=model_tempreature, model_max_tokens=max_token, 
                                                                                                    overall_prompt=overall_prompt, use_seed=use_seed)
                correctness = check_answer_correctness(truth=row['Correct Answer'], answer=structure_answer)
                return idx, Experiment_detail, response, structure_answer, correctness
            else:
                return idx, 'ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname'
    
        # Loop through llm list
        for llm in llm_list:
            # Create a column for each llm (to store response) if it doesn't exist
            if llm not in df.columns:
                df[llm] = ''
            
            tasks = []
            
            i=0
            max_index = df.index.max()
            for index, row in df.iterrows():
                if row[llm] != 'EXTRACTED':

                    task = asyncio.create_task(process_row(row, llm, idx=index,model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input))
                    tasks.append(task)
                    i+=1
                    
                    #saving output after finishing number_of_task_to_save tasks    
                    if i==number_of_task_to_save or index == max_index: 
                        results = await asyncio.gather(*tasks)

                        for result in results:
                            if result:  # Ensure result is not None or handle as needed

                                idx, Experiment_detail, response, structure_answer, correctness = result
                                
                                # Update the DataFrame based on the result
                                df.at[idx, llm] = 'EXTRACTED'
                                df.at[idx, f'{llm}_rawoutput'] = str(response) 
                                df.at[idx, f'{llm}_answer'] = str(structure_answer)
                                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                                df.at[idx, f'{llm}_correctness'] = correctness
                                
                        #save draft
                        try:
                            df.to_excel(excel_file_path)
                            print(f"Draft excel file saved at {excel_file_path}")
                        except Exception as e:
                            print(f"Error in saving temporary excel. Error:   {e}")
                            continue
                        
                        if show_performance_during_loop:
                            try:
                                temptable=analyze_model_accuracy(excel_file_path)
                                print(temptable)
                            except Exception as e:
                                print(f"Error in showing performance during loop. Error:   {e}")
                                continue
                            
                        
                        #reset for continue
                        i=0
                        tasks = []
                        print('sleep like a baby')
                        await asyncio.sleep(add_delay_sec)
            
            
            
                            
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        OpenAI_QA_FunctionCall.has_run_before = False
        return df
    
    except KeyboardInterrupt or asyncio.CancelledError:
        print("Operation interrupted. Cleaning up...")
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        OpenAI_QA_FunctionCall.has_run_before = False
        return df
        

# Example Usage
#overall_prompt='Answer this question and select one option.'
#model_tempreature=0
#model_max_tokens=2048
#question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
#choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
#
#model_name='gpt-3.5-turbo'
#openai_api=os.getenv('OPENAI_API_KEY')
#request_timeout=60
#
#Experiment_detail, response, structure_answer= await OpenAI_QA_FunctionCall(question=question, choices=choices, model_name=model_name, openai_api=openai_api, 
#                                                                                            overall_prompt=overall_prompt,
#                                                                                            model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                                                                            request_timeout=request_timeout)

# Example of final run


# #excel_file_path = r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_all.xlsx"
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_round_Julyy.xlsx"

# overall_prompt="Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. If the question has images, ignore it and answer based on the text only. Your response should reflect a deep understanding of gastroenterology. "

                        
# llm_list=[#'gpt-4-0613', 
#         #'gpt-3.5-turbo-1106', #Up to Sep 2021
#         #'gpt-3.5-turbo-0613',#new version: June 13th 2023      -> #temp0DONE #temp0.2DONE
#         #'gpt-3.5-turbo-0125',# New Updated GPT 3.5 Turbo   #NEW model
#         #'gpt-4-0125-preview', # understand image,
#         #'gpt-4o-2024-05-13'
#         'gpt-4o-mini-2024-07-18'
#         ]

# openai_api=os.getenv('OPENAI_API_KEY')

# open_at_end = False
# number_of_task_to_save=5 #12 gpt3.5 
# add_delay_sec=10 #20 gpt3.5
# request_timeout=35
# return_tokentime=True

# show_performance_at_end=True
# show_performance_during_loop=True


# model_temperature=1
# model_max_tokens=512
# max_token_plus_input=True


# final_df= await handle_llm_response_functioncall(excel_file_path=excel_file_path, llm_list=llm_list, openai_api=openai_api, 
#                         open_at_end=open_at_end,
#                         number_of_task_to_save=number_of_task_to_save, 
#                         request_timeout=request_timeout, add_delay_sec=add_delay_sec,

#                         show_performance_at_end=show_performance_at_end,
#                         show_performance_during_loop=show_performance_during_loop,
                        
#                         model_tempreature=model_temperature, model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input,
                        
#                         overall_prompt=overall_prompt)


#----------- LLM Generation:  Old GPT-------------
from openai import OpenAI
import pandas as pd
import os 
# OPENAI - TEXT ---------------------------------------------
def OpenAI_QA_Old(question: str, choices: str, model_name: str, openai_api, 
                                         overall_prompt:str,
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):

    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    if not hasattr(OpenAI_QA_Old, 'has_run_before') or not OpenAI_QA_Old.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        OpenAI_QA_Old.has_run_before = True

    try:
    
        client = OpenAI()

        prompt=f"""
        {overall_prompt}
        Question:
        {question}
        Choices:
        {choices}
        """

        completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=model_tempreature,
        max_tokens=model_max_tokens,
        timeout=request_timeout

        )
        rawoutput=completion.choices[0].text
        
    except:
        Experiment_detail=completion=rawoutput='ERROR in response generation'
    return Experiment_detail, completion, rawoutput

def handle_old_OpenAI(excel_file_path: str, llm_list: list, openai_api,  overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,
                              
                              model_tempreature=0, model_max_tokens=512,
                              
                               ):
    
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    def process_row(row, llm_name, idx):
        if llm_name in ['gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002']:
            Experiment_detail, response, rawoutput=  OpenAI_QA_Old(question=row['Question'], choices=row['Options'], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, 
                                                                                                overall_prompt=overall_prompt)
            return idx, Experiment_detail, response, rawoutput
        else:
            return idx, 'ERRORinLLMname','ERRORinLLMname'
    try:
        # Loop through llm list
        for llm in llm_list:
            # Create a column for each llm (to store response) if it doesn't exist
            if llm not in df.columns:
                df[llm] = ''
            
            tasks = []
            
            i=0
            max_index = df.index.max()
            for index, row in df.iterrows():
                if row[llm] != 'EXTRACTED':

                    idx, Experiment_detail, response, rawoutput = process_row(row, llm, idx=index)
                    # Update the DataFrame based on the result
                    df.at[idx, llm] = 'EXTRACTED'
                    df.at[idx, f'{llm}_rawresponse'] = str(response) 
                    df.at[idx, f'{llm}_rawoutput'] = str(rawoutput) 
                    df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                                    
                    df.at[idx, f'{llm}_answer'] = None
                    df.at[idx, f'{llm}_correctness'] = None
                    print (rawoutput)
                    i+=1
                    
                #saving output after finishing number_of_task_to_save tasks    
                if i==number_of_task_to_save or index == max_index:            
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    
                    
                    #reset for continue
                    i=0
    except Exception as e:
        print(f'Erorr in handler: {e}')
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
    
    return df
                                 
#----------- LLM Generation: Langchain Structure -------------
async def Langchain_OpenAI_LCfunction(question: str, choices: str, model_name: str, openai_api, just_raw:bool,
                                         system_prompt:str,overall_prompt:str,answer_prompt:str,rationale_prompt:str,certainty_prompt:str, 
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         
                                         return_tokentime: bool=False, 
                                         request_timeout:int=30,
                                         ):
    """
    Answer a medical question using the OPENAI language model.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['system_prompt'] = system_prompt
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['answer_prompt'] = answer_prompt
    Experiment_detail['rationale_prompt'] = rationale_prompt
    Experiment_detail['certainty_prompt'] = certainty_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    
    if not hasattr(Langchain_OpenAI_LCfunction, 'has_run_before') or not Langchain_OpenAI_LCfunction.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        Langchain_OpenAI_LCfunction.has_run_before = True

    try:

        
        if just_raw is True:
            human_prompt = f"""Question:
            {overall_prompt}
            
            {question}
            Choices:
            {choices}
            """
            
            QA_schema_raw= {
                "title": "Answer Choice",
                "description": "You can generate your unstructured output and selected option. ",
                "type": "object",
                "properties": {
                    "unstructured output":
                        {
                            "title": "Choice",
                            "type": "string",
                            "description": 'Your unstructured output for the answer, if needed.',
                        },
                    "answer":
                        {
                            "title": "Choice",
                            "type": "string",
                            "description": "The selected option from the list of provided choices. Exactly output the option string from this list ['A', 'B', 'C', 'D', 'E'].",
                        },
                },  
                "required": ["answer"]
            }
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", " "),
                    ("human", human_prompt)
                ]
            )

            llm = ChatOpenAI(temperature=model_tempreature, model=model_name, api_key=openai_api, model_kwargs={"seed": 123}, max_tokens=model_max_tokens, 
                            timeout=request_timeout)
            
            if return_tokentime:
                
                with get_openai_callback() as cb:
                    time_start = monotonic()
                    chain = create_structured_output_runnable(QA_schema_raw, llm, prompt)
                    output = await chain.ainvoke({"question": question, "choices": choices})
                    time_end = monotonic()
                    time_duration = time_end - time_start
                    tokentime=f"""
                    Total Tokens: {cb.total_tokens}
                    Prompt Tokens: {cb.prompt_tokens}
                    Completion Tokens: {cb.completion_tokens}
                    Total Cost (USD): ${cb.total_cost}
                    Q&A time duration (s): {time_duration}
                    """
                    
                    return output['answer'], output['rationale'], output['certainty'], str(output), tokentime, str(Experiment_detail)
            else:
                chain = create_structured_output_runnable(QA_schema_raw, llm, prompt)
                output = await chain.ainvoke({"question": question, "choices": choices})
                return output.get('answer', ""), output.get('rationale', ""), output.get('certainty', ""), str(output), r"return_tokentime=False -> not stored", str(Experiment_detail) # this last empty string will be returned instead of tokentime
        
        else:
            human_prompt = f"""Question:
                {question}
                Choices:
                {choices}
                """
            
            QA_schema= {
                "title": "Answer",
                "description": overall_prompt,
                "type": "object",
                "properties": {
                    "answer":
                        {
                            "title": "Answer Choice",
                            "type": "string",
                            "description": answer_prompt,
                        },
                    "rationale":
                        {
                            "title": "Answer Rationale",
                            "type": "string",
                            "description": rationale_prompt
                        },
                    "certainty":
                        {
                            "title": "Answer Certainty",
                            "type": "integer",
                            "description": certainty_prompt
                        },
                },  
                "required": ["answer", "rationale", "certainty"]
            }

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", human_prompt)
                ]
            )
            
            llm = ChatOpenAI(temperature=model_tempreature, model=model_name, api_key=openai_api, model_kwargs={"seed": 123}, max_tokens=model_max_tokens, 
                            timeout=request_timeout)
                        
            if return_tokentime:
                
                with get_openai_callback() as cb:
                    time_start = monotonic()
                    chain = create_structured_output_runnable(QA_schema, llm, prompt)
                    output = await chain.ainvoke({"question": question, "choices": choices})
                    time_end = monotonic()
                    time_duration = time_end - time_start
                    tokentime=f"""
                    Total Tokens: {cb.total_tokens}
                    Prompt Tokens: {cb.prompt_tokens}
                    Completion Tokens: {cb.completion_tokens}
                    Total Cost (USD): ${cb.total_cost}
                    Q&A time duration (s): {time_duration}
                    """
                    
                    return output['answer'], output['rationale'], output['certainty'], str(output), tokentime, str(Experiment_detail)
            else:
                chain = create_structured_output_runnable(QA_schema, llm, prompt)
                output = await chain.ainvoke({"question": question, "choices": choices})
                return output.get('answer', ""), output.get('rationale', ""), output.get('certainty', ""), str(output), r"return_tokentime=False -> not stored", str(Experiment_detail) # this last empty string will be returned instead of tokentime
                            

    except Exception as e:
        # Handle errors by printing the error and returning all values as None
        print(f"An error occurred for '{model_name}' generation: {str(e)}")
        return None, None, None, None , None,None
    

async def handle_llm_response_langchain(excel_file_path: str, llm_list: list, openai_api, 
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,
                              return_tokentime :bool=False,
                                                            
                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512,
                              
                              just_raw=False,
                              
                              system_prompt= " ",
                              overall_prompt= " ",
                              answer_prompt= " ",
                              rationale_prompt= " ",
                              certainty_prompt= " "):
    
    list_of_vision_models=['gpt-4-0125-preview']
    list_of_text_models=['gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview','gpt-3.5-turbo-0125'
                         "mistral8x7b",
                         "llama2-70b"]
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    async def process_row(row, llm_name, idx):
        
        answer = certainty = rationale = raw = correctness = None
        
        if row['URL_generated'] == "GENERATED":
            #running image models on image questions
            if llm_name in list_of_vision_models:
                answer, rationale, certainty, raw = openai_wimage_answer_medical_question(question=row['Question'], image_string=row['Image urls'], choices=row['Options'], openai_api=openai_api)
                correctness = check_answer_correctness(truth=row['Correct Answer'], answer=answer)
                return idx, answer, rationale, certainty, raw, "tokentime code not completed for openai image", correctness
            
            # returning empty output for text models for image questions 
            if llm_name in list_of_text_models:
                return idx, "", "", "", "", "", ""
            else:
                return idx, 'ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname', 'ERRORinLLMname'
        
        
        else: #running text models on text-only questions
            if llm_name in ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview','gpt-3.5-turbo-0125'):
                answer, rationale, certainty, raw, tokentime, experimental = await Langchain_OpenAI_LCfunction(question=row['Question'], choices=row['Options'], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, return_tokentime=return_tokentime,
                                                                                                    model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, just_raw=just_raw,
                                                                                                    system_prompt=system_prompt,overall_prompt=overall_prompt,answer_prompt=answer_prompt, rationale_prompt=rationale_prompt, certainty_prompt=certainty_prompt)
                correctness = check_answer_correctness(truth=row['Correct Answer'], answer=answer)
                return idx, answer, rationale, certainty, raw, tokentime, experimental, correctness
            elif llm_name == "mistral8x7b":
                answer, rationale, certainty, raw = await mixtral_answer_medical_question(question=row['Question'], choices=row['Options'])
                correctness = check_answer_correctness(truth=row['Correct Answer'], answer=answer)
                return idx, answer, rationale, certainty, raw, "tokentime code not completed for mistral",correctness
            elif llm_name == "llama2-70b":
                answer, rationale, certainty, raw = await llama2_answer_medical_question(question=row['Question'], choices=row['Options'])
                correctness = check_answer_correctness(truth=row['Correct Answer'], answer=answer)
                return idx, answer, rationale, certainty, raw, "tokentime code not completed for llama2", correctness
            else:
                return idx, 'ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname','ERRORinLLMname', 'ERRORinLLMname'
 
    # Loop through llm list
    for llm in llm_list:
        # Create a column for each llm (to store response) if it doesn't exist
        if llm not in df.columns:
            df[llm] = ''
        
        tasks = []
        
        i=0
        max_index = df.index.max()
        for index, row in df.iterrows():
            if row[llm] != 'EXTRACTED':
                if row['URL_generated'] == "GENERATED":
                    task = asyncio.create_task(process_row(row, llm, idx=index))
                    tasks.append(task)
                    i+=1
                else:
                    task = asyncio.create_task(process_row(row, llm, idx=index))
                    tasks.append(task)
                    i+=1
                
                #saving output after finishing 15 tasks    
                if i==number_of_task_to_save or index == max_index: 
                    results = await asyncio.gather(*tasks)
                    
                    for result in results:
                        if result:  # Ensure result is not None or handle as needed
                            
                            idx, answer, rationale, certainty, raw, tokentime, experimental, correctness = result
                            # Update the DataFrame based on the result
                            df.at[idx, llm] = 'EXTRACTED'
                            df.at[idx, f'{llm}_rawoutput'] = raw
                            df.at[idx, f'{llm}_answer'] = answer
                            df.at[idx, f'{llm}_playground'] = rationale
                            df.at[idx, f'{llm}_certainty'] = certainty
                            df.at[idx, f'{llm}_tokentime'] = tokentime
                            df.at[idx, f'{llm}_experiment'] = experimental
                            df.at[idx, f'{llm}_correctness'] = correctness
                            
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    
                    if show_performance_during_loop:
                        try:
                            temptable=analyze_model_accuracy(excel_file_path)
                            print(temptable)
                        except Exception as e:
                            print(f"Error in showing performance during loop. Error:   {e}")
                            continue
                        
                    
                    #reset for continue
                    i=0
                    tasks = []
                    
                        
    df = save_and_open_excel(df, excel_file_path, open_at_end)
    
    if show_performance_at_end:
        try:
            performance_table= analyze_model_accuracy(excel_file_path)
            print(performance_table)
        except Exception as e:
            print(f"Error in showing the performance. Error:   {e}")
            
    #reset experiment detail
    Langchain_OpenAI_LCfunction.has_run_before = False
    return df



#----------- LLM Generation: Claude Structured Output -------------
# LLM generation
#https://www.datacamp.com/tutorial/getting-started-with-claude-3-and-the-claude-3-api
#https://docs.anthropic.com/claude/docs/control-output-format#prefilling-claudes-response
#https://github.com/anthropics/anthropic-cookbook/blob/main/function_calling/function_calling.ipynb
import shutil
import time
from anthropic import AsyncAnthropic
import json
import os
import asyncio
import re
import async_timeout

# creating the schema (it is something like openai schema but in XML)
def construct_format_tool_for_claude_prompt(name, description, parameters):
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )
    return constructed_prompt

tool_name = "answer_to_QA"
tool_description = """You can generate your unstructured output and selected option. Make sure to return, both, your unstructured output and selected answer."""

def construct_format_parameters_prompt(parameters):
    constructed_prompt = "\n".join(f"<parameter>\n<name>{parameter['name']}</name>\n<type>{parameter['type']}</type>\n<description>{parameter['description']}</description>\n</parameter>" for parameter in parameters)

    return constructed_prompt

parameters = [
    {
        "name": "unstructured output",
        "type": "str",
        "description": "Your unstructured output for the answer, if needed."
    },
    {
        "name": "selected option",
        "type": "str",
        "description": "The selected option from the list of provided choices. (Example: 'F')"
    },
]
tool = construct_format_tool_for_claude_prompt(tool_name, tool_description, parameters)

def construct_tool_use_system_prompt(tools):
    tool_use_system_prompt = (
        "In this environment you have access to a set of tools you can use to answer the user's question, following user's prompt.\n"
        "\n"
        "You may call them like this:\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<parameters>\n"
        "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
        "...\n"
        "</parameters>\n"
        "</invoke>\n"
        "</function_calls>\n"
        "\n"
        "Here are the tools available:\n"
        "<tools>\n"
        + '\n'.join([tool for tool in tools]) +
        "\n</tools>"
    )
    return tool_use_system_prompt

claude_functioncall_system_prompt = construct_tool_use_system_prompt([tool])

# function for extracting outputs from the LLM response
def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list



# Claude3 - TEXT ---------------------------------------------
async def Claude3_QA_FunctionCall(question: str, choices: str, overall_prompt:str, model_name, calude3_api=os.getenv('ANTHROPIC_API_KEY'), 
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):

    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    
    if not hasattr(Claude3_QA_FunctionCall, 'has_run_before') or not Claude3_QA_FunctionCall.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        Claude3_QA_FunctionCall.has_run_before = True


    try:
        prompt_messages=[
            {"role": "user", 
             "content": f"""
             {overall_prompt}
             
             Question:
             {question}
             Choices:
             {choices}
             """
             }]
        
        async with async_timeout.timeout(request_timeout):
            
            client = AsyncAnthropic(api_key=calude3_api)

            response = await client.messages.create(
                max_tokens=model_max_tokens,
                messages= prompt_messages,
                system=claude_functioncall_system_prompt,
                model=model_name,
                temperature=model_tempreature,
                stop_sequences=["\n\nHuman:", "\n\nAssistant", "</function_calls>"])
            
            try:
                response_text= response.content[0].text
                structure_answer=extract_between_tags("selected option", response_text)[0]
            except Exception as ee:
                structure_answer=f'ERROR in extracting selected answer: {ee}'
    except Exception as e:
        Experiment_detail=response=structure_answer=f'ERROR in response generation: {e}'

    return Experiment_detail, response, structure_answer
                                           
# Example use
#question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
#choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
#overall_prompt='Answer the following question and select the option from provided options.'
#model_name='claude-3-haiku-20240307'
#Experiment_detail, response, structure_answer = await Claude3_QA_FunctionCall(question, choices, overall_prompt, model_name, calude3_api=os.getenv('ANTHROPIC_API_KEY'), 
#                                            model_tempreature=0, model_max_tokens=512,
#                                            request_timeout=30)
#print(Experiment_detail, '\n', response,'\n', structure_answer)
    
    
from asyncio import CancelledError
import pandas as pd

async def handle_llm_response_claude3_functioncall(excel_file_path: str, llm_list: list, calude3_api,  overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, add_delay_sec:int=1,
                              request_timeout:int=15,

                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512, max_token_plus_input=False,
                               ):
    

    
    try:
        df = pd.read_excel(excel_file_path,)
        
        async def process_row(row, llm_name, idx, model_max_tokens, max_token_plus_input,):
            if max_token_plus_input:
                input_token_count= row['input_token_count']
                
                max_token=input_token_count+model_max_tokens
                max_token=int(max_token)
            else:
                max_token=int(model_max_tokens)
            Experiment_detail= response=structure_answer= correctness = None

            Experiment_detail, response, structure_answer = await Claude3_QA_FunctionCall(question=row['Question'], choices=row['Options'], model_name=llm_name, calude3_api=calude3_api,request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=max_token, 
                                                                                                overall_prompt=overall_prompt)
            correctness = check_answer_correctness(truth=row['Correct Answer'], answer=structure_answer)
            return idx, Experiment_detail, response, structure_answer, correctness

    
        # Loop through llm list
        for llm in llm_list:
            # Create a column for each llm (to store response) if it doesn't exist
            if llm not in df.columns:
                df[llm] = ''
            
            tasks = []
            
            i=0
            max_index = df.index.max()
            for index, row in df.iterrows():
                if row[llm] != 'EXTRACTED':
                    

                    task = asyncio.create_task(process_row(row, llm, idx=index,model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input))
                    tasks.append(task)
                    i+=1
                    
                    #saving output after finishing number_of_task_to_save tasks    
                    if i==number_of_task_to_save or index == max_index: 
                        results = await asyncio.gather(*tasks)
                        
                        for result in results:
                            if result:  # Ensure result is not None or handle as needed

                                idx, Experiment_detail, response, structure_answer, correctness = result
                                # Update the DataFrame based on the result
                                df.at[idx, llm] = 'EXTRACTED'
                                df.at[idx, f'{llm}_rawoutput'] = str(response) 
                                df.at[idx, f'{llm}_answer'] = str(structure_answer)
                                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                                df.at[idx, f'{llm}_correctness'] = correctness
                                
                        #save draft
                        try:
                            df.to_excel(excel_file_path)
                            print(f"Draft excel file saved at {excel_file_path}")
                        except Exception as e:
                            print(f"Error in saving temporary excel. Error:   {e}")
                            continue
                        
                        if show_performance_during_loop:
                            try:
                                temptable=analyze_model_accuracy(excel_file_path)
                                print(temptable)
                            except Exception as e:
                                print(f"Error in showing performance during loop. Error:   {e}")
                                continue
                            
                        
                        #reset for continue
                        i=0
                        tasks = []
                        print('sleep like a baby')
                        await asyncio.sleep(add_delay_sec)
            

    except Exception as e:
        print(f'Error in the handler: {e}')
    except KeyboardInterrupt or CancelledError:
        print("Operation interrupted. Cleaning up...")
        
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        Claude3_QA_FunctionCall.has_run_before = False
        return df


# Example of final run

# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_round_Julyy.xlsx"


# llm_list=[
#     'claude-3-5-sonnet-20240620'
# ]

# calude3_api=os.getenv('ANTHROPIC_API_KEY')

# #overall_prompt="Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. If the question has images, ignore it and answer based on the text only. Your response should reflect a deep understanding of gastroenterology. "

# open_at_end=True

# number_of_task_to_save=1
# add_delay_sec=1

# request_timeout=60

# show_performance_at_end=False
# show_performance_during_loop=True

# model_tempreature=1
# model_max_tokens=1024
# max_token_plus_input=True

# final_df = await handle_llm_response_claude3_functioncall(excel_file_path=excel_file_path, llm_list=llm_list, calude3_api=calude3_api, message_content_column='Message_content',
#                               open_at_end=open_at_end,
#                               number_of_task_to_save=number_of_task_to_save, add_delay_sec=add_delay_sec,
#                               request_timeout=request_timeout,

#                               show_performance_at_end=show_performance_at_end,
#                               show_performance_during_loop=show_performance_during_loop,
                              
#                               model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, max_token_plus_input=max_token_plus_input)

#----------- LLM Generation: Groq -------------
#%pip install groq
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(temperature=0.8,
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
            }
    ],
    model="gemma2-9b-it",
)

print(chat_completion.choices[0].message.content)

from openai import OpenAI
import os
import pandas as pd

# OPENAI - TEXT ---------------------------------------------
def Groq_LLManswer(question: str, choices: str, model_name,
                                         overall_prompt:str,
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):
    """
    Answer a medical question using the OPENAI language model asynchronously.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['model_name'] = model_name
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    
    if not hasattr(Groq_LLManswer, 'has_run_before') or not Groq_LLManswer.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")

    try:
        Groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

        response  =  Groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", 
                "content": f"""{overall_prompt}
                
                Question:
                {question}
                Choices:
                {choices}
                """
                }
                ],
            max_tokens=model_max_tokens,
            temperature=model_tempreature,
            timeout=request_timeout
            )
        
        raw_output=response.choices[0].message.content
    except Exception as e:
        print(f'Error in Groq_LLManswer Generation: {e}')
        Experiment_detail=response=f'Error in Groq_LLManswer Generation: {e}'
        raw_output=None
    finally:
        return Experiment_detail, response, raw_output


def handle_Groq(excel_file_path: str,overall_prompt:str, model_name:str, 
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,
                              model_tempreature=0.5, model_max_tokens=512,

                               ):
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    def process_row(row, model_name, idx):
        Experiment_detail, response, raw_output =  Groq_LLManswer(question=row['Question'], choices=row['Options'],model_name=model_name, request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, 
                                                                                                overall_prompt=overall_prompt)
        return idx, Experiment_detail, response, raw_output
         
    try:
        llm = model_name
        if llm not in df.columns:
            df[llm] = ''

        max_index = df.index.max()
        i=0
        for index, row in df.iterrows():
            if row[llm] != 'EXTRACTED':
                idx, Experiment_detail, response, raw_output=process_row(row, llm, idx=index)
                print('   index:',idx, 'output:',raw_output)
                # Update the DataFrame based on the result
                df.at[idx, f'{llm}_rawoutput'] = str(raw_output) 
                df.at[idx, f'{llm}_rawresponse'] = str(response) 
                df.at[idx, f'{llm}_answer'] = None
                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                df.at[idx, f'{llm}_correctness'] = None
                
                if raw_output:
                    df.at[idx, llm] = 'EXTRACTED'
                
                i+=1        
                #saving output after finishing number_of_task_to_save tasks    
                if i==number_of_task_to_save or index == max_index: 
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    i=0

    except CancelledError or KeyboardInterrupt:
        print('Interrupted the loop. Finishing ...')
    except Exception as e:
        print(f'Error in the loop: {e}')
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        Groq_LLManswer.has_run_before = False
        return df


# Example of usage on one inference
# overall_prompt='Answer this question and select one option.'
# model_tempreature=0.7
# model_max_tokens=512

#question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
#choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
# 
# request_timeout=150
# model_name= "gemma2-9b-it"
# Experiment_detail, response, raw_output= Groq_LLManswer(question=question, choices=choices,model_name=model_name,
#                                                                                             overall_prompt=overall_prompt,
#                                                                                             model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                                                                             request_timeout=request_timeout)
# print(Experiment_detail, response, raw_output, sep="\n\n")

# Example of final run an excel file
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_groq_gemma.xlsx"
# model_name= "gemma2-9b-it"
# overall_prompt="Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. Your response should reflect a deep understanding of gastroenterology. "
# open_at_end=True
# request_timeout=180
# model_tempreature=0.8
# model_max_tokens=1024
# number_of_task_to_save=1
# final_df = handle_Groq(excel_file_path=excel_file_path, overall_prompt=overall_prompt, model_name=model_name,
#                               open_at_end=open_at_end,
#                               number_of_task_to_save=number_of_task_to_save, 
#                               request_timeout=request_timeout,
#                               model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                )
      

#----------- LLM Generation: Poe -------------
# %pip install -U poe-api-wrapper
#https://github.com/snowby666/poe-api-wrapper
#%pip install -U poe-api-wrapper
#%pip install ballyregan
#llama_2_13b_chat  Google-PaLM	 Mistral-7B-T

# tokens = {
#     'p-b': "soco2nnPLRp8sjOUzBBLIA%3D%3D", 
#     'p-lat': "Z9fbwAHlWfsbG52eGViDOfbx8fWHq27gHUUhL4wCJQ%3D%3D",
# }

from zmq import proxy
from poe_api_wrapper import PoeApi
import os

tokens = { 
     'p-b': os.getenv("Poe_pb"), 
     'p-lat': os.getenv("Poe_lat"),
     'formkey': os.getenv("Poe_formkey"),
     '__cf_bm': '__cf_bm cookie here', 
     'cf_clearance': 'cf_clearance cookie here'
 }
bot = "Llama-2-13b"
message = "What is reverse engineering?"

# Default setup

client = PoeApi(tokens=tokens)
#print(client.get_available_creation_models())

for chunk in client.send_message(bot, message):
    pass
print(chunk["text"])

import os
from socket import timeout
import pandas as pd
from poe_api_wrapper import PoeApi
import time


# OPENAI - TEXT ---------------------------------------------
def PoeWrapper_LLManswer(question: str, choices: str, model_name,
                                         overall_prompt:str,
                                         request_timeout:int=30):
    
    tokens = { 
     'p-b': os.getenv("Poe_pb"), 
     'p-lat': os.getenv("Poe_lat"),
     'formkey': os.getenv("Poe_formkey"),
     '__cf_bm': '__cf_bm cookie here', 
     'cf_clearance': 'cf_clearance cookie here'
 }
    
    
    Experiment_detail={}
    Experiment_detail['model_name'] = model_name
    Experiment_detail['overall_prompt'] = overall_prompt

    
    if not hasattr(PoeWrapper_LLManswer, 'has_run_before') or not PoeWrapper_LLManswer.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")


                
    try:
        message = f"""{overall_prompt}
        # Question:
        {question}
        # Choices:
        {choices}
        """
        
        client = PoeApi(tokens=tokens)
        for response in client.send_message(model_name, message):
            pass
        raw_output=response["text"]
        
    except Exception as e:
        print(f'Error in PoeWrapper_LLManswer Generation: {e}')
        Experiment_detail=response=f'Error in PoeWrapper_LLManswer Generation: {e}'
        raw_output=None
    finally:
        return Experiment_detail, response, raw_output


def handle_PoeWrapper(excel_file_path: str,overall_prompt:str, model_name:str, 
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,
                              ):
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    def process_row(row, model_name, idx):
        Experiment_detail, response, raw_output =  PoeWrapper_LLManswer(question=row['Question'], choices=row['Options'],model_name=model_name, request_timeout=request_timeout, overall_prompt=overall_prompt)
        return idx, Experiment_detail, response, raw_output
         
    try:
        llm = model_name
        if llm not in df.columns:
            df[llm] = ''

        max_index = df.index.max()
        i=0
        for index, row in df.iterrows():
            if row[llm] != 'EXTRACTED':
                idx, Experiment_detail, response, raw_output=process_row(row, llm, idx=index)
                time.sleep(2)
                print('   index:',idx, 'output:',raw_output)
                # Update the DataFrame based on the result
                df.at[idx, f'{llm}_rawoutput'] = str(raw_output) 
                df.at[idx, f'{llm}_rawresponse'] = str(response) 
                df.at[idx, f'{llm}_answer'] = None
                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                df.at[idx, f'{llm}_correctness'] = None
                
                if raw_output:
                    df.at[idx, llm] = 'EXTRACTED'
                
                i+=1        
                #saving output after finishing number_of_task_to_save tasks    
                if i==number_of_task_to_save or index == max_index: 
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    i=0

    except CancelledError or KeyboardInterrupt:
        print('Interrupted the loop. Finishing ...')
    except Exception as e:
        print(f'Error in the loop: {e}')
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        PoeWrapper_LLManswer.has_run_before = False
        return df


# Example
# overall_prompt='Answer this question and select one option.'
# model_tempreature=0.7
# model_max_tokens=512

# question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
# choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
# #
# request_timeout=150
# model_name= "gemma2-9b-it"
# Experiment_detail, response, raw_output= PoeWrapper_LLManswer(question=question, choices=choices,model_name=model_name,
#                                                                                             overall_prompt=overall_prompt,
#                                                                                             model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                                                                             request_timeout=request_timeout)
# print(Experiment_detail, response, raw_output, sep="\n\n

# Example of Final run
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_round_Julyy.xlsx"

# model_names= ["Llama-2-13b","Mistral-7B-T",]
# for model_name in model_names:
#     overall_prompt="Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. Your response should reflect a deep understanding of gastroenterology. "
#     open_at_end=True
#     request_timeout=60
#     number_of_task_to_save=1
    


#     final_df = handle_PoeWrapper(excel_file_path=excel_file_path, overall_prompt=overall_prompt, model_name=model_name,
#                                 open_at_end=open_at_end,
#                                 number_of_task_to_save=number_of_task_to_save, 
#                                 request_timeout=request_timeout,
#                                 )

#----------- LLM Generation: Local LLM with LM studio -------------
# Before running this code, you should follow these steps for each run (llama.cpp is better as you can directly read the model GGUF and do your stuff, but LLM studio is easier to setup):
# (1) install LM studio, 
# (2) download your model and place it in a location that LM studio can recognize, 
# (3) load the model in the LM studio environment and check if it works (you should also select the correct preset) 
# (4) Create a server in LM studio
# (5) enter the required information for each run (number of GPU layers, CPU thread, ...)
# (6) off load the model, and load the next one and do these steps again (I know it is not a smart way of doing things, but it worked for me)

from openai import OpenAI
import os
import pandas as pd
from IPython.display import HTML, display #to display performance tables
import time

# OPENAI - TEXT ---------------------------------------------
def LMstudio_playing_openai(question: str, choices: str,
                                         overall_prompt:str,
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):
    """
    Answer a medical question using the OPENAI language model asynchronously.

    Args:
        question (str): The medical question.
        choices (str): A string containing answer choices.

    Returns:
        tuple: A tuple containing the best answer, certainty, and rationale.
               If an error occurs, all values will be None.
               
    Notes: 2024027version: The token counter was added and returned. The timeout was added. The function will print the model parameters and prompts on the first run.
    """
    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens

    
    if not hasattr(LMstudio_playing_openai, 'has_run_before') or not LMstudio_playing_openai.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        LMstudio_playing_openai.has_run_before = True
        Experiment_detail['GPU Offload'] = input('First run: What is the number of GPU Offload layers')
        Experiment_detail['CPU thread'] = input('First run: What is the number of CPU Threads')

    try:
        LMstudio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        response  =  LMstudio_client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": " "},
                {"role": "user", 
                "content": f"""{overall_prompt}
                
                Question:
                {question}
                Choices:
                {choices}
                """
                }
                ],
            max_tokens=model_max_tokens,
            temperature=model_tempreature,
            timeout=request_timeout
            )
        
        raw_output=response.choices[0].message.content
    except Exception as e:
        print(f'Error in LMstudio_playing_openai Generation: {e}')
        Experiment_detail=response=f'Error in LMstudio_playing_openai Generation: {e}'
        raw_output=None
    finally:
        return Experiment_detail, response, raw_output


def handle_LMstudio(excel_file_path: str, llm: str, overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, 
                              request_timeout:int=15,
                              model_tempreature=0.5, model_max_tokens=512,

                               ):
    input("Are you sure that the name of llm is the same as loaded llm in LM studio? Press any key to continue")
    # Read excel
    df = pd.read_excel(excel_file_path,)
    
    def process_row(row, llm_name, idx):
        Experiment_detail, response, raw_output =  LMstudio_playing_openai(question=row['Question'], choices=row['Options'],request_timeout=request_timeout, 
                                                                                                model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, 
                                                                                                overall_prompt=overall_prompt)
        return idx, Experiment_detail, response, raw_output
         
    try:
        if llm not in df.columns:
            df[llm] = ''

        max_index = df.index.max()
        i=0
        for index, row in df.iterrows():
            if row[llm] != 'EXTRACTED':
                idx, Experiment_detail, response, raw_output=process_row(row, llm, idx=index)
                print('   index:',idx, 'output:',raw_output)
                # Update the DataFrame based on the result
                df.at[idx, f'{llm}_rawoutput'] = str(raw_output) 
                df.at[idx, f'{llm}_rawresponse'] = str(response) 
                df.at[idx, f'{llm}_answer'] = None
                df.at[idx, f'{llm}_Experiment_detail'] = str(Experiment_detail)
                df.at[idx, f'{llm}_correctness'] = None
                
                if raw_output:
                    df.at[idx, llm] = 'EXTRACTED'
                
                
                i+=1
                time.sleep(3)        
                #saving output after finishing number_of_task_to_save tasks    
                if i==number_of_task_to_save or index == max_index: 
                    #save draft
                    try:
                        df.to_excel(excel_file_path)
                        print(f"Draft excel file saved at {excel_file_path}")
                    except Exception as e:
                        print(f"Error in saving temporary excel. Error:   {e}")
                        continue
                    i=0

    except CancelledError or KeyboardInterrupt:
        print('Interrupted the loop. Finishing ...')
    except Exception as e:
        print(f'Error in the loop: {e}')
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        LMstudio_playing_openai.has_run_before = False
        return df


# Example of one run
#overall_prompt='Answer this question and select one option.'
#model_tempreature=0.7
#model_max_tokens=512
#
# question="I have a high temperature and cough from 3 days ago, and I recently visited a patient with the same symptoms who lives in a under developed region. What is the best diagnosis among the following options? "
# choices="A Tuberculosis B herniation of disk C Lupus D Arthritis"
#
#request_timeout=150
#
#Experiment_detail, response, raw_output= LMstudio_playing_openai(question=question, choices=choices,
#                                                                                            overall_prompt=overall_prompt,
#                                                                                            model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
#                                                                                            request_timeout=request_timeout)

# Example of final run
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_round_Julyy.xlsx"
# #llm='llama2-13B-Q5KM'
# #llm='medicine-chat-Q8'
# #llm='llama2-7B-Q8'
# #llm='mistral-instruct-v2-Q8'
# #llm='llama3-8b-Q8'
# #llm='llama3-8b-Q8'
# #llm='phi3-3b-Q16'
# #llm='openbioLLM-7B-Q8'

# #llm='Phi3-medium14b-Q6'
# llm='Gemma2-9b-Q8'

# overall_prompt="Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. Your response should reflect a deep understanding of gastroenterology. "
# open_at_end=True
# request_timeout=180
# model_tempreature=0.8
# model_max_tokens=1024
# number_of_task_to_save=1

# final_df = handle_LMstudio(excel_file_path=excel_file_path, llm=llm, overall_prompt=overall_prompt,
#                               open_at_end=open_at_end,
#                               number_of_task_to_save=number_of_task_to_save, 
#                               request_timeout=request_timeout,
#                               model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,







# ------------- Clean Errors --------------------
# Note: As I got errors such as request limit, and coroption in parsing of the structured outputs, I prepared this code to clean the errors, and then run the generation again. 4
import pandas as pd

def clean_error_rows(excel_file_path, target_column):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)
    
    # Initialize a flag to track if any changes are made
    changes_made = False
    
    # Get the prefix from the target column (e.g., 'gpt-4-0613' from 'gpt-4-0613_answer')
    prefix = target_column.split('_')[0]
    
    # Find columns to be cleared based on the prefix
    columns_to_clear = [col for col in df.columns if col.startswith(prefix)]
    
    i=0
    # Iterate through the DataFrame to find rows with the specified error
    for index, row in df.iterrows():
        if str(row[target_column]).startswith('ERROR in response generation: Error code: 429') or str(row[target_column]).startswith( 'ERROR in response generation: Connection error.') or str(row[target_column]).startswith( 'ERROR in response generation: request timeout'):
            # Clear the content of all specified columns for this row
            for col in columns_to_clear:
                df.at[index, col] = None
            i+=1
            changes_made = True
    
    # If changes were made, clear the content of the last row in specified columns
    if changes_made:
        last_index = df.index[-1]
        for col in columns_to_clear:
            df.at[last_index, col] = None
    
    # Save the modified DataFrame back to the Excel file
    df.to_excel(excel_file_path, index=False)
    print("Excel file updated.")
    print(f'     {i} rows with Error 429 were removed and stored at {excel_file_path}' )
    
    return i

# Example usage
#excel_file_path = r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E0P1-MT1024-GiveModelTimetoThink.xlsx"
#target_column = r'gpt-4-0613_answer'
#clean_error_rows(excel_file_path, target_column)
#                                )


