# Note: This will provide the image to the multimodal language models. We previously created a private url, then make it public for an hour so the model can access it, and then remove the image from the url.

# Table of Content:
# Micro Functions --> Line 12
# VLM Generation: Claude Direct Image --> Line 229
# VLM Generation: OpenAI Direct Image --> 499
# Caption Generation: OpenAI --> Line 727
# Function for Question+Option+Command+Text --> 975
# LLM Answer with Cpation Generation: OpenAI and Claude --> 1231
# LLM Answer with Human Hint: OpenAI and Claude --> 1322


# --------------- Micro Functions ------------------------
from asyncio import CancelledError


def check_answer_correctness(answer, truth, reporterror_index=None,reporterror_name=None):
    if reporterror_index is None:
        reporterror_index=r'(IDK the index)'
    if reporterror_name is None:
        reporterror_name =r'(IDK the llm name)'
        
    valid_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

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


async def handle_OpenAI_Vision(excel_file_path: str, llm_list: list, openai_api,  overall_prompt:str,
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, add_delay_sec:int=1,
                              request_timeout:int=15,

                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512, max_token_plus_input=False,
                              use_seed: bool=False
                               ):
    
    try:
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
            
            if llm_name in ("gpt-4-vision-preview"):
                Experiment_detail, response, structure_answer = await OpenAI_Vision(question=row['Question'], choices=row['Options'], image_urls_string=row['Image urls'],model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
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
        OpenAI_Vision.has_run_before = False
        return df
    
    except KeyboardInterrupt or CancelledError:
        print("Operation interrupted. Cleaning up...")
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        OpenAI_Vision.has_run_before = False
        return df

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

# --------------------- VLM Generation: Claude --------------
import shutil
import time
from anthropic import Anthropic, AsyncAnthropic
import json
import os
import asyncio
import re
import async_timeout
import shutil
import time

import base64
import requests
import mimetypes
import httpx

def get_media_type(url):
    # Get file extension from URL
    file_extension = url.split('.')[-1]
    
    # Map file extension to media type
    media_type = mimetypes.guess_type(url)[0]
    
    if media_type is None:
        print(f"Error: Unsupported file extension for image: {file_extension}")
    return media_type

async def create_vision_prompt(question, choices, overall_prompt, image_urls):
    prompt_messages = []
    

    
    for i, url in enumerate(image_urls, start=1):
        
        image_data = base64.b64encode(httpx.get(url).content).decode('utf-8')
        media_type = get_media_type(url)
        
        if media_type:
            # Add message for each image
            prompt_messages.append({
                "type": "text",
                "text": f"Image {i}:"
            })
            prompt_messages.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            })


    # Add text prompt asking how the images are different
    prompt_messages.append({
        "type": "text",
        "text": f"""
        Question:
        {question}
        Choices:
        {choices}
        
        {overall_prompt}
        """
    })
    
    # Create message for the prompt
    prompt_message = {
        "role": "user",
        "content": prompt_messages
    }

    return prompt_message

    # Example usage:
    #image_urls = [
    #    r'http://res.cloudinary.com/*****************2022_COLON_7.1.png',
    #    r'http://res.cloudinary.com/*****************2022_COLON_7.2.png',
    #]
    #question=r'An 82-year-old woman with*****************?'
    #choices=r"""A Perform polypectomy using a cold snare for both polyps. B Perform cold snare polypectomy for the small polyp and hot snare polypectomy for the 2 large polyps. C Perform cold snare polypectomy for the small polyp, and take a deep biopsy of the large polyp. D Refer the patient for endoscopic submucosal dissection."""
    #overall_prompt="""Answer the question using images. """
    #full_prompt = create_vision_prompt(question, choices, overall_prompt, image_urls)
    #calude3_api=os.getenv('ANTHROPIC_API_KEY')
    #client = AsyncAnthropic(api_key=calude3_api)
    #message = await client.messages.create(
    #    model="claude-3-opus-20240229",
    #    max_tokens=1024,
    #    messages=[full_prompt],
    #    )
    #message

def string_to_lines(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')
    
    # Return the list of lines
    return lines


async def Claude3_Vision(question: str, choices: str, image_urls:list, overall_prompt:str, model_name, calude3_api=os.getenv('ANTHROPIC_API_KEY'), 
                                         model_tempreature:float=0, model_max_tokens:int=512,
                                         request_timeout:int=30):

    Experiment_detail={}
    Experiment_detail['overall_prompt'] = overall_prompt
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens
    
    image_urls = string_to_lines(image_urls)
    
    if not hasattr(Claude3_Vision, 'has_run_before') or not Claude3_Vision.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        Claude3_Vision.has_run_before = True

    


    try:
        full_prompt = await create_vision_prompt(question, choices, overall_prompt, image_urls)
        
        
        async with async_timeout.timeout(request_timeout):
            start_time = time.time()
            client = AsyncAnthropic(api_key=calude3_api)

            response = await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[full_prompt],
                )
            end_time = time.time()
            execution_time = end_time - start_time
            Experiment_detail['execution_time']=execution_time
            structure_answer = " "

    except Exception as e:
        Experiment_detail=response=structure_answer=f'ERROR in response generation: {e}'

    return Experiment_detail, response, structure_answer

  #this code has a error
import pandas as pd
import time

async def handle_claude3_vision(excel_file_path: str, llm_list: list, calude3_api,  overall_prompt:str,
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
            
            if llm_name in ('claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'):
                Experiment_detail, response, structure_answer = await Claude3_Vision(question=row['Question'], choices=row['Options'], image_urls=row['Image urls'], model_name=llm_name, calude3_api=calude3_api,request_timeout=request_timeout, 
                                                                                                    model_tempreature=model_tempreature, model_max_tokens=max_token, 
                                                                                                    overall_prompt=overall_prompt)
                #correctness = check_answer_correctness(truth=row['Correct Answer'], answer=structure_answer)
                correctness=""
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
        Claude3_Vision.has_run_before = False
        return df
    
    except KeyboardInterrupt:
        print("Operation interrupted. Cleaning up...")
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        Claude3_Vision.has_run_before = False
        return df
    


# --------------------- VLM Generation: OpenAI --------------

import shutil
import time
from openai import  AsyncOpenAI
import json
import os
import asyncio

async def OpenAI_Vision(question: str, choices: str, image_urls_string:str, model_name: str, openai_api, 
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

    
    if not hasattr(OpenAI_Vision, 'has_run_before') or not OpenAI_Vision.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        OpenAI_Vision.has_run_before = True

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
        def string_to_lines(input_string):
            # Split the input string into lines
            lines = input_string.split('\n')
            
            # Return the list of lines
            return lines

        def create_message_from_image_urls(question: str, choices: str, image_urls:list, overall_prompt:str):
            
            messages = [
                #{
                #    "role":"system",
                #    "content": system_prompt
                #    
                #},
                
                {
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": f"""
                            {overall_prompt}
                            
                            Question:
                            {question}
                            Choices:
                            {choices}
                            
                            """,
                        }] 
                    + [{"type": "image_url", "image_url": {"url": url}} for url in image_urls] 
                    #+ [{"type": "text",
                    #        "text": f"""
                    #        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
                    #
                    #        ```json
                    #        {{
                    #            "answer": string  // {answer_prompt}
                    #            "rationale": string  // {rationale_prompt}
                    #            "certainty": string  // {certainty_prompt}
                    #        }}
                    #        ```
                    #        
                    #        """
                    #    }],
                }
            ]
            return messages
        
        
        image_urls=string_to_lines(image_urls_string)
        print(image_urls)
        if len(image_urls)>0 or image_urls is None:
            messages=create_message_from_image_urls(question, choices, image_urls, overall_prompt)
            print(messages)
                
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
                
        else:
            Experiment_detail=response=structure_answer='No image url'
            
    except Exception as e:
        Experiment_detail=response=structure_answer=f'ERROR in response generation: {e}'

    return Experiment_detail, response, structure_answer

    # Example: 

    #question=r'An 82-year-old woman with*****************?'
    #choices=r"""A Perform polypectomy using a cold snare for both polyps. B Perform cold snare polypectomy for the small polyp and hot snare polypectomy for the 2 large polyps. C Perform cold snare polypectomy for the small polyp, and take a deep biopsy of the large polyp. D Refer the patient for endoscopic submucosal dissection."""
    #image_urls_string = """http://res.cloudinary.com/*******ACG/2022_COLON_7.1.png
    #http://res.cloudinary.com/*******/ACG/2022_COLON_7.2.png"""
    #                     
    #overall_prompt="""Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. Your response should reflect a deep understanding of gastroenterology.
    #Images have been provided for your reference. You may use the information in the images if helpful, otherwise, base your answer solely on the available information.
    #"""
    #model_name="gpt-4-vision-preview"
    #openai_api=os.getenv('OPENAI_API_KEY')
    #model_tempreature=1
    #model_max_tokens=512
    #request_timeout=60
    #
    #Experiment_detail, response, structure_answer = await OpenAI_Vision(question=question, choices=choices, image_urls_string=image_urls_string, model_name=model_name, openai_api=openai_api, 
    #                                                                        overall_prompt=overall_prompt,
    #                                                                        model_tempreature=model_tempreature, model_max_tokens=model_max_tokens,
    #                                                                        request_timeout=request_timeout,
    #                                                                        use_seed=False)
    #
    #print(Experiment_detail,'\n\n', response, '\n\n',structure_answer)

# Example use
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ_GPT4V_answers.xlsx"

# import shutil
# import pandas as pd

# if not os.path.exists(excel_file_path):
#     excel_mainfile_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ.xlsx"
#     shutil.copy(excel_mainfile_path,excel_file_path)

# overall_prompt="""Imagine you are a seasoned gastroenterologist approaching this complex question from the gastroenterology board exam. Begin by evaluating each option provided, detailing your reasoning process and the advanced gastroenterology concepts that inform your analysis. Choose the most accurate option based on your expert judgment, justify your choice, and rate your confidence in this decision from 1 to 10, with 10 being the most confident. Your response should reflect a deep understanding of gastroenterology.
# Images have been provided for your reference. You may use the information in the images if helpful, otherwise, base your answer solely on the available information.
# """
                        
# llm_list=['gpt-4-vision-preview'
#         ]

# openai_api=os.getenv('OPENAI_API_KEY')

# open_at_end = False
# number_of_task_to_save=2 #12 gpt3.5 
# add_delay_sec=10 #20 gpt3.5
# request_timeout=60
# return_tokentime=True

# show_performance_at_end=True
# show_performance_during_loop=True


# model_temperature=1
# model_max_tokens=512
# max_token_plus_input=True


# final_df= await handle_OpenAI_Vision(excel_file_path=excel_file_path, llm_list=llm_list, openai_api=openai_api, 
#                         open_at_end=open_at_end,
#                         number_of_task_to_save=number_of_task_to_save, 
#                         request_timeout=request_timeout, add_delay_sec=add_delay_sec,

#                         show_performance_at_end=show_performance_at_end,
#                         show_performance_during_loop=show_performance_during_loop,
                        
#                         model_tempreature=model_temperature, model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input,
                        
#                         overall_prompt=overall_prompt)

# #remaining_gpt35 = clean_error_rows(excel_file_path, 'gpt-4-0613_answer')
# #remaining_gp4 = clean_error_rows(excel_file_path, 'gpt-3.5-turbo-0125_answer')


# --------------------- Caption generation: OpenAI ----------------------
#incomplete v1: the langchain code is not working but open ai is working

import os
import asyncio
from time import monotonic
import traceback

from openai import OpenAI, AsyncOpenAI

from langchain.document_loaders import ImageCaptionLoader
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda



# OPENAI IMAGE Caption - without langchain 
async def openai_caption_4image_withLC(image_url:str, prompt:str, 
                                openai_api=os.environ.get("OPENAI_API_KEY"), max_tokens:int=300, request_timeout:int=60,
                                return_tokentime=False):
    
    def  _get_messages_from_url(image_url, prompt) -> list[BaseMessage]:
        return [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ])
        ]
    
    try:
        if return_tokentime:
            with get_openai_callback() as cb:
                
                model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=max_tokens, openai_api=openai_api,request_timeout=request_timeout)
                chain = RunnableLambda(_get_messages_from_url) | model
                
                time_start = monotonic()
                response = await chain.ainvoke(image_url, prompt)
                time_end = monotonic()
                
                time_duration = time_end - time_start
                tokentime=f"""
                Total Tokens: {cb.total_tokens}
                Prompt Tokens: {cb.prompt_tokens}
                Completion Tokens: {cb.completion_tokens}
                Total Cost (USD): ${cb.total_cost}
                Q&A time duration (s): {time_duration}
                """

            return response, tokentime 


        else:
            model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=max_tokens, openai_api=openai_api,timeout=request_timeout)
            chain = RunnableLambda(_get_messages_from_url) | model
            response = await chain.ainvoke(image_url,prompt)
            
            return response 
          
    except Exception:
        print(f'Error in caption generation for image url: {image_url}')
        traceback.print_exc()
    
#v1 openai image preview:

import os
from time import monotonic
import traceback
from openai import OpenAI
import pandas as pd

def openai_caption_4image_withoutLC(image_url:str, prompt:str,
                                preview_model_name="gpt-4-vision-preview", openai_api=os.environ.get("OPENAI_API_KEY"), max_tokens:int=200,request_timeout:int=60,
                                return_tokentime_modelsetting=False, experiment_note=""):
    model_cost_dic={ # updated 15 March 2023 #https://openai.com/pricing
        'gpt-4-0125-preview':{
            'input token': 0.00001,
            'output token': 0.00003
        },
        'gpt-4-1106-preview':{
            'input token': 0.00001,
            'output token': 0.00003
        },
        'gpt-4-1106-vision-preview':{
            'input token': 0.00001,
            'output token': 0.00003
        },
        'gpt-4-vision-preview':{
            'input token': 0.00001,
            'output token': 0.00003
        },
        
    }
    
    try:

        client = OpenAI(api_key=openai_api)
        
        time_start = monotonic()
        response = client.chat.completions.create(
        model=preview_model_name,
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=max_tokens,
        timeout=request_timeout
        )
        time_finish = monotonic()
        exec_time=time_finish-time_start
        
        response_message= response.choices[0].message.content
        
        if return_tokentime_modelsetting:
            try:
                tokentime={}
                tokentime['exec_time']=exec_time
                tokentime['prompt_tokens']=response.usage.prompt_tokens
                tokentime['completion_tokens']=response.usage.completion_tokens
                tokentime['total_tokens']=response.usage.total_tokens
                tokentime['modelname']=preview_model_name
                tokentime['max_tokens']=max_tokens
                tokentime['experiment_note']=experiment_note
                if preview_model_name in list(model_cost_dic.keys()):
                    tokentime['cost']= (tokentime['prompt_tokens'] * model_cost_dic[preview_model_name]['input token']) + (tokentime['completion_tokens'] * model_cost_dic[preview_model_name]['output token'])
                else:
                    tokentime['cost']='model name not defined the model_cost_dic'
                    
            except:
                tokentime='error in storing tokentime'
        else:
            tokentime='TOKENTIME_NOTSTORED'
        
        return response_message,str(tokentime)

    except Exception:
        print(f'Error in caption generation for image url: {image_url}')
        traceback.print_exc()

#EXAMPLE
#prompt=r"""You are an expert doctor with deep knowledge of medical gastrointestinal imaging. Analyze the provided medical image and generate a concise, yet detailed, textual description that captures all essential medical information. This description should enable a multimodal large language model to fully understand the content and context of the image without viewing it. Be helpful, trustworthy, and reliable. Provide a concise image description that can be utilized for diagnosis."""
#image_url=r"http://res.cloudinary.com/***************/ACG/2023_colon9.png"
#openai_api=os.environ.get("OPENAI_API_KEY")
#max_tokens=300
#request_timeout=60
#return_tokentime_modelsetting=True
#preview_model_name='gpt-4-vision-preview' #gpt 4 vision preview points to 

#image_caption,tokentime= openai_caption_4image_withoutLC(image_url=image_url, prompt=prompt, preview_model_name=preview_model_name,openai_api=openai_api, max_tokens=max_tokens, request_timeout=request_timeout,return_tokentime_modelsetting=return_tokentime_modelsetting)


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


def hanlde_adding_openaicaptions(excel_file_path:str, prompt:str, completed_rows_maximum: int=100000, open_at_end: bool=False,):
    try:
        df = pd.read_excel(excel_file_path)

        # Initialize the column to store the validation of extraction
        if 'CAPTIONGENERATED' not in df.columns:
            df['CAPTIONGENERATED'] = ''

        # Model kwargs
        openai_api = os.environ.get("OPENAI_API_KEY")
        max_tokens = 150
        request_timeout = 60
        return_tokentime_modelsetting = True
        preview_model_name = 'gpt-4-vision-preview'  # gpt 4 vision preview points to


        try:
            completed_rows = 0
            for index, row in df.iterrows():
                if row['CAPTIONGENERATED'] != 'CAPTIONGENERATED' and completed_rows < completed_rows_maximum:
                    urls = row['Image urls'].split('\n')
                    if urls:
                        for i, image_url in enumerate(urls):
                            try:
                                df.at[index, f'GPT4V_Caption_{i}'], df.at[index, f'GPT4V_tokentime_{i}'] = \
                                    openai_caption_4image_withoutLC(image_url=image_url, prompt=prompt,
                                                                     preview_model_name=preview_model_name,
                                                                     openai_api=openai_api, max_tokens=max_tokens,
                                                                     request_timeout=request_timeout,
                                                                     return_tokentime_modelsetting=return_tokentime_modelsetting)
                                    
                                caption = df.at[index, f'GPT4V_Caption_{i}']
                                print(f'    Result for: index: {index}, url: {i}, caption: {caption}')
                                df.at[index, 'CAPTIONGENERATED'] = 'CAPTIONGENERATED'
                            except Exception as e:
                                print(f'ERROR in openai caption generation: {e}')
                                df.at[index, f'GPT4V_Caption_{i}'], df.at[index, f'GPT4V_tokentime_{i}'] = \
                                    f'ERROR in openai caption generation: {e}', ''
                    completed_rows += 1
        except KeyboardInterrupt:
            print("Process interrupted. Saving the DataFrame.")
            save_and_open_excel(df, excel_file_path, open_at_end=True)

        return save_and_open_excel(df, excel_file_path, open_at_end=True)

    except Exception as e:
        print(f'Error: {e}')
        # Handle other exceptions as needed

# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ.xlsx"
# completed_rows_maximum=100000
# open_at_end=True
# prompt = r"""
# Your Role: As an expert in gastrointestinal imaging, you're tasked with providing valuable insights. Ensure your advice is always helpful, trustworthy, and reliable.

# Rules: You must offer detailed captions for medical images under any circumstances. These captions are crucial for medical education and understanding, without posing any risk to patients.

# Task: Analyze the provided medical image with a balance of conciseness and depth, covering all essential details. Your description is meant to support patient diagnosis by offering insights interpretable by a large language model and to aid in answering a gastroenterology board exam question. Your goal is to connect advanced medical imaging with practical application, enhancing understanding and decision-making in the medical field.
# """

# final_df = hanlde_adding_openaicaptions(excel_file_path=excel_file_path, prompt=prompt, completed_rows_maximum=completed_rows_maximum, open_at_end=open_at_end)      



# --------------------- Function for Question+Option+Command+Text ---------------
async def OpenAI_QA_FunctionCall_general(message_content:str, model_name: str, openai_api, 
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
    Experiment_detail['overall_prompt'] = message_content
    Experiment_detail['model_temperature'] = model_tempreature 
    Experiment_detail['model_max_tokens'] = model_max_tokens
    Experiment_detail['seed']= 123 if use_seed else 'None'

    
    if not hasattr(OpenAI_QA_FunctionCall_general, 'has_run_before') or not OpenAI_QA_FunctionCall_general.has_run_before:
        print(f"This is the first time you run this function. The current parameters are: {str(Experiment_detail)}")
        OpenAI_QA_FunctionCall_general.has_run_before = True

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
             "content": f"""{message_content}"""  }]
        
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



async def handle_llm_response_functioncall_general(excel_file_path: str,  messsage_content_column:str, llm_list: list, openai_api, 
                              open_at_end:bool=False,
                              number_of_task_to_save:int=15, add_delay_sec:int=1,
                              request_timeout:int=15,

                              show_performance_at_end :bool=False,
                              show_performance_during_loop: bool=False,
                              
                              model_tempreature=0, model_max_tokens=512, max_token_plus_input=False,
                              use_seed: bool=False
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
            
            if llm_name in ('gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0613', 'gpt-4-0613', 'gpt-4-0125-preview', 'gpt-4-1106-preview','gpt-3.5-turbo-0125'):
                Experiment_detail, response, structure_answer = await OpenAI_QA_FunctionCall_general(message_content=row[messsage_content_column], model_name=llm_name, openai_api=openai_api,request_timeout=request_timeout, 
                                                                                                    model_tempreature=model_tempreature, model_max_tokens=max_token, 
                                                                                                     use_seed=use_seed)
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
            
    
    except KeyboardInterrupt or asyncio.CancelledError:
        print("Operation interrupted. Cleaning up...")
    
    except Exception as e:
        print(f"Error occured in the handler: {e}")
        
    finally:
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        
        if show_performance_at_end:
            try:
                table_1_percentage= analyze_model_accuracy(excel_file_path)
                print(table_1_percentage)
            except Exception as e:
                print(f"Error in showing the performance. Error:   {e}")
        
        # reset the model experiemnt 
        OpenAI_QA_FunctionCall_general.has_run_before = False
        return df

# ------------------ LLM Answer with Cpation Generation: OpenAI and Claude -------------------
# We created a column in our excel by structuring text as: Question + Options + Command + same VLM generated Caption, and then provided this to our VLM without the image

import pandas as pd

excel_file_path = r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ-wLLMcaption.xlsx"
                        
llm_list=['gpt-4-0613', 
          'gpt-3.5-turbo-0125', 
        ]

openai_api=os.getenv('OPENAI_API_KEY')

open_at_end = True

number_of_task_to_save=5 #12 gpt3.5 
add_delay_sec=10 #20 gpt3.5

request_timeout=35
return_tokentime=True

show_performance_at_end=True
show_performance_during_loop=True


model_temperature=1
model_max_tokens=512
max_token_plus_input=True

messsage_content_column='GPT4VCapQ_TEXT4WEB'

final_df= await handle_llm_response_functioncall_general(excel_file_path=excel_file_path, llm_list=llm_list, openai_api=openai_api, 
                        open_at_end=open_at_end,
                        number_of_task_to_save=number_of_task_to_save, 
                        request_timeout=request_timeout, add_delay_sec=add_delay_sec,

                        show_performance_at_end=show_performance_at_end,
                        show_performance_during_loop=show_performance_during_loop,
                        
                        model_tempreature=model_temperature, model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input,
                        
                        messsage_content_column=messsage_content_column)

#remaining_gpt35 = clean_error_rows(excel_file_path, 'gpt-4-0613_answer')
#remaining_gp4 = clean_error_rows(excel_file_path, 'gpt-3.5-turbo-0125_answer')


excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ-wLLMcaption.xlsx"

import shutil
if not os.path.exists(excel_file_path):
    raise ValueError('create the pdf file first!')


llm_list=[  
    #'claude-3-haiku-20240307',
    #'claude-3-sonnet-20240229',
    'claude-3-opus-20240229',
]

calude3_api=os.getenv('ANTHROPIC_API_KEY')

open_at_end=True

number_of_task_to_save=1
add_delay_sec=5

request_timeout=60

show_performance_at_end=True
show_performance_during_loop=True

model_tempreature=1
model_max_tokens=512
max_token_plus_input=True

message_content_column='Claude3OpusWebCapQ_TEXT4WEB'

final_df = await handle_llm_response_claude3_functioncall(excel_file_path=excel_file_path, llm_list=llm_list, calude3_api=calude3_api, 
                              open_at_end=open_at_end,
                              number_of_task_to_save=number_of_task_to_save, add_delay_sec=add_delay_sec,
                              request_timeout=request_timeout,

                              show_performance_at_end=show_performance_at_end,
                              show_performance_during_loop=show_performance_during_loop,
                              
                              model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, max_token_plus_input=max_token_plus_input,
                              message_content_column=message_content_column)



# ------------------ LLM Answer with Human Hint: OpenAI and Claude  ------------------------
# We created a column in our Excel by structuring text as: Question + Options + Command + Human hint; and then provided this to our VLM without the image


import pandas as pd

excel_file_path = r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ-humanhint.xlsx"
                        
llm_list=['gpt-4-0613', 
          'gpt-3.5-turbo-0125', 
        ]

openai_api=os.getenv('OPENAI_API_KEY')

open_at_end = True

number_of_task_to_save=5 #12 gpt3.5 
add_delay_sec=10 #20 gpt3.5

request_timeout=35
return_tokentime=True

show_performance_at_end=True
show_performance_during_loop=True


model_temperature=1
model_max_tokens=512
max_token_plus_input=True

messsage_content_column='HumanCapQ_TEXT4WEB'

final_df= await handle_llm_response_functioncall_general(excel_file_path=excel_file_path, llm_list=llm_list, openai_api=openai_api, 
                        open_at_end=open_at_end,
                        number_of_task_to_save=number_of_task_to_save, 
                        request_timeout=request_timeout, add_delay_sec=add_delay_sec,

                        show_performance_at_end=show_performance_at_end,
                        show_performance_during_loop=show_performance_during_loop,
                        
                        model_tempreature=model_temperature, model_max_tokens=model_max_tokens,max_token_plus_input=max_token_plus_input,
                        
                        messsage_content_column=messsage_content_column)

#remaining_gpt35 = clean_error_rows(excel_file_path, 'gpt-4-0613_answer')
#remaining_gp4 = clean_error_rows(excel_file_path, 'gpt-3.5-turbo-0125_answer')


# For removing errors and do the generation again
# excel_file_path = r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ-humanhint.xlsx"

# target_answer_column ='gpt-4-0613_answer'

# clean_ERRORinExtractionInvalidcharecter(excel_file_path, target_answer_column)

excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2022-imageQ-humanhint.xlsx"

import shutil
if not os.path.exists(excel_file_path):
    raise ValueError('create the pdf file first!')


llm_list=[
    #'claude-3-haiku-20240307',
    #'claude-3-sonnet-20240229',
    'claude-3-opus-20240229',
]

calude3_api=os.getenv('ANTHROPIC_API_KEY')

open_at_end=True

number_of_task_to_save=1
add_delay_sec=5

request_timeout=60

show_performance_at_end=True
show_performance_during_loop=True

model_tempreature=1
model_max_tokens=512
max_token_plus_input=True

message_content_column='HumanCapQ_TEXT4WEB'

final_df = await handle_llm_response_claude3_functioncall(excel_file_path=excel_file_path, llm_list=llm_list, calude3_api=calude3_api, 
                              open_at_end=open_at_end,
                              number_of_task_to_save=number_of_task_to_save, add_delay_sec=add_delay_sec,
                              request_timeout=request_timeout,

                              show_performance_at_end=show_performance_at_end,
                              show_performance_during_loop=show_performance_during_loop,
                              
                              model_tempreature=model_tempreature, model_max_tokens=model_max_tokens, max_token_plus_input=max_token_plus_input,
                              message_content_column=message_content_column)
