# Note: Structured outputs will fail to provide the output in the correct structure.
# Note: The code in Line 4 aims to find the option from the corrupted structured output.
# Note: For unstructured output, I provided the options and the generated unstructured answer, and asked GPT4 to find the selected option in the given text (accuracy = 99%).

# --------------------- Function Find answer when missed ---------------------
def find_option_from_Chatcompletion(long_string: str) -> str:
    # Regular expression pattern to find the desired string
    pattern = r'"selected\s*option"\s*:\s*"([^"]+)"'

    # Search for the pattern in the long string
    match = re.search(pattern, long_string)

    if match:
        selected_option = match.group(1)
        return str(selected_option)
    else:
        return "Pattern not found."
    
    
def clean_ERRORinExtractionInvalidcharecter(excel_file_path,target_answer_column):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Get the prefix from the target column (e.g., 'gpt-4-0613' from 'gpt-4-0613_answer')
    prefix = target_answer_column.split('_')[0]

    i_corrected=0
    i_error=0
    print('I will manually check the raw output to get the selected option')
    # Iterate through the DataFrame to find rows with the specified error
    for index, row in df.iterrows():
            
        if str(row[target_answer_column]).startswith('ERROR in extracting selected answer: Invalid control character at:'):
            i_error+=1
            
            rawcolumn=prefix+'_rawoutput'
            correctnesscolumn=prefix+'_correctness'
            
            foundoption = find_option_from_Chatcompletion(row[rawcolumn]) 
            
            if foundoption in ['A','B', 'C', 'D', 'E']:
                modified_answer = row[target_answer_column].replace('ERROR in extracting selected answer', f'resolvedERROR (found option : {foundoption}) in extracting selected answer')
                modified_correctness=check_answer_correctness(foundoption, row['Correct Answer'])
                df.at[index, correctnesscolumn]=modified_correctness
                df.at[index, target_answer_column]=modified_answer
                i_corrected+=1
                
    print(f'    I fixed {i_corrected} rows from total of {i_error} rows with << ERROR in extracting ... Invalid control character >>')
    df.to_excel(excel_file_path, index=False)
    print(f"Excel file updated and saved at {excel_file_path}.")


#Example
#excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E2-2021-2-3-textQ - Copy.xlsx"
#target_answer_column ='gpt-4-0613_answer'

import re
import pandas as pd

def find_option_from_Chatcompletion(long_string: str) -> str:
    """
    Finds the selected option from a long string using regular expression.
    
    Args:
    - long_string (str): The long string to search within.
    
    Returns:
    - str: The selected option found.
    """
    pattern = r'"selected\s*option"\s*:\s*"([^"]+)"'
    match = re.search(pattern, long_string)
    if match:
        selected_option = match.group(1)
        return selected_option
    else:
        return "Pattern not found."

def clean_ERRORinExtractionInvalidcharecter(excel_file_path, target_answer_column):
    """
    Cleans the specified target answer column in the Excel file by replacing certain error strings.
    
    Args:
    - excel_file_path (str): The path to the Excel file.
    - target_answer_column (str): The name of the target answer column in the Excel file.
    """
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file_path)

        # Get the prefix from the target column (e.g., 'gpt-4-0613' from 'gpt-4-0613_answer')
        prefix = target_answer_column.split('_')[0]

        i_corrected = 0
        i_error = 0
        
        print('    START: manually check the raw output to get the selected option')
        
        # Iterate through the DataFrame to find rows with the specified error
        for index, row in df.iterrows():
            if str(row[target_answer_column]).startswith('ERROR in extracting selected answer: Invalid control character at:'):
                i_error += 1
                rawcolumn = prefix + '_rawoutput'
                correctnesscolumn = prefix + '_correctness'
                
                foundoption = find_option_from_Chatcompletion(row[rawcolumn])
                
                if foundoption in ['A', 'B', 'C', 'D', 'E']:
                    modified_answer = row[target_answer_column].replace('ERROR', f'resolvedERROR (found option : {foundoption})')
                    modified_correctness = check_answer_correctness(foundoption, row['Correct Answer'])  # You need to define this function
                    df.at[index, correctnesscolumn] = modified_correctness
                    df.at[index, target_answer_column] = modified_answer
                    i_corrected += 1
                    
        print(f'     I fixed {i_corrected} rows from a total of {i_error} rows with "ERROR in extracting ... Invalid control character"')
        
    except InterruptedError or CancelledError:
        print('Interrupted. Cleaning')
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save the modified DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)
        print(f"    Excel file updated and saved at {excel_file_path}.")
        


# Example usage:
#excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_all.xlsx"
#target_answer_column ='gpt-4-0613_answer'
#
#clean_ERRORinExtractionInvalidcharecter(excel_file_path, target_answer_column)


from math import nan
from random import choice
import pandas as pd
import asyncio
import os
import time
from openai import AsyncOpenAI
import json
import re


async def OpenAI_Extract_SelectedOption_fromtext(index, llmresponsetext:str,choice:str,true_answer:str,):
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
    openai_api=os.getenv('OPENAI_API_KEY')
    model_max_tokens=512
    model_name='gpt-4o-mini' # this will use the most affordable and new GPT3.5turno, curruntly points to gpt-3.5-turbo-0125
    model_temperature=0
    request_timeout=30
    
    try:
    
        client = AsyncOpenAI(api_key=openai_api)
        
        correct_answer_schema = [
        {
            "type": "function",
            "function": {
                "name": "extract_true_choice",
                "description": f"""Act as a helpful, reliable, and accurate assistant. This response is generated in the context of a multiple-choice question.  The LLM was asked to think about each option and then select an option. Your task is to identify the exact chosen option based on the provided text.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected option": {
                            "type": "string",
                            "description": "The option selected by the LLM.",
                            "enum": ["A", "B", "C", "D", "E", "No option was selected","The answer lies beyond the options provided" ,"More than one option was selected", "I am not sure."]
                        },
                    },
                    "required": ["selected option"],
                },
            }
        }]

    
        message=[{'role': 'user', 
                   'content': f"""
                   Possible options that LLM could chose:
                   {choice}
                   
                   LLM Answer was:
                   {llmresponsetext}
                   """
                   }
                 ]
        
        extraction_response = await client.chat.completions.create(
            model = model_name,
            messages = message,
            tools= correct_answer_schema,
            tool_choice=  {"type": "function", "function": {"name": "extract_true_choice"}},
            max_tokens=model_max_tokens,
            temperature=model_temperature,
            timeout=request_timeout,
        )
        print(extraction_response)
        correct_answer_dic=json.loads(extraction_response.choices[0].message.tool_calls[0].function.arguments)
        structure_answer=correct_answer_dic['selected option']
        correctness=check_answer_correctness(answer=structure_answer, truth=true_answer)

    except Exception as e:
        structure_answer=f'ERROR in extraction: {e}'
        correctness=None
        
    finally:
        return index, structure_answer, correctness

    # Example use
    #ClaudeWebrawHiaku_rawoutput="""As a seasoned gastroenterologist, I would approach this complex question from the gastroenterology board exam with a deep understanding of the advanced concepts in the field.
    #Based on my expert analysis, the most accurate option is Option A: Decreased morbidity. Colonic stenting, as a minimally invasive approach, has the potential to reduce postoperative complications and overall morbidity when compared to emergency surgery.
    #
    #My confidence in this decision is 8 out of 10. While the impact on mortality is less clear, the decreased surgical trauma and associated complications associated with colonic stenting strongly support the conclusion that it can lead to decreased morbidity in the short term.
    #"""
    #
    #structure_answer = await OpenAI_Extract_SelectedOption_fromtext(ClaudeWebrawHiaku_rawoutput)
    #structure_answer


async def handle_llm_response_answerextraction(excel_file_path: str, list_rawoutput_columns: list,
                                               open_at_end: bool = False, with_second_try_ifERROR=False,
                                               number_of_task_to_save: int = 15,add_delay_sec:int=5,):
    df = pd.read_excel(excel_file_path)
    
    try:
        # Read Excel file into DataFrame
            
        
        # Double check if column names mentioned in list_rawoutput_columns are valid
        for column_name in list_rawoutput_columns:
            if column_name not in df.columns:
                print(f"Error: Column '{column_name}' not found in the DataFrame.")
                return
        
        # Loop through each column in list_rawoutput_columns
        for column_name in list_rawoutput_columns:
            prefix = column_name.rsplit('_', 1)[0]
            correctness_col_name = prefix + '_correctness'
            answer_col_name = prefix + '_answer'
            previousextract_col_name = prefix + '_ANSWEREXTRACTED'
            
            if previousextract_col_name not in df.columns:
                df[previousextract_col_name] = ''
            
            tasks = []
            i = 0
            max_index = df.index.max()
            # a lis to append second tries and avoid more tries in an never-ending loop (third, forth, ..)
            second_try_index_column_list=[]
            
            # Loop through each row in the DataFrame
            for index, row in df.iterrows():
                if row[previousextract_col_name] != 'ANSWEREXTRACTED' and row[column_name]!= None:
                    task = asyncio.create_task(OpenAI_Extract_SelectedOption_fromtext(index=index, llmresponsetext=row[str(column_name)], choice=row['Options'], true_answer=row['Correct Answer']))
                    tasks.append(task)
                    i+=1
                    
                    if i == number_of_task_to_save or index == max_index:
                        results = await asyncio.gather(*tasks)  # Gather only the tasks
                        
                        for result in results:
                            idx, selected_option, correctness = result
                            df.at[idx, answer_col_name] = selected_option
                            print("Extracted Option:", selected_option)
                            df.at[idx, correctness_col_name] = correctness
                            
                            df.at[idx, previousextract_col_name] = 'ANSWEREXTRACTED'
                            
                            if with_second_try_ifERROR:
                                column_index_key_4doublecheck=f"{column_name}_{idx}"
                                if column_index_key_4doublecheck not in second_try_index_column_list and selected_option.startswith('ERROR'):
                                    task = asyncio.create_task(OpenAI_Extract_SelectedOption_fromtext(index=idx, llmresponsetext=df.at[idx, str(column_name)], choice=df.at[idx, 'Options'], true_answer=df.at[idx, 'Correct Answer']))
                                    tasks.append(task)
                                    second_try_index_column_list.append(column_index_key_4doublecheck)
                                    df.at[idx, previousextract_col_name] = ""
                                    print(f'Second try added to asyncio task list for: {column_index_key_4doublecheck}')
                                    
                        try:
                            
                            prefix_columns = [col for col in df.columns if col.startswith(prefix)]
                            other_columns = [col for col in df.columns if not col.startswith(prefix)]
                            df = df[other_columns + prefix_columns]  # reordering columns to have prefixed columns at the end

                            # Save the DataFrame to an Excel file
                            df.to_excel(excel_file_path, index=False)
                            print(f"Draft excel file saved at {excel_file_path}")
                        except Exception as e:
                            print(f"Error in saving temporary excel: {e}")
                            continue
                        
                        i = 0
                        tasks = []
                        print(f'Sleep like a baby at index: {index} and column name: {column_name}')
                        await asyncio.sleep(add_delay_sec)
                        
                
    except Exception as e:
        print(f"Error generating extracted answers {e}")
    except CancelledError or InterruptedError:
        print(f"Interrupted. Cleaning up ... ")
    finally:    
        df = save_and_open_excel(df, excel_file_path, open_at_end)
        return df



# #Example use ----------------------
# excel_file_path=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\E1_Bestprompt-2022_round_Julyy.xlsx"
# list_rawoutput_columns=[#"gemma2-9b-it_rawoutput",
#                         #"Gemma2-9b-Q8_rawoutput",
#                         #"Mistral-7B-T_rawoutput",
#                         #"Llama-2-13b_rawoutput",
#                         "Phi3-medium14b-Q6_rawoutput"
#                         ]
# open_at_end=True
# with_second_try_ifERROR=True
# number_of_task_to_save=10

# final_df= await handle_llm_response_answerextraction(excel_file_path=excel_file_path, list_rawoutput_columns=list_rawoutput_columns,
#                               open_at_end=open_at_end, with_second_try_ifERROR=with_second_try_ifERROR,
#                               number_of_task_to_save=number_of_task_to_save)
