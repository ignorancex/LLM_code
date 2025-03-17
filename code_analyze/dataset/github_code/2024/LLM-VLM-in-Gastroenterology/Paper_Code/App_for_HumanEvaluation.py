#
### STreamlit for answer review
#conda create --name streamlit_env
#conda activate streamlit_env
#conda install streamlit pandas openpyxl
#streamlit run Evaluation_streamlitapp.py
#conda deactivate


import streamlit as st
import pandas as pd
import time
import os
import shutil
import ast
import re
def find_and_concat_substrings(input_text, start_delimiter, end_delimiter=None, separator=' '):
    # Escape delimiters to handle any characters that might be interpreted as special regex characters
    start = re.escape(start_delimiter)
    if end_delimiter:
        end = re.escape(end_delimiter)
    else:
        # Capture until the end of the line or string if no end delimiter is given
        end = '$'
    
    # Create a regex pattern to find all occurrences of text between the start and end delimiters
    pattern = f"{start}(.*?){end}"
    
    # Find all matches using re.findall which returns all non-overlapping matches of pattern in string
    matches = re.findall(pattern, input_text)
    
    # Join all matches using the specified separator
    concatenated_string = separator.join(matches)
    
    return concatenated_string

def display_question_options_llm_sidebar(llm:str,df,index:int):
    df_row=df.iloc[index]
    question = df_row["Question"]
    options = df_row[f"Options"]
    correct_answer = df_row["Correct Answer"]
    current_evaluation = df_row[f"{llm}_correctness"]
    
    # Display options
    st.write(f"LLM: {llm}   row_index: {index}")
    st.write(f"Options: {options}")
    # Display correct answer
    st.write(f"Correct Answer: {correct_answer}")
    # Display current evaluation
    st.write(f"Current Evaluation: {current_evaluation}")
        
# Function to display question, options, and LLm response for manual review
def display_question_options_llm(llm:str,df,index:int, tab_i=0):
    df_row=df.iloc[index]
    question = df_row["Question"]
    options = df_row[f"Options"]
    correct_answer = df_row["Correct Answer"]
    llm_response = df_row[f"{llm}{st.session_state.ending}"]
    llm_answer = df_row[f"{llm}_answer"]
    current_evaluation = df_row[f"{llm}_correctness"]
    # Display question
    st.subheader("Question:")
    st.write(question)

    # Display options
    st.subheader("Options:")
    st.write(options)

    # Display correct answer
    st.subheader("Correct Answer:")
    st.write(correct_answer)

    # Display LLM response
    st.subheader("LLM Response:",)
    
    col1_toclean,col2_toclean=st.columns([1,1])
    with col1_toclean:
        what_to_remove_before=st.text_input("What should I look for, and remove text before that",key=f"remove_before_textinput_{tab_i}")
    with col2_toclean:
        what_to_remove_after=st.text_input("What should I look for, and remove text after that ?",key=f"remove_after_textinput_{tab_i}")
    
    if what_to_remove_before and not what_to_remove_after:
        bef_result = find_and_concat_substrings(llm_response, what_to_remove_before, "", separator='\n')
        if bef_result:
            bef_result=bef_result.split(r"\n")
            for chunk in bef_result:
                st.write(chunk)
            
            st.write(bef_result)  
    elif what_to_remove_before and what_to_remove_after:
        befaf_result=find_and_concat_substrings(llm_response, what_to_remove_before, what_to_remove_after, separator='\n')
        if befaf_result:
            befaf_result=befaf_result.split(r"\n")
            for chunk in befaf_result:
                st.write(chunk)
    else:
        rawoutput=llm_response.split(r"\n")    
        for chunk in rawoutput:
            st.write(chunk)
    
    # Display LLM answer
    st.subheader("LLM Answer:")
    st.write(llm_answer)
    
    # Display current evaluation
    st.subheader("Current Evaluation:")
    st.write(current_evaluation)


          
# Function to read the content of a text file
def read_text_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

# Function to write content to a text file
def write_text_file(file_path, content):
    with open(file_path, "w") as file:
        while len(read_text_file(file_path))!=content: 
            file.write(content)
        
def Create_backup2work_Gostreamlit(original_file_path):
    # Split the input filepath into directory path and filename
    directory_path, filename = os.path.split(original_file_path)
    name, extension = os.path.splitext(filename)
    
    backup2work_name = f"{name}_GoStreamlit{extension}"
    backup2work_filepath = os.path.join(directory_path, backup2work_name)

    shutil.copyfile(original_file_path, backup2work_filepath)
    
    return backup2work_filepath


def index_minus_one(new):   
    if st.session_state["index"] > 0:
        st.session_state.current_index -= 1
        st.write(f'<span style="blue: red;">Moved to {st.session_state.current_index}</span>', unsafe_allow_html=True)
    elif st.session_state["index"] <= 0:
        st.write('<span style="color: red;">there is no previous data</span>',unsafe_allow_html=True)
    else:
        pass
    return


css_blue_line = """
<style>
/* Change the color and thickness of the line */
hr {
    border: none;
    border-top: 2px solid blue; /* Change the color */
    height: 2px; /* Change the thickness */
}
</style>
"""

def current_index_plus_one(current_index:int,list_length:int):
    try:
        current_index +=1
        if current_index > list_length -1:
            st.session_state.current_index=0
        st.session_state.current_index=current_index
    except Exception as e: 
        st.session_state.current_index=0
        print(f"Errpr in current_index_plus_one: {e}")
    return

def current_index_minus_one(current_index:int,list_length:int):
    try:
        current_index -=1
        if current_index < 0:
            st.session_state.current_index= list_length-1
        st.session_state.current_index=current_index
    except Exception as e: 
        st.session_state.current_index=0
        print(f"Errpr in current_index_plus_one: {e}")
    return

def validated_indexes_string_2list(list_of_valid_indexes:str):
    """
    Extracts all list representations from a given string, evaluates them,
    and combines them into a single list.

    Args:
    input_string (str): A string that contains multiple list representations.

    Returns:
    list: A combined list of all elements from the list representations in the input string.
    """
    # Use regex to find all list-like substrings
    list_strings = re.findall(r'\[.*?\]', list_of_valid_indexes)

    # Initialize an empty list to hold all the elements
    combined_list = []

    # Loop through each substring, convert it to a list, and extend the combined list
    for list_string in list_strings:
        try:
            # Convert the substring into a Python list or integer
            element = ast.literal_eval(list_string)
            # If the element is a list, extend the combined list, otherwise appen
            if isinstance(element, list):
                combined_list.extend(element)
            else:
                combined_list.append(element)
        except SyntaxError:
            print(f"Skipping invalid list: {list_string}")
    unique_list = list(set(combined_list))
    return unique_list    


# Streamlit App
def main():
    st. set_page_config(layout="wide") 
    st.title("LLM Evaluation App")
    with st.sidebar:
        st.write(st.session_state)

    
    # Upload Excel file
    uploaded_file=original_file_path=st.text_input("Insert your excel file path",value=r"C:\Users\LEGION\Documents\GIT\LLM_answer_GIBoard\DO_NOT_PUBLISH\ACG self asses\__Milestone Datasets\E2-2022-imageQ-CompleteCaptions_GenerationComplete_prepared_evaluated_Final_wAPIs.xlsx")
    st.session_state.ending=ending=st.text_input("columns that end wit ...", value="_rawoutput")
    #uploaded_file = Create_backup2work_Gostreamlit(original_file_path)
    
    
    if uploaded_file is not None:

        # Load data
        df = pd.read_excel(uploaded_file)

        # Select LLMs for evaluation
        st.header("Select LLMs for Evaluation")
        # Get list of LLMs from columns ending with "_rawoutput"
        llm_columns = [col for col in df.columns if col.endswith(ending)]
        selected_llms_prefix=[llm.split(ending)[0] for llm in llm_columns]
        llm = st.radio("Select LLMs", selected_llms_prefix, index =None)

        if llm:
            
            st.header(f"Evaluation for {llm}")
            # Display unique values and counts in evaluation column
            col1_count,col2_count  = st.columns([1,1])
            with col1_count:
                values_counts = df[llm+"_correctness"].value_counts()
                st.write(values_counts)
                selected_correctness_tag = st.radio("Select _correctness values to change", list(df[llm+"_correctness"].unique())+[None],
                                    horizontal=True, index =None)

            with col2_count:
                st.write(df[llm+"_answer"].value_counts()) 
                selected_answer_tag = st.radio("Select _answer values to change", list(df[llm+"_answer"].unique())+[None],
                                                    horizontal=True, index =None)

            #desired_columns = [col for col in df.columns if col.startswith(llm)] # this will be used for showing "Current LLM columns"
            if selected_answer_tag != None and selected_correctness_tag == None:
                # Allow user to manually review and update evaluation
                
                st.subheader("Manual Review")
                st.session_state.valid_indexes = df[df[llm+"_answer"] == selected_answer_tag].index.tolist()
                st.session_state.current_index=0
                
            elif selected_answer_tag == None and selected_correctness_tag != None:
                st.subheader("Manual Review")
                st.session_state.valid_indexes = df[df[llm+"_correctness"] == selected_correctness_tag].index.tolist()
                st.session_state.current_index=0
                
                
            elif selected_answer_tag!=None and selected_correctness_tag != None:
                st.session_state.valid_indexes=None
                st.write('<span style="color: red;">WARNING ! You should either chose _correctness or _answer values !!!</span>',
                         unsafe_allow_html=True)
            else:
                st.session_state.valid_indexes=None
                st.write('<span style="color: red;">You should either chose _correctness or _answer values.</span>',
                         unsafe_allow_html=True)  
            
            
            if st.session_state.valid_indexes:
                st.success("Copy this list of filtered indexes and paste in the box below")
                st.success(f"{st.session_state.valid_indexes}")
            
            list_of_valid_indexes=st.text_input("The list of indexes to show/modify")        
            if list_of_valid_indexes:
                list_of_valid_indexes = validated_indexes_string_2list(list_of_valid_indexes)
                # Create tabs for each valid index
                tab_labels = [f"Index {idx}" for idx in list_of_valid_indexes]
                tabs = st.tabs(tab_labels)

                for i, idx in enumerate(list_of_valid_indexes):
                    with tabs[i]:

                        display_question_options_llm(llm, df, index=idx,tab_i=i)
                         #---------------------
                        st.markdown(css_blue_line, unsafe_allow_html=True)
                        st.markdown("<hr>", unsafe_allow_html=True)
                        data_col1,data_col2=st.columns([1,1])
                        with data_col2:
                            new_evaluation = new_answer=None
                            new_evaluation = st.radio("Evaluation (correctness)", ["Correct", "Incorrect", "EOP", "NOP", "2OP", "NoA"], index =None,horizontal=True,key=f"correctness_{i}")
                            new_answer = st.text_input("answer",key=f"answer_{i}")

                            if st.button("Update and Save",key=f"update_{i}"):
                                if new_evaluation:
                                    df.at[list_of_valid_indexes[i], llm+"_correctness"] = new_evaluation
                                if new_answer:
                                    df.at[list_of_valid_indexes[i], llm+"_answer"] = f"Resolved: {new_answer}"
                            
                                try:
                                    df.to_excel(uploaded_file, index=False)
                                    st.success(f"Excel file saved successfully at {uploaded_file}!")
                                except Exception as e:
                                    st.error(f"Failed to save the Excel file: {e}")
                        with data_col1:
                            display_question_options_llm_sidebar(llm, df, index=idx)
                        st.markdown("<hr>", unsafe_allow_html=True)
                        #---------------------
            else:
                st.warning(f"""No questions found. \n
                        Answer tag: {selected_answer_tag}          Correctness tag: {selected_correctness_tag}
                        """)


    if st.button("Open Excel"):
        try:
            os.system(f'start excel "{original_file_path}"')
        except Exception as e:
            st.write(f"ERROR in opening Excel: {e}")
            
    st.header("Updated Dataframe")    
    #st.dataframe(df)
    if st.button("Enable data editor"):
        temp_df=df.copy()
        temp_df = st.data_editor(temp_df)
        if st.button('Save temporary data '):
            st.write("I will save it to the same location with the name _temp at the end of file name")
            temp_df_path = original_file_path+"_temp.xlsx"
            temp_df.to_excel(temp_df_path, index=False)
            st.success(f"temp dataframe saved at {temp_df_path}")
            
            if st.button('DANGEROUS ! Save it as original File !'):
                temp_df.to_excel(original_file_path, index=False)
                st.success(f"temp dataframe overwrite original file at {temp_df_path}")
                
                        

if __name__ == "__main__":
    main()

