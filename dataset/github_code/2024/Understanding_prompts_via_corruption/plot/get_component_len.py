from transformers import AutoTokenizer
import re
import json

inline_instruction_mapping = {
    'task116_com2sense_commonsense_reasoning': 'Does this statement make sense to you?',
    'task1344_glue_entailment_classification': 'Does Sentence 1 entail Sentence 2?',
    'task1645_medical_question_pair_dataset_text_classification': 'Are these two questions similar or dissimilar?',
    'task843_financial_phrasebank_classification': "Is the sentiment of the sentence 'Negative', 'Neutral', or 'Positive'?" ,
    'task512_twitter_emotion_classification': 'Which emotion is expressed in this tweet?',
    'task1564_triviaqa_answer_generation': 'The answer to this question is',
    'task835_mathdataset_answer_generation': 'The answer to this math problem is',
    'task284_imdb_classification': 'Is this review positive or negative?',
    'task379_agnews_topic_classification': 'What label best describes this news article?',
    'task828_copa_commonsense_cause_effect': 'Is the second sentence cause or effect of the first sentence?', # replace one with sentence for end window corrption
    'task1346_glue_cola_grammatical_correctness_classification': 'Is this sentence meaningful and grammatically correct?',
}

random_words_inline_inst_mapping = {
    'task116_com2sense_commonsense_reasoning': 'complaisant pilfer Honolulu Dene.',
    'task1344_glue_entailment_classification': 'complaisant pilfer Honolulu Deneb.',
    'task1645_medical_question_pair_dataset_text_classification': 'complaisant pilfer Honolulu Deneb.',
    'task843_financial_phrasebank_classification': 'complaisant pilfer Honolulu Deneb pea A&P Vladimir bazaar admittance.' ,
    'task512_twitter_emotion_classification': 'complaisant pilfer Honolulu Dene.',
    'task1564_triviaqa_answer_generation': 'complaisant pilfer Honolulu D',
    'task835_mathdataset_answer_generation': 'complaisant pilfer Honolulu Dene',
    'task284_imdb_classification': 'complaisant pilfer Honolulu D.',
    'task379_agnews_topic_classification': 'complaisant pilfer Honolulu Dene.',
    'task828_copa_commonsense_cause_effect': 'complaisant pilfer Honolulu Deneb pea A.',
    'task1346_glue_cola_grammatical_correctness_classification': 'complaisant pilfer Honolulu Deneb.',
}

component_colors = {
    'taskinstr': 'yellow',
    'input': 'cyan',
    'inline': 'orange',
    'label': 'green',
    'test_instance': 'grey',
    'garbage': 'brown',
    'sep' : 'blue',
    'period': 'red',
}

garbage_text = "kjghg tgwbg dw eurfter eytfway udwqted uwd."

def get_components_color(all_components_names):
    '''
    get colors for each components in prompt
    '''
    all_components_color = []
    for item in all_components_names:
        if item == "TaskInst":
            all_components_color.append(component_colors["taskinstr"])

        if re.match(r'^D\d+_input$', item):
            all_components_color.append(component_colors["input"])
        
        if "inline" in item:
            all_components_color.append(component_colors["inline"])

        if re.match(r'^D\d+_label$', item):
            all_components_color.append(component_colors["label"])

        if item == "Test_input":
            all_components_color.append(component_colors["test_instance"])

        if "\n\n" in item:
            all_components_color.append(component_colors["sep"])

        if item == ".":
            all_components_color.append(component_colors["period"])

        if item == "garbage":
            all_components_color.append(component_colors["garbage"])
    
    return all_components_color


def get_item_by_id(corr_name, sample_id, modelname):

    if "gpt" in modelname:
        modelname = "6.05B_gpt-j-6B"
    else:
        modelname = "30.33B_opt-30b"

    file_path = f"../predictions/{modelname}/all_corruptions/predicted_examples_all_instr_n_demo.jsonl"

    if "random" in corr_name:
        file_path = f"../predictions/{modelname}/all_corruptions/predicted_examples_instr_randomwords.jsonl"

    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            if item['id'] == sample_id:
                return item
    return None

def get_splits(splitted_item, corr_name, mapping_task, demo_no):  

    if "0" in corr_name:
        if demo_no in [0, 1, 2, 3]: 
            splitted_item = [splitted_item[0], splitted_item[1]]
        else:   
            splitted_item = [splitted_item[0], mapping_task, splitted_item[1]]

    if "1" in corr_name:
        if demo_no in [1, 2, 3]: 
            splitted_item = [splitted_item[0], splitted_item[1]]
        else:
            splitted_item = [splitted_item[0], mapping_task, splitted_item[1]]

    return  splitted_item          

def compute_segment_lengths_anytask(prompt, modelname):

    input_text = prompt["Instance"]["input"]
    
    garbage_text = "kjghg tgwbg dw eurfter eytfway udwqted uwd."
    if "garbage" in prompt["Corruption_name"]: 
        if prompt["Corruption_name"] == "end_window_garbage_in_mid":
             garbage_text = " " + garbage_text
        input_text = input_text.replace(garbage_text, "")

    print(input_text)

    input_text = input_text.split(".\n\n")   # we have [instr, demo1, demo2, demo3, demo4, test input+inline instr]

    # Iterate through the original list
    components_text_list = []
    for components in input_text:
        components_text_list.append(components)  # Add the original element
        components_text_list.append(".") 
        components_text_list.append("\n\n")   # Add "\n\n" after the original element

    print(components_text_list)
    # Remove the trailing "\n\n" from the last element if needed
    components_text_list[0] = components_text_list[0] + '.' # this is due to separated by .\n\n but Task desc should end with a period.
    components_text_list.pop(1)  # [instr, "\n\n",  demo1, ".", "\n\n", demo2, ".", "\n\n", demo3, ".", "\n\n", demo4, ".", "\n\n", test input+inline instr]
    components_text_list.pop() # remove last "\n\n"
    components_text_list.pop() # remove last "."

    print("COMPONENTS before START ", components_text_list)

    if prompt["Corruption_name"] in ["all_instr_n_demo", "instr_randomwords", "labels_randomwords", "inline_instr_in_0_demo"]: 
                inline_instruction_mapping['task828_copa_commonsense_cause_effect'] = 'Is the second sentence cause or effect of the first one?'

    if prompt["Corruption_name"] in [
        "all_instr_n_demo", 
        "instr_randomwords", 
        "labels_randomwords", 
        ] :
        # separate the components of the demostrations
        task = prompt["Task"]

        all_components = []
        for item in components_text_list:
    
            mapping = inline_instruction_mapping
            if prompt["Corruption_name"] == "instr_randomwords":
                mapping = random_words_inline_inst_mapping

            if mapping[task] in item:
                print("it is demo")
                # means it is demos, so split it in 3 components
                splitted_item = item.split(mapping[task])
                splitted_item = [splitted_item[0], mapping[task], splitted_item[1]]
                all_components.extend(splitted_item)
            else:
                all_components.append(item) # [instr, "\n\n",  demo1_inline, inline_q, label1 , 
                                            # "\n\n", demo2_inline, inline_q, label2 , "\n\n", 
                                            # demo3_inline, inline_q, label3 , "\n\n", 
                                            # demo4_inline, inline_q, label4 , "\n\n", 
                                            # test input, inline instr, '']                          
        all_components.pop() # last item is empty

        all_components_names = [
            "TaskInst", "\n\n",  
            "D1_input" , "D1_inline", "D1_label" , "." , "\n\n",  
            "D2_input" , "D2_inline", "D2_label" , "." , "\n\n", 
            "D3_input" , "D3_inline", "D3_label" , "." , "\n\n", 
            "D4_input" , "D4_inline", "D4_label" , "." , "\n\n",
            "Test_input", "test_inline"]

        if prompt["Corruption_name"] == "end_window_garbage_in_begin":
                all_components_names.insert(0, "garbage")
                all_components.insert(0, garbage_text)

        if prompt["Corruption_name"] == "end_window_garbage_in_mid":
                all_components_names.insert(11, "garbage")
                all_components.insert(11, garbage_text)

        if prompt["Corruption_name"] == "end_window_garbage_in_end":
                all_components_names.append("garbage")
                space = "  " if "trivia" in task or "math" in task else " "
                garbage_text = space + garbage_text
                all_components.append(garbage_text)

        print("final_components", all_components)
        
    # ############################# repeated text corruptions, code is repeated for convenience ######################
    if prompt["Corruption_name"] in [ 
    'inline_instr_in_0_demo', 
    'inline_instr_in_1_demo', 
    'randomwords_inline_instr_in_0_demo',
    'randomwords_inline_instr_in_1_demo'] :

        task = prompt["Task"]
        sample_id = prompt["id"]

        # get the same sample from basefile to construct components. basefile can be 
        # predicted_examples_all_instr_n_demo or predicted_examples_instr_randomwords
        base_file_same_sample = get_item_by_id(prompt["Corruption_name"], sample_id, modelname)
        base_file_same_prompt = base_file_same_sample["Instance"]["input"]
        base_file_same_prompt = base_file_same_prompt.split(".\n\n")

        components_text_list = [] # components from base file
        for components in base_file_same_prompt:
                components_text_list.append(components)  # Add the original element
                components_text_list.append(".") 
                components_text_list.append("\n\n")   # Add "\n\n" after the original element

        # Remove the trailing "\n\n" from the last element if needed
        components_text_list[0] = components_text_list[0] # this is due to separated by .\n\n but Task desc should end with a period.
        components_text_list.pop(1)  # [instr, "\n\n",  demo1, ".", "\n\n", demo2, ".", "\n\n", demo3, ".", "\n\n", demo4, ".", "\n\n", test input+inline instr]
        components_text_list.pop() # remove last "\n\n"
        components_text_list.pop() # remove last "." [instr, "\n\n",  demo1, "\n\n", demo2, "\n\n", demo3, "\n\n", demo4, "\n\n", test input+inline instr]

        components_text_list[0] = input_text[0] + '.' # as we removed this before, this is task desc + "."

        print("COMPONENTS", components_text_list)

        all_components = []
        demo_no = 0
        for item in components_text_list: # iterating over components from base file
            
            mapping = inline_instruction_mapping
            if "random" in prompt["Corruption_name"]:
                mapping = random_words_inline_inst_mapping

            if mapping[task] in item:
                splitted_item = item.split(mapping[task])
                splitted_item = get_splits(splitted_item, prompt["Corruption_name"], mapping[task], demo_no)  
                all_components.extend(splitted_item)
                demo_no += 1
            else:
                all_components.append(item) # [instr, "\n\n",  demo1_inline, inline_q, label1 , ".", "\n\n", 
                                            # demo2_inline, inline_q, label2 ,  ".", "\n\n", 
                                            # demo3_inline, inline_q, label3 ,  ".", "\n\n", 
                                            # demo4_inline, inline_q, label4 ,  ".", "\n\n", 
                                            # test input, inline instr, '']                          
        all_components.pop() # last item is empty
                        
        all_components_names = [
                "TaskInst", "\n\n",  
                "D1_input" , "D1_inline", "D1_label" ,  ".", "\n\n",  
                "D2_input" , "D2_inline", "D2_label" , ".", "\n\n",   
                "D3_input" , "D3_inline", "D3_label" , ".", "\n\n",   
                "D4_input" , "D4_inline", "D4_label" , ".", "\n\n",  
                "Test_input", "test_inline"]

        if "0" in prompt["Corruption_name"]:
            components_to_remove = ["D1_inline", "D2_inline", "D3_inline", "D4_inline"]
        else:
            components_to_remove = ["D2_inline", "D3_inline", "D4_inline"]
        all_components_names = [comp for comp in all_components_names if comp not in components_to_remove]

    ##################################################################################################
    
    # get colors
    colors = get_components_color(all_components_names)

    # Tokenize the input text
    modelname = modelname
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    total_tokens_in_prompt = len(tokenizer.tokenize(prompt["Instance"]["input"])) # from corruption file and not basefile

    # Calculate the length of each tokenized components
    component_token_lengths = [len(tokenizer.tokenize(segment)) for segment in all_components]

    # sanity check
    print(task)
    print(total_tokens_in_prompt)
    print(sum(component_token_lengths))

    assert total_tokens_in_prompt == sum(component_token_lengths), "Values are not the same"

    print("Total length of the tokenized prompt", sum(component_token_lengths))
    print("Components length", component_token_lengths)
    print(len(component_token_lengths))

    print("all_components", all_components)
    print("all_components_names", all_components_names)
    print("component_token_lengths", component_token_lengths)

    assert len(all_components) == len(all_components_names) == len(colors) == len(component_token_lengths), "components list is not of same length"

    return all_components_names, component_token_lengths, colors