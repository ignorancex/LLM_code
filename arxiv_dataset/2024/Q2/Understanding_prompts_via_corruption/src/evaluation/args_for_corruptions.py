def args_for_corruptions(corruption_name):
    '''return argument based on the corruption name

    Argument: 
        corruption_name (str) : corrutpion name e.g. 'instr_n_demo'
    
    Return:
        corruption_base_file (str)
        add_task_definition (bool)
        num_pos_examples (int)
        corruption_name (str)
        placement_based_corruption (str)
    '''
    placement_based_corruption = None

    if corruption_name == "empty":
        corruption_base_file = "no_inline_instr"
        add_task_definition = False 
        num_pos_examples = 0 
        corruption_name = "empty"

    if corruption_name == "only_instruction":
        #  only_instruction - [I | x]
        corruption_base_file = "no_inline_instr"
        add_task_definition = True 
        num_pos_examples = 0 
        corruption_name = "only_instructions"

    if corruption_name == "only_inline":
        corruption_base_file = "baseline"
        add_task_definition = False 
        num_pos_examples = 0 
        corruption_name = "only_inline"
    
    if corruption_name == "both_instructions":
        corruption_base_file = "baseline"
        add_task_definition = True 
        num_pos_examples = 0 
        corruption_name = "both_instructions"
    

    if corruption_name == "only_demostration":
        corruption_base_file = "no_inline_instr"
        add_task_definition = False 
        num_pos_examples = 4 
        corruption_name = "only_demostration"

    if corruption_name == "instr_n_demo":
        #  instr_n_demo - [I | xy | xy | xy| xy |x] same as no_inline_instr
        corruption_base_file = "no_inline_instr"
        add_task_definition = True 
        num_pos_examples = 4 
        corruption_name = "instr_n_demo"

    if corruption_name == "inline_n_demo":
        corruption_base_file = "baseline"
        add_task_definition = False 
        num_pos_examples = 4 
        corruption_name = "inline_n_demo"

    if corruption_name == "all_instr_n_demo":
        corruption_base_file = "baseline" # include in-line instruction
        add_task_definition = True 
        num_pos_examples = 4 
        corruption_name = "all_instr_n_demo"

    corruptions_list = [
        'instr_randomwords',
        'instr_frequentwords',
        'wrong_labels_labelspace',
        'labels_randomwords',
        'input_ood',
        'input_randomwords',
        'inline_instr_in_0_demo', # repeated text corruptions
        'inline_instr_in_1_demo',
        'inline_instr_in_2_demo',
        'inline_instr_in_3_demo',
        'randomwords_inline_instr_in_0_demo', # repeated text corruptions -  random words case
        'randomwords_inline_instr_in_1_demo', 
        'randomwords_inline_instr_in_2_demo',
        'randomwords_inline_instr_in_3_demo',
        ]

    for corrup in corruptions_list: # all_instr_n_demo (baseline) and apply corruptions on it (we already have corrupted files in processed_tasks folder)
        if corrup == corruption_name:
            corruption_base_file = corrup
            add_task_definition = True 
            num_pos_examples = 4 
            corruption_name = corrup 
   
    placement_based_corruptions_list = [
        'input_empty', 
        'label_empty', 
        'instr_after_demo',
        ]

    # for thiese corruptions there are some conditions on in the data_collator.py
    for corrup in placement_based_corruptions_list: # all_instr_n_demo (baseline) and apply corruptions on it (we already have corrupted files in processed_tasks folder)
        if corrup == corruption_name:
            corruption_base_file = "baseline"
            add_task_definition = True 
            num_pos_examples = 4 
            corruption_name = corrup 
            placement_based_corruption = corrup

    return corruption_base_file, add_task_definition, num_pos_examples, corruption_name, placement_based_corruption
    