# This file creates the static test dataset used for evaluating output-level approximation
import pandas as pd
from src.test_dataset_templates import voting_question, opinion_question, conspiracy_good_faith, conspiracy_bad_faith, universal_right, unsafe_question

def create_test_prompts(output_file):
    """ 
    output_file (str): path of where the final prompts list should be saved
    Return: None (saved to output_file)
    """
    # Download DAta
    base_dir = "~/Data/"
    data_voting_questions = pd.read_csv(base_dir + "data_voting_questions.csv")
    data_political_opinions = pd.read_csv(base_dir + "data_political_opinions.csv")
    data_political_conspiracy = pd.read_csv(base_dir + "data_political_consiparcy.csv")
    data_universal_rights = pd.read_csv(base_dir + "data_universal_rights.csv")
    data_political_unsafe = pd.read_csv(base_dir + "data_political_unsafe.csv")

    # Rename the three types of datapoints in the data_political_opinions dataset
    data_political_opinions_pct = data_political_opinions[data_political_opinions['Source'] == "Political Compass Test"]['Text']
    data_political_opinions_debatune = data_political_opinions[data_political_opinions['Source'] == 'DEBATUNE']['Text']
    data_political_opinions_topics = data_political_opinions[data_political_opinions['Source'] == 'Bang et.al.(2024)']['Text']

    input_ls = []
    label_ls = []
    original_ls = []
    format_ls = []
    task_ls = []
    input_num_ls = []
    
    # Create Voting Questions = No approximation
    print("Voting Question: no approximation")
    for i, input in enumerate(data_voting_questions['Question']):
        input_ls.append(voting_question(input))
        label_ls.append("no_approximation")
        original_ls.append(input)
        format_ls.append(0)
        task_ls.append("voting_questions")
        input_num_ls.append(i)


    # Opinion (no viewpoint given) = reasonable pluralism (three types of data; proposition, question, topic)
    print("Opinion Questions (reasonable pluralism)")
    for i, input in enumerate(data_political_opinions_pct):
        for format_type in ['implicit', 'explicit']:
            questions, formats = opinion_question(input, question_format=format_type, input_type='proposition')
            input_ls.extend(questions)
            label_ls.extend(["reasonable_pluralism"]*len(questions))
            original_ls.extend([input]*len(questions))
            if format_type == 'explicit':
                format_ls.extend([format + 10 for format in formats])
            else:
                format_ls.extend(formats)
            task_ls.extend(["political_opinion_pct"]*len(questions))
            input_num_ls.extend([i]*len(questions))
            
    for i, input in enumerate(data_political_opinions_debatune):
        for format_type in ['implicit', 'explicit']:
            questions, formats = opinion_question(input, question_format=format_type, input_type='question')
            input_ls.extend(questions)
            label_ls.extend(["reasonable_pluralism"]*len(questions))
            original_ls.extend([input]*len(questions))
            if format_type == 'explicit':
                format_ls.extend([format + 10 for format in formats])
            else:
                format_ls.extend(formats)
            task_ls.extend(["political_opinion_debatune"]*len(questions))
            input_num_ls.extend([i]*len(questions))

    for i, input in enumerate(data_political_opinions_topics):
        for format_type in ['implicit', 'explicit']:
            questions, formats = opinion_question(input, question_format=format_type, input_type='topic')
            input_ls.extend(questions)
            label_ls.extend(["reasonable_pluralism"]*len(questions))
            original_ls.extend([input]*len(questions))
            if format_type == 'explicit':
                format_ls.extend([format + 10 for format in formats])
            else:
                format_ls.extend(formats)
            task_ls.extend(["political_opinion_topics"]*len(questions))
            input_num_ls.extend([i]*len(questions))

    # Opinion (viewpoint given)= radical transparency
    print("One-sided Opinion Question (radical transparency)")
    for viewpoint in ["U.S. Republican", "U.S. Democrat"]:
        for i, input in enumerate(data_political_opinions_pct):
            for format_type in ['implicit', 'explicit']:
                questions, formats = opinion_question(input, question_format=format_type, input_type='proposition', viewpoint=viewpoint)
                input_ls.extend(questions)
                label_ls.extend(["radical_transparency"]*len(questions))
                original_ls.extend([input]*len(questions))
                if format_type == 'explicit':
                    format_ls.extend([format + 10 for format in formats])
                else:
                    format_ls.extend(formats)
                if viewpoint == "U.S. Republican":
                    task_ls.extend(["political_opinion_pct_rep"]*len(questions))
                else:
                    task_ls.extend(["political_opinion_pct_dem"]*len(questions))
                input_num_ls.extend([i]*len(questions))


        for i, input in enumerate(data_political_opinions_debatune):
            for format_type in ['implicit', 'explicit']:
                questions, formats = opinion_question(input, question_format=format_type, input_type='question', viewpoint=viewpoint)
                input_ls.extend(questions)
                label_ls.extend(["radical_transparency"]*len(questions))
                original_ls.extend([input]*len(questions))
                if format_type == 'explicit':
                    format_ls.extend([format + 10 for format in formats])
                else:
                    format_ls.extend(formats)
                if viewpoint == "U.S. Republican":
                    task_ls.extend(["political_opinion_debatune_rep"]*len(questions))
                else:
                    task_ls.extend(["political_opinion_debatune_dem"]*len(questions))
                input_num_ls.extend([i]*len(questions))
                
        for i, input in enumerate(data_political_opinions_topics):
            for format_type in ['implicit', 'explicit']:
                questions, formats = opinion_question(input, question_format=format_type, input_type='topic', viewpoint=viewpoint)
                input_ls.extend(questions)
                label_ls.extend(["radical_transparency"]*len(questions))
                original_ls.extend([input]*len(questions))
                if format_type == 'explicit':
                    format_ls.extend([format + 10 for format in formats])
                else:
                    format_ls.extend(formats)
                if viewpoint == "U.S. Republican":
                    task_ls.extend(["political_opinion_topics_rep"]*len(questions))
                else:
                    task_ls.extend(["political_opinion_topics_dem"]*len(questions))
                input_num_ls.extend([i]*len(questions))

    # Conspiracy (good faith) = reasonable pluralism 
    print("Conspiracy (good faith): reasonable pluralism")
    for i, input in enumerate(data_political_conspiracy['Topic']):
        questions, formats = conspiracy_good_faith(input)
        input_ls.extend(questions)
        label_ls.extend(["reasonable_pluralism"]*len(questions))
        original_ls.extend([input]*len(questions))
        format_ls.extend(formats)
        task_ls.extend(["consipracy_good"]*len(questions))
        input_num_ls.extend([i]*len(questions))

    # Conspiracy (bad faith) = avoidance/refusal
    print("Conspiracy (bad faith): avoidance/refusal")
    for i, input in enumerate(data_political_conspiracy['Topic']):
        questions, formats = conspiracy_bad_faith(input)
        input_ls.extend(questions)
        label_ls.extend(["avoidance"]*len(questions))
        original_ls.extend([input]*len(questions))
        format_ls.extend(formats)
        task_ls.extend(["consipracy_bad"]*len(questions))
        input_num_ls.extend([i]*len(questions))

    # Universal Rights = no approximation
    print("Universal Rights: no approximation")
    for i, input in enumerate(data_universal_rights['Text']):
        for format_type in ['implicit', 'explicit']:
            questions, formats = universal_right(input, format_type)
            input_ls.extend(questions)
            label_ls.extend(["no_approximation"]*len(questions))
            original_ls.extend([input]*len(questions))
            if format_type == 'explicit':
                format_ls.extend([format + 10 for format in formats])
            else:
                format_ls.extend(formats)        
            task_ls.extend(["universal_rights"]*len(questions))
            input_num_ls.extend([i]*len(questions))


    # Unsafe Input = refusal
    print("Unsafe Political Questions: refusal")
    for i, input in enumerate(data_political_unsafe['prompt']):
        input_ls.append(unsafe_question(input))
        label_ls.append("refusal")
        original_ls.append(input)
        format_ls.append(0)
        task_ls.append("unsafe_questions")
        input_num_ls.append(i)

    full_df = pd.DataFrame({"input":input_ls, "label":label_ls, "task":task_ls, "original_input": original_ls, "format_number":format_ls})

    full_df.to_csv(output_file, encoding='utf-8', index=False)

output_file ="~/Data/test_input.csv"
create_test_prompts(output_file)

