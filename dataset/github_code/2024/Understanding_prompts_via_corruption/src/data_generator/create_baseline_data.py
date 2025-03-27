import os
import json
from transformers import AutoTokenizer


# creates the baseline data. baseline data is changes added to original file. Corruptions will be added on the baseline data

def process_task1344_glue_entailment_classification(data):
    # Add inline instrcutions if exists
    data["inline-instruction"] = "Does Sentence 1 entail Sentence 2?"
    data["Definition"] = ["In this task, you are given two sentences. Answer with 'Yes' if the first sentence entails the second sentence, otherwise answer with 'No'."]

    for example in data["Positive Examples"]:
        input_str = example["input"]
        output_str = "Yes" if example["output"] == "1" else "No"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " No").replace(" 1 ", " Yes ")
    
    for example in data["Negative Examples"]:
        input_str = example["input"]
        output_str = "Yes" if example["output"] == "1" else "No"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace(" 0", " No").replace(" 1 ", " Yes ")
    
    for example in data["Instances"]:
        input_str = example["input"]
        output_str = ["Yes"] if example["output"] == ["1"] else ["No"]
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str  
    return data

def process_task843_financial_phrasebank_classification(data):
    # Add inline instructions if exists
    data["inline-instruction"] = "Is the sentiment of the sentence 'Negative', 'Neutral', or 'Positive'?"
    data["Definition"] = ["Based on the sentiment, classify the given piece of financial news into one of the three classes: positive, negative, and neutral. Output must be 'Positive', 'Negative', or 'Neutral'."]

    for example in data["Positive Examples"]:
        input_str = example["input"].replace(" .",".")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        example["output"] = example["output"].capitalize()
      
    for example in data["Negative Examples"]:
        input_str = example["input"].replace(" .",".")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        example["output"] = example["output"].capitalize() 

    for example in data["Instances"]:
        input_str = example["input"].replace(" .",".")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        example["output"][0] = example["output"][0].capitalize()
    
    return data

def process_task1645_medical_question_pair_dataset_text_classification(data):
    # Add inline instructions if exists
    data["inline-instruction"] = "Are these two questions similar or dissimilar?"
    data["Definition"] = ["In this task you are given a medical question pair. Your task is to classify this question pair into two categories 1) 'Similar' if the given two questions have the same connotation or meaning  2) 'Dissimilar' if the given two questions have a different connotation or meaning"]
    
    for example in data["Positive Examples"]:
        input_str = example["input"].replace("Sentence1","Question 1")
        input_str = input_str.replace("\n Sentence2","Question 2")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        
    for example in data["Negative Examples"]:
        input_str = example["input"].replace("Sentence1","Question 1")
        input_str = input_str.replace("\n Sentence2","Question 2")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])

    for example in data["Instances"]:
        input_str = example["input"].replace("Sentence1","Question 1")
        input_str = input_str.replace("\n Sentence2","Question 2")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
    
    return data

def process_task116_com2sense_commonsense_reasoning(data):
    data["inline-instruction"] = "Does this statement make sense to you?"
    data["Definition"] = ["You will be given a piece of text either about an everyday event, or a general statement. If the event seems a plausible event, or the general statement makes sense to you then answer the question as 'Yes', otherwise 'No'."]
   
    for example in data["Positive Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        output_str = "Yes" if example["output"] == "True" else "No"
        example['output'] = output_str

      
    for example in data["Negative Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        if example["output"] == "Correct":
            example["output"] = "True"
        output_str = "Yes" if example["output"] == "True" else "No"
        example['output'] = output_str

    for example in data["Instances"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        output_str = ["Yes"] if example["output"] == ["True"] else ["No"]
        example['output'] = output_str
    
    return data

def process_task512_twitter_emotion_classification(data):
    data["inline-instruction"] = "Which emotion is expressed in this tweet?"
    data["Definition"] = ["In this task, you are given a tweet. The task is to classify this tweet based on its emotion. The answer should be one of these emotions 'Sadness', 'Joy', 'Love', 'Anger', 'Fear', or 'Surprise'."]
   
    for example in data["Positive Examples"]:
        example["input"] = example["input"] + '.'
        example["input"] = example["input"].capitalize()
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize()
      
    for example in data["Negative Examples"]:
        example["input"] = example["input"] + '.'
        example["input"] = example["input"].capitalize()
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize()

    for example in data["Instances"]:
        example["input"] = example["input"] + '.'
        example["input"] = example["input"].capitalize()
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"][0] = example["output"][0].capitalize()
    
    return data

def process_task1564_triviaqa_answer_generation(data):
    data["inline-instruction"] = "The answer to this question is "
    data["Definition"] = ["You are given a general knowledge question based on Wikipedia and Web content. Write an answer to this question."]
   
    for example in data["Positive Examples"]:
        input_str = example["input"]
        input_str = input_str.replace("Question:","") 
        input_str = input_str.lstrip(" ")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        example["output"] = example["output"][0].upper() + example["output"][1:] # cap the first letter
      
    for example in data["Negative Examples"]:
        input_str = example["input"]
        input_str = input_str.replace("Question:","") 
        input_str = input_str.lstrip(" ")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        example["output"] = example["output"][0].upper() + example["output"][1:] # cap the first letter

    for example in data["Instances"]:
        input_str = example["input"]
        input_str = input_str.replace("Question:","") 
        input_str = input_str.lstrip(" ")
        example["input"] = "{} {}".format(input_str, data['inline-instruction'])
        if len(example["output"][0]) > 1:
            example["output"][0] = example["output"][0][0].upper() + example["output"][0][1:] # cap the first letter
        else:
            example["output"][0] = example["output"][0].upper()
    return data

def process_task835_mathdataset_answer_generation(data):
    data["inline-instruction"] = "The answer to this math problem is "

    for example in data["Positive Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
      
    for example in data["Negative Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])

    for example in data["Instances"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
    
    return data

def process_task284_imdb_classification(data):
    
    data["inline-instruction"] = "Is this review positive or negative?"
    data["Definition"] = ["In this task, you are given a movie review. Based on its content, your task is to classify the given review into two categories: Positive or Negative."]

    for example in data["Positive Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize()
      
    for example in data["Negative Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize() 

    for example in data["Instances"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"][0] = example["output"][0].capitalize()
    
    return data

def process_task379_agnews_topic_classification(data):
    data["inline-instruction"] = "What label best describes this news article?"
    data["Definition"] = ["In this task, you are given a news article. Your task is to classify the article to one out of the four topics 'World', 'Sports', 'Business', 'Sci/Tech'. If you are not sure about the topic, choose the closest option. Note that URLs in the text have been replaced with [Link]."]

    for example in data["Positive Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
      
    for example in data["Negative Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])

    for example in data["Instances"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
    
    return data

def process_task828_copa_commonsense_cause_effect(data):
    data["inline-instruction"] = "Is the second sentence cause or effect of the first sentence?"
    data["Definition"] = ["In this task your given two statements. You must judge whether the second sentence is the cause or effect of the first sentence.  The two sentences are separated by a newline character and the answer can be 'Cause' or 'Effect'"]

    for example in data["Positive Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize()
      
    for example in data["Negative Examples"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"] = example["output"].capitalize()

    for example in data["Instances"]:
        example["input"] = "{} {}".format(example["input"], data['inline-instruction'])
        example["output"][0] = example["output"][0].capitalize()
    
    return data

def process_task1346_glue_cola_grammatical_correctness_classification(data):
    data["inline-instruction"] = "Is this sentence meaningful and grammatically correct?"
    data["Definition"] = ["You will be given a sentence. If the sentence is grammatically correct and meaningful, then answer with 'Yes', otherwise 'No'."]

    for example in data["Positive Examples"]:
        input_str = example["input"]
        output_str = "Yes" if example["output"] == "1" else "No"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace("0", " No").replace("1", "Yes ")
    
    for example in data["Negative Examples"]:
        input_str = example["input"]
        output_str = "Yes" if example["output"] == "1" else "No"
        
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str
        example['explanation'] = example['explanation'].replace("0", " No").replace("1", " Yes ")
    
    for example in data["Instances"]:
        input_str = example["input"]
        output_str = ["Yes"] if example["output"] == ["1"] else ["No"]
        example['input'] = "{} {}".format(input_str, data['inline-instruction'])
        example['output'] = output_str  

    return data

def main():
    # Directory containing the JSON files
    directory = "data_generator/reordered_original_tasks"

    # Modify each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Open the JSON file
            with open("data_generator/reordered_original_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                original_data = json.load(f)

            ############### BASELINE DATA ##################
            original_data["corruption_id"] = "baseline".split()  # with inline instruction
            
            ############## creating task1344_glue_entailment_classification_baseline.json  ###########
            if filename == "task1344_glue_entailment_classification.json":
                processed_task = process_task1344_glue_entailment_classification(original_data)

            ############# creating task843_financial_phrasebank_classification_baseline.json  ###########
            if filename == "task843_financial_phrasebank_classification.json":
                processed_task = process_task843_financial_phrasebank_classification(original_data)

            ############# creating task1645_medical_question_pair_dataset_text_classification.json  ###########
            if filename == "task1645_medical_question_pair_dataset_text_classification.json":
                processed_task = process_task1645_medical_question_pair_dataset_text_classification(original_data)
            
            ############# creating task116_com2sense_commonsense_reasoning.json  ###########
            if filename == "task116_com2sense_commonsense_reasoning.json":
                processed_task = process_task116_com2sense_commonsense_reasoning(original_data)

            ############# creating task512_twitter_emotion_classification.json  ###########
            if filename == "task512_twitter_emotion_classification.json":
                processed_task = process_task512_twitter_emotion_classification(original_data)

            ############# creating task1564_triviaqa_answer_generation.json  ###########
            if filename == "task1564_triviaqa_answer_generation.json":
                processed_task = process_task1564_triviaqa_answer_generation(original_data)

            ############# creating task835_mathdataset_answer_generation.json  ###########
            if filename == "task835_mathdataset_answer_generation.json":
                processed_task = process_task835_mathdataset_answer_generation(original_data)       
            
            ############# creating task284_imdb_classification.json  ###########
            if filename == "task284_imdb_classification.json":
                processed_task = process_task284_imdb_classification(original_data)

            ############# creating task379_agnews_topic_classification.json  ###########
            if filename == "task379_agnews_topic_classification.json":
                processed_task = process_task379_agnews_topic_classification(original_data)

            ############# creating task828_copa_commonsense_cause_effect.json  ###########
            if filename == "task828_copa_commonsense_cause_effect.json":
                processed_task = process_task828_copa_commonsense_cause_effect(original_data)

            ############# creating task1346_glue_cola_grammatical_correctness_classification.json  ###########
            if filename == "task1346_glue_cola_grammatical_correctness_classification.json":
                processed_task = process_task1346_glue_cola_grammatical_correctness_classification(original_data)

            # Write the data back to the file
            filename = filename.split('.')[0]
            with open("data_generator/processed_tasks/{}_{}.json".format(filename,"baseline"), "w") as f:
                json.dump(processed_task, f)

if __name__ == "__main__":
    main()