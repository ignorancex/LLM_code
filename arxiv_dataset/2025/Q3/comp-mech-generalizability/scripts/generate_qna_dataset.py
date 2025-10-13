import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Statement to Question Generation
from transformers import T5ForConditionalGeneration, AutoTokenizer 

qa_model_name = "mrm8488/t5-base-finetuned-question-generation-ap" 
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = T5ForConditionalGeneration.from_pretrained(qa_model_name)

qa_model.to("cuda")

# parallel execution using threading
def get_question(answer, context, verbose=False, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = qa_tokenizer([input_text], return_tensors='pt')
    features.to("cuda")
    output = qa_model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)
    
    question = qa_tokenizer.decode(output[0], skip_special_tokens=True)
    if verbose:
        print(input_text, question)

    return question

def process_row(row):
    question = get_question(row["target_new"], row["base_prompt"] + row["target_new"])
    row["question"] = question.split("question: ")[-1]
    return row

def modify_dataset(dataset, subset=None):
    # Use tqdm
    if subset:
        return [process_row(row) for row in tqdm(dataset[:subset])]
    else:
        return [process_row(row) for row in tqdm(dataset)]

def parallel_modify_dataset(dataset, subset=None):
    # Use ThreadPoolExecutor for I/O-bound tasks (or ProcessPoolExecutor for CPU-bound tasks)
    with ThreadPoolExecutor() as executor:
        if subset:
            results = list(tqdm(executor.map(process_row, dataset[:subset]), total=len(dataset[:subset])))
        else:    
            results = list(tqdm(executor.map(process_row, dataset), total=len(dataset)))

    return results

# dataset name
with open("../data/full_data_sampled_pythia-6.9b_with_subjects.json", "r") as f:
    dataset = json.load(f)

qa_dataset = modify_dataset(dataset, subset=None)
qa_dataset

# saving the data
with open("../data/full_data_sampled_pythia-6.9b_with_questions.json", "w") as f:
    json.dump(qa_dataset, f)
