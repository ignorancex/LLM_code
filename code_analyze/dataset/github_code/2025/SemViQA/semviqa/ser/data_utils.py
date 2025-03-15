import pandas as pd
from datasets import Dataset

def preprocess_training_dataset(examples, tokenizer, config):
    questions = [q.strip() for q in examples["claim"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        if pd.notna(examples["evidence"][i]) and examples["evidence"][i] not in ["", " ","nan","none"]:
    #         answer = answers[i]
            start_char =  examples["context"][i].find(examples["evidence"][i])
            if start_char == -1:
                print("Evidence not found in context")
            end_char = start_char + len(examples["evidence"][i])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            try:
                while sequence_ids[idx] != 1:
                    idx += 1
            except:
                print(questions[i])
                print("+++++")
                print(examples["context"][i])
                print(sequence_ids)
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        else:
            start_positions.append(0)
            end_positions.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["id"] = examples["id"]
    list_tagg_rational = []
    for i in range(len(end_positions)):
        tagging = [0]*len(inputs['input_ids'][i])
        if end_positions[i] != start_positions[i]:
            tagging[start_positions[i]:end_positions[i] + 1] = [1] * (end_positions[i] - start_positions[i] + 1)
        list_tagg_rational.append(tagging) 
    inputs['Tagging'] = list_tagg_rational  
    return inputs


def load_data(config, tokenizer):   
    train_dataset = None
    test_dataset = None
    
    if config.is_train:
        train = pd.read_csv(config.train_data)

        train.id = range(len(train))
        train = train.astype(str)

        train = Dataset.from_dict({
            "context": train.context,
            "claim": train['claim'],
            "verdict": train.verdict,
            "evidence": train.evidence,
            "id": train.id,
        })

        train_dataset = train.map(
            lambda examples: preprocess_training_dataset(examples, tokenizer,config),
            batched=True,
            remove_columns=train.column_names,
        )
        
    if config.is_eval:
        test = pd.read_csv(config.eval_data)

        test.id = range(len(test))
        test = test.astype(str)

        test = Dataset.from_dict({
            "context": test.context,
            "claim": test.claim,
            "verdict": test.verdict,
            "evidence": test.evidence,
            "id": test.id,
        })

        test_dataset = test.map(
            lambda examples: preprocess_training_dataset(examples, tokenizer,config),
            batched=True,
            remove_columns=test.column_names,
        )
        
    if config.is_train and config.is_eval:
        return train_dataset, test_dataset
    elif config.is_train:
        return train_dataset
    elif config.is_eval:
        return test_dataset
    else:
        return None