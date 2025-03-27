import os
import json
import random
import dataset_labels
from english_words import english_words_set
from transformers import AutoTokenizer


# creates the baseline data. baseline data is changes added to original file. Corruptions will be added on the baseline data

def replace_wth_randomwords(text, tokenizer, seednum):
    # text = text.split()
    text_tok_len = len(tokenizer(text)["input_ids"]) # number of tokens

    randomwords = []
    english_words = sorted(english_words_set)
    english_words = list(english_words)

    random.seed(seednum)
    random.shuffle(english_words)

    randomwords = " ".join(english_words) 

    if text_tok_len == 1 : text_tok_len += 1
    randomwords_text_ids = tokenizer(randomwords)["input_ids"][:text_tok_len]
    randomwords_text = tokenizer.decode(randomwords_text_ids, skip_special_tokens=True)
    
    return randomwords_text

def replace_wth_frequentwords(text, tokenizer, seednum):
    '''replace a text with randomly selected words from top 2000 high frequent words (from wikipedia) list
    '''

    text_tok_len = len(tokenizer(text)["input_ids"]) # number of tokens

    frequentwords = []
    all_frequentwords = open("data_generator/data_for_corruptions/frequentwords.txt","r")
    raw_lines = all_frequentwords.readlines()
   
    for i in range(2000):    # create a sentence with 500 high frequent words 
        frequentwords.append(raw_lines[i].strip("\n"))
    random.seed(seednum)
    random.shuffle(frequentwords)
    frequent_text = " ".join(frequentwords) 

    frequentwords_text_ids = tokenizer(frequent_text)["input_ids"][:text_tok_len]
    frequentwords_text = tokenizer.decode(frequentwords_text_ids, skip_special_tokens=True)
    return frequentwords_text

def replace_50per_wth_frequentwords(text, tokenizer, seednum):
    '''replace 50% of the text with randomly selected words from top 2000 high frequent words (from wikipedia) list
    '''
    text_tok_len = len(tokenizer(text)["input_ids"]) # number of tokens
    half_text_tok_len = int(text_tok_len/2)

    frequentwords = []
    all_frequentwords = open("data_generator/data_for_corruptions/frequentwords.txt","r")
    raw_lines = all_frequentwords.readlines()
   
    for i in range(2000):    # create a sentence with 2000 high frequent words 
        frequentwords.append(raw_lines[i].strip("\n"))
    random.seed(seednum)
    random.shuffle(frequentwords)
    frequentwords_text = " ".join(frequentwords)

    random_index = sorted(random.sample(range(0, text_tok_len), half_text_tok_len))
   
    actual_text_ids = tokenizer(text)["input_ids"]
    freq_words_ids = tokenizer(frequentwords_text)["input_ids"]
    i=0
    for idx in random_index:
        actual_text_ids[idx] = freq_words_ids[i]
        i += 1
    half_frequentwords_text = tokenizer.decode(actual_text_ids, skip_special_tokens=True)
    return half_frequentwords_text

def add_frequentwords(text, tokenizer, seednum, percent):
    '''add frequent words to the text with randomly selected words from top 2000 high frequent words (from wikipedia) list
    '''
    text_tok_len = len(tokenizer(text)["input_ids"]) # number of tokens
    len_to_add = int(text_tok_len * percent)
    total_len = text_tok_len + len_to_add

    combined_ids = [None] * total_len

    random_index = sorted(random.sample(range(0, total_len), text_tok_len)) # to insert actual text ids

    for i in range(text_tok_len):
        combined_ids[random_index[i]] = tokenizer(text)["input_ids"][i]

    frequentwords = []
    all_frequentwords = open("data_generator/data_for_corruptions/frequentwords.txt","r")
    raw_lines = all_frequentwords.readlines()
   
    for i in range(2000):    # create a sentence with 2000 high frequent words 
        frequentwords.append(raw_lines[i].strip("\n"))
    random.seed(seednum)
    random.shuffle(frequentwords)
    frequentwords_text = " ".join(frequentwords)

    freq_words_ids = tokenizer(frequentwords_text)["input_ids"]

    j = 0  
    for i in range(total_len):
        if combined_ids[i] == None:
            combined_ids[i] = freq_words_ids[j]
            j += 1

    added_frequentwords_text = tokenizer.decode(combined_ids, skip_special_tokens=True)
    return added_frequentwords_text

def main():
      
    # Directory containing the JSON files
    directory = "data_generator/processed_tasks"
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", return_tensors="pt")

    # Modify each JSON file in the directory
    for filename in os.listdir(directory):

        if filename.endswith("instr_randomwords.json"):
            
            ################ repeated_text_corruption ##################
            # remove inline intsr one by one - randomwords case
            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                inline_intr_random_words = baseline_data["inline-instruction"]
                
                for i in range(1, 5):
                    for example_type in ["Positive Examples"]:
                        j=i
                        for example in baseline_data[example_type][::-1]:  # Reverse the list to remove from the last
                            example["input"] = example["input"].replace(inline_intr_random_words, "").strip()
                            j -= 1
                            if j == 0:
                                break
                    
                    baseline_data["corruption_id"] = "repeated_text_corruption_ranomwords_inline_in_" + str(4-i) + "_demos"
                    filename_revised = filename.split(".")[0] # without .json
                    filename_revised = filename_revised.replace("_instr_randomwords", "") # without instr_randomwords
                    # Write the data back to the file
                    with open("data_generator/processed_tasks/{}_{}_in_{}_demo.json".format(filename_revised, "randomwords_inline_instr", 4-i), "w") as f:
                        json.dump(baseline_data, f)
            

        if filename.endswith("baseline.json"):
        
            ############### OPEN BASELINE FILE AND ADD CORRUPTIONS AND DUMP INTO JSON FILE FOR EACH CORRUPTION AND FOR EACH TASK #############

            ############### input_randomwords ##################
            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

            baseline_data["corruption_id"] = "input_randomwords"  # with inline instruction
    
            baseline_data["Definition"][0] =  baseline_data["Definition"][0] 
            inline_intr = baseline_data["inline-instruction"]
            # inline_intr_randomwords = replace_wth_randomwords(baseline_data["inline-instruction"], tokenizer, 256)
            
            seed_num = 42
            for example in baseline_data["Positive Examples"]:
                print("----------------------------------------")
                example["input"] = example["input"].replace(inline_intr, '') # remove inline instruction
                example["input"] = replace_wth_randomwords(example["input"], tokenizer, seed_num) # replace input with randomw ords
                example["input"] = example["input"] + '. ' + inline_intr # add relevant inline instruction back
                seed_num += 50
            
            for example in baseline_data["Negative Examples"]:
                example["input"] = example["input"].replace(inline_intr, '') # remove inline instruction
                example["input"] = replace_wth_randomwords(example["input"], tokenizer, seed_num) # replace input with randomw ords
                example["input"] = example["input"] + ' ' + inline_intr # add relevant inline instruction back
                seed_num += 50
            
            for example in baseline_data["Instances"]:
                example["input"] = example["input"]
                
            filename_revised = filename.split(".")[0] # without .json
            filename_revised = filename_revised.replace("_baseline", "") # without baseline
            # Write the data back to the file
            with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "input_randomwords"), "w") as f:
                json.dump(baseline_data, f)

            
            ############### instr_randomwords ##################
            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

            baseline_data["corruption_id"] = "instr_randomwords"  # with inline instruction
    
            baseline_data["Definition"][0] =  replace_wth_randomwords(baseline_data["Definition"][0], tokenizer, 42) # replace definition with random words
            inline_intr = baseline_data["inline-instruction"]
            inline_intr_randomwords = replace_wth_randomwords(baseline_data["inline-instruction"], tokenizer, 256)
            baseline_data["inline-instruction"] = inline_intr_randomwords

            for example in baseline_data["Positive Examples"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_randomwords)
            
            for example in baseline_data["Negative Examples"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_randomwords)
            
            for example in baseline_data["Instances"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_randomwords)
                
            filename_revised = filename.split(".")[0] # without .json
            filename_revised = filename_revised.replace("_baseline", "") # without baseline
            # Write the data back to the file
            with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_randomwords"), "w") as f:
                json.dump(baseline_data, f)


             ############### instr_frequentwords ##################

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

            baseline_data["corruption_id"] = "instr_frequentwords"  # with inline instruction
            baseline_data["Definition"][0] =  replace_wth_frequentwords(baseline_data["Definition"][0], tokenizer, 42) # replace definition with frequent words
            inline_intr = baseline_data["inline-instruction"]
            inline_intr_frequentwords = replace_wth_frequentwords(baseline_data["inline-instruction"], tokenizer, 256)

            for example in baseline_data["Positive Examples"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
            
            for example in baseline_data["Negative Examples"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
            
            for example in baseline_data["Instances"]:
                example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
            filename_revised = filename.split(".")[0] # without .json
            filename_revised = filename_revised.replace("_baseline", "") # without baseline
            # Write the data back to the file
            with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_frequentwords"), "w") as f:
                json.dump(baseline_data, f)

            
            ############### wrong_labels_labelspace ################## works only if the dataset has a labelspace

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline

                if filename_revised in dataset_labels.LABELS: # if labelspace exists for a dataset

                    baseline_data["corruption_id"] = "wrong_labels_labelspace"  # with inline instruction
                    
                    for example in baseline_data["Positive Examples"]:
                        labelspace = dataset_labels.LABELS[filename_revised].copy()
                        true_label = example["output"]
                        labelspace.remove(true_label)
                        example["output"] = random.choice(labelspace)
                    
                    for example in baseline_data["Negative Examples"]:
                        labelspace = dataset_labels.LABELS[filename_revised].copy()
                        true_label = example["output"]
                        labelspace.remove(true_label)
                        example["output"] = random.choice(labelspace)
                    
                    # Write the data back to the file
                    with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "wrong_labels_labelspace"), "w") as f:
                        json.dump(baseline_data, f)

            ############### wrong_labels_labelspace_halfcases ################## works only if the dataset has a labelspace

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
            
                if filename_revised in dataset_labels.LABELS: # if labelspace exists for a dataset

                    baseline_data["corruption_id"] = "wrong_labels_labelspace_halfcases"  # with inline instruction
                    
                    count = 0
                    for example in baseline_data["Positive Examples"]:
                        if count%2==0:
                            labelspace = dataset_labels.LABELS[filename_revised].copy()
                            true_label = example["output"]
                            labelspace.remove(true_label)
                            example["output"] = random.choice(labelspace)
                        count += 1

                    count = 0
                    for example in baseline_data["Negative Examples"]:
                        if count%2==0:
                            labelspace = dataset_labels.LABELS[filename_revised].copy()
                            true_label = example["output"]
                            labelspace.remove(true_label)
                            example["output"] = random.choice(labelspace)
                        count += 1
                        
                    # Write the data back to the file
                    with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "wrong_labels_labelspace_halfcases"), "w") as f:
                        json.dump(baseline_data, f)

            
            ############### labels_randomwords ################## 

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                baseline_data["corruption_id"] = "labels_randomwords"
                start_idx, end_idx = 0, 20 # assuming target len wont be greater than 20

                seednum = 1
                for example in baseline_data["Positive Examples"]:
                    example["output"] = replace_wth_randomwords(example["output"], tokenizer, seednum)
                    start_idx = end_idx
                    end_idx += 10
                    seednum += 1
                
                seednum=200
                for example in baseline_data["Negative Examples"]:
                    example["output"] = replace_wth_randomwords(example["output"], tokenizer, seednum)
                    start_idx = end_idx
                    end_idx += 10
                    seednum += 1

                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "labels_randomwords"), "w") as f:
                    json.dump(baseline_data, f)


            ############### input_ood ################## 

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                baseline_data["corruption_id"] = "input_ood"
            
                ood_corpus = open("data_generator/data_for_corruptions/corpus.txt","r")
                raw_lines = ood_corpus.readlines()
                ood_sent = []
                for i in range(len(raw_lines)):
                    ood_input = raw_lines[i].strip("\n")
                    ood_sent.append(ood_input)

                index = 0
                for example in baseline_data["Positive Examples"]:
                    example["input"] = ood_sent[index] + ". " + baseline_data["inline-instruction"]
                    index += 1
                  
                for example in baseline_data["Negative Examples"]:
                    example["input"] = ood_sent[index] +  ". " + baseline_data["inline-instruction"]
                    index += 1
                    
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "input_ood"), "w") as f:
                    json.dump(baseline_data, f)

            ############### no_inline_instr ################## 

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                baseline_data["corruption_id"] = "no_inline_instr"
               
                for example in baseline_data["Positive Examples"]:
                    example["input"] = example["input"].replace(baseline_data["inline-instruction"],"").strip()

                for example in baseline_data["Negative Examples"]:
                    example["input"] = example["input"].replace(baseline_data["inline-instruction"],"").strip()
                
                for example in baseline_data["Instances"]:
                    example["input"] = example["input"].replace(baseline_data["inline-instruction"],"").strip()
                    
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "no_inline_instr"), "w") as f:
                    json.dump(baseline_data, f)
            
            
            ############### instr_50per_replaced_frequentwords ##################

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                baseline_data["corruption_id"] = "instr_50per_replaced_frequentwords"  # with inline instruction
                baseline_data["Definition"][0] =  replace_50per_wth_frequentwords(baseline_data["Definition"][0], tokenizer, 42) # replace 50% definition with frequent words
                inline_intr = baseline_data["inline-instruction"]
                inline_intr_frequentwords = replace_50per_wth_frequentwords(baseline_data["inline-instruction"], tokenizer, 256)

                for example in baseline_data["Positive Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Negative Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Instances"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                    
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_50per_replaced_frequentwords"), "w") as f:
                    json.dump(baseline_data, f)


            ############### instr_add_50percent_frequqentwords ##################

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

                baseline_data["corruption_id"] = "instr_add_50percent_frequqentwords"  # with inline instruction

                baseline_data["Definition"][0] =  add_frequentwords(baseline_data["Definition"][0], tokenizer, 42, 0.5) # replace 50% definition with frequent words
                inline_intr = baseline_data["inline-instruction"]
                inline_intr_frequentwords = add_frequentwords(baseline_data["inline-instruction"], tokenizer, 256, 0.5)

                for example in baseline_data["Positive Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Negative Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Instances"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                    
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_add_50percent_frequqentwords"), "w") as f:
                    json.dump(baseline_data, f)

                   ############### instr_add_100percent_frequqentwords ##################

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)

                baseline_data["corruption_id"] = "instr_add_100percent_frequqentwords"  # with inline instruction

                baseline_data["Definition"][0] =  add_frequentwords(baseline_data["Definition"][0], tokenizer, 42, 1) # add 100% definition len high frequent words randomly
                inline_intr = baseline_data["inline-instruction"]
                inline_intr_frequentwords = add_frequentwords(baseline_data["inline-instruction"], tokenizer, 256, 1)

                for example in baseline_data["Positive Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Negative Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Instances"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                    
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_add_100percent_frequqentwords"), "w") as f:
                    json.dump(baseline_data, f)
                    
            
            ############### instr_add_at_end_frequentwords ##################
            # add high frequent words randomly shufffled at teh end of instrcution

            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                baseline_data["corruption_id"] = "instr_add_at_end_frequentwords"  # with inline instruction
                def_len = len(baseline_data["Definition"][0])
                def_len_half = int(def_len/2)
                baseline_data["Definition"][0] = baseline_data["Definition"][0] + " " + replace_wth_frequentwords(baseline_data["Definition"][0], tokenizer, 42)[:def_len_half]

                inline_len = len(baseline_data["inline-instruction"])
                inline_len_half = int(inline_len/2)
                inline_intr_frequentwords = baseline_data["inline-instruction"] + " " +  replace_wth_frequentwords(baseline_data["inline-instruction"], tokenizer, 256)[:inline_len_half]

                for example in baseline_data["Positive Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Negative Examples"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                
                for example in baseline_data["Instances"]:
                    example["input"] = example["input"].replace(inline_intr, inline_intr_frequentwords)
                    
                filename_revised = filename.split(".")[0] # without .json
                filename_revised = filename_revised.replace("_baseline", "") # without baseline
                # Write the data back to the file
                with open("data_generator/processed_tasks/{}_{}.json".format(filename_revised, "instr_add_at_end_frequentwords"), "w") as f:
                    json.dump(baseline_data, f)


            ############### repeated_text_corruption ##################
            
            # remove inline intsr one by one
            # Open the baseline JSON file
            with open("data_generator/processed_tasks/{}".format(filename), "r") as f:
                # Load the JSON data
                baseline_data = json.load(f)
                inline_intr = baseline_data["inline-instruction"]
                
                for i in range(1, 5):
                    for example_type in ["Positive Examples"]:
                        j=i
                        for example in baseline_data[example_type][::-1]:  # Reverse the list to remove from the last
                            example["input"] = example["input"].replace(inline_intr, "").strip()
                            j -= 1
                            if j == 0:
                                break
                    
                    baseline_data["corruption_id"] = "repeated_text_corruption_inline_in_" + str(4-i) + "_demos"
                    filename_revised = filename.split(".")[0] # without .json
                    filename_revised = filename_revised.replace("_baseline", "") # without baseline
                    # Write the data back to the file
                    with open("data_generator/processed_tasks/{}_{}_in_{}_demo.json".format(filename_revised, "inline_instr", 4-i), "w") as f:
                        json.dump(baseline_data, f)
   
if __name__ == "__main__":
    main()