import pandas as pd
from transformers import BertTokenizer,AutoTokenizer
import json
import re

# Function to substitute words based on dictionary
def substitute_words(sentence, cues):
    words = sentence.split()

    filepath = "./data/affixal/dictionary.txt"
    df = pd.read_csv(filepath, delimiter='\t')

    # Extract the first two columns
    keys = df.iloc[:, 1]
    values = df.iloc[:, 0]

    # Create a dictionary
    dict_affixal = dict(zip(keys, values))


    for cue in cues.split(' | '):
        word, index = cue.split(' [')
        index = int(index[:-1])  # Convert to zero-based index
        if index < len(words):
            # Apply specific rules for affixes
            if word.startswith('un') and len(word) > 2:
                words[index] = word[2:]  # Remove 'un' prefix
            elif word.endswith('less') and len(word) >= 4:
                words[index] = word[:-4] + 'ful'  # Replace 'less' with 'ful'
            elif word in dict_affixal:
                words[index] = dict_affixal[word]  # Replace using dictionary
            else:
                words = ""

    return ' '.join(words)

def load_affixal():

    # Specify the path to your Excel file
    file_path = "./data/affixal/sst.xlsx"

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Filter affixal negations
    filtered_df = df[df['negation_type'] == 'AFFIX']

    # Apply substitution to each row based on cues column
    filtered_df['text_substituted'] = filtered_df.apply(lambda row: substitute_words(row['text'], row['cues']), axis=1)

    # Remove rows where 'text_substituted' is empty
    filtered_df = filtered_df[filtered_df['text_substituted'].str.strip() != '']

    # Optionally reset index if needed
    filtered_df.reset_index(drop=True, inplace=True)

    # Save the filtered DataFrame as a pickle file
    pickle_path = './data/affixal/filtered_df.pkl'
    filtered_df.to_pickle(pickle_path)

def load_verbal():
    
    # Load JSON data from file
    filepath = "./data/annotated/wiki_negated.json"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create DataFrame
    #df = pd.DataFrame(
    # {"Original_Sentence": original_sentences, "Negated_Sentence": negated_sentences})

    # Define a function to detokenize the tokens back into sentences
    def detokenize(tokens):
        return tokenizer.convert_tokens_to_string(tokens)

    json_data = []
    with open(filepath, "r") as file:
        for line in file:
            try:
                json_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print("Error decoding JSON on line:", line)
                print("Error message:", str(e))

    # Detokenize and create DataFrame
    original_sentences = []
    negated_sentences = []
    for item in json_data:
        tokens = item["tokens"]
        sentence = detokenize(tokens).replace("[CLS]", "").replace("[SEP]", "").strip()
        
        # Split sentence at each period not followed by a digit or a space and a digit
        sentences = re.split(r'\.(?![\d\s]\d)', sentence)
        #sentences = sentence.split(".")
        
        original_sentences.append(sentences[0])
        negated = (sentences[1] if len(sentences) > 1 else "").strip()
        negated_sentences.append(negated.replace("[MASK]",item["masked_lm_labels"][0])) #substitute mask with the right word
        

    # Create DataFrame
    df = pd.DataFrame({"Original_Sentence": original_sentences, "Negated_Sentence": negated_sentences})

    # Save the filtered DataFrame as a pickle file
    pickle_path = './data/verbal/verbal_df.pkl'
    df.to_pickle(pickle_path)



if __name__ == "__main__":

    print("hello")
    load_affixal()
    load_verbal ()





