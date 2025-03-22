import re

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes the input text by:
    - Removing punctuation marks (', ", ., ?, :, !).
    - Stripping leading and trailing whitespace.
    - Replacing multiple spaces with a single space.
    - Converting all characters to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned and normalized text.
    """
    text = re.sub(r"['\",\.\?:\!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    return text.lower()

# def split_chill(t: str) -> list:
#     return re.split(r'(?<=\.)\s*', t)
 
# def split_sentence(paragraph: str) -> list:
#     context_list = sent_tokenize(paragraph)
#     updated_list = []
#     for context in context_list:
#         updated_list.extend(split_chill(context) if '. ' in context else [context])
#     return updated_list

def split_sentence(paragraph: str) -> list:
    """
    Splits a paragraph into sentences based on specific rules:
    - Recognizes sentence endings based on periods (".") followed by spaces.
    - Handles cases where a period is followed by a newline and an uppercase letter.
    - Ensures sentences have more than two words before adding them to the list.

    Args:
        paragraph (str): The input paragraph.

    Returns:
        list: A list of cleaned and preprocessed sentences.
    """
    sentences = []
    
    if paragraph.endswith("\n\n"):
        paragraph = paragraph[:-2]

    paragraph = paragraph.rstrip()
    start = 0
    paragraph_length = len(paragraph)

    while start < paragraph_length:
        sentence = ""
        initial_start = start

        for i in range(start, paragraph_length):
            if paragraph[i] == ".":
                if i + 2 < paragraph_length and paragraph[i + 1] == "\n":
                    if paragraph[i + 2].isalpha() and paragraph[i + 2].isupper():
                        break
                
                if i + 1 < paragraph_length and paragraph[i + 1] == " ":
                    sentence += paragraph[i]
                    start = i + 1
                    break

            sentence += paragraph[i]

            if i == paragraph_length - 1:
                start = paragraph_length
                break

        if start == paragraph_length:
            sentence += paragraph[start:]

        cleaned_sentence = preprocess_text(sentence.strip())
        if len(cleaned_sentence.split()) > 2:
            sentences.append(cleaned_sentence)

        if start == initial_start:
            print("Warning: No progress detected. Exiting loop.")
            break

    return sentences

def process_data(text: str) -> str:
    """
    Processes a paragraph by splitting it into sentences and returning them as a formatted string.

    Args:
        text (str): The input paragraph.

    Returns:
        str: A string where sentences are joined by ". ".
    """
    return ". ".join(split_sentence(text))


def load_data(data):
    data_old = {}

    for i in data.index:
        if data.id[i] not in data_old.keys():
            data_old[data.id[i]] = [
                {
                    'id': data.id[i],
                    'context': data.context[i],
                    'claim': data.claim[i]
                }
            ]
        else:
            data_old[data.id[i]].append(
                    {
                        'id': data.id[i],
                        'context': data.context[i],
                        'claim': data.claim[i]
                    }
                )
    return data_old
