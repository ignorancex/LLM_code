import json
import re
import os
from datasets import load_dataset, Dataset
import inflect
import string
from vllm import LLM, SamplingParams
import os
import torch
import copy
import math
import logging
from typing import Union,List

class FormatConverter:

    @staticmethod
    def convert_text2fact(text, output_file=None):
        # Use regex to find all matches between <seq> and </seq>
        matches = re.findall(r'<seq>(.*?)</seq>', text, re.DOTALL)
        
        # Create a dictionary with numbered keys
        result = {f"seq_{i+1}": match.strip() for i, match in enumerate(matches)}
        
        # Save to JSON Lines file if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as file:
                for key, value in result.items():
                    json.dump({key: value}, file, ensure_ascii=False)
                    file.write('\n')
        
        return result

    @staticmethod
    def remove_brackets_and_content(input_str: str) -> str:
        cleaned_str = re.sub(r'\<.*?\>', '', input_str)
        cleaned_str = re.sub(r'\[.*?\]', '', cleaned_str)
        cleaned_str = ' '.join(cleaned_str.split())
        return cleaned_str
            
    @staticmethod
    def normalize_answer(s):
        p = inflect.engine()
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace('_', ' ')
        
        def convert_numbers_to_words(text):
            words = text.split()
            result = []
            for word in words:
                if word.isdigit() and int(word) < 100:
                    word_in_words = p.number_to_words(int(word))
                    result.append(word_in_words)
                else:
                    result.append(word)
            return ' '.join(result)

        return white_space_fix(remove_articles(handle_punc(convert_numbers_to_words(lower(replace_underscore(s)))))).strip()

    @staticmethod
    def convert_context(context, chunks):
        for chunk in chunks:
            context = re.sub(re.escape(chunk), f"[important facts: {chunk}]", context)
        return context
    
    @staticmethod
    def extract_answer(prediction):
        try:
            data = json.loads(prediction)
            answer_key = next((key for key in data if "answer" in key.lower()), None)
            if answer_key:
                answer = data[answer_key]
                return answer
            else:
                print("JSON does not contain an 'answer' field. Returning the original prediction.")
                return prediction
        except json.JSONDecodeError:
            print("The string is not a valid JSON format. Returning the original prediction.")
            return prediction
