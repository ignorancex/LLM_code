import ast
import logging
import json
import random
import string
from typing import Optional, Union
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    text_only: bool=False
    placement_based_corruption: bool = False
   
    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        id = []
        task = []
        categories = []
        reasoning = []
        label = []
        garbage_text = "kjghg tgwbg dw eurfter eytfway udwqted uwd."

        for instance in batch:
        
            id.append(instance['id'])
            task.append(instance['Task'])
            categories.append(instance['Categories'])
            reasoning.append(instance['Reasoning'])

            add_task_definition = self.add_task_definition
            num_pos_examples = self.num_pos_examples

            # Creating the prompt
            task_input = ""

            # Test input
            task_input += f"{instance['Instance']['input'].strip()}"

            task_without_period = ['task835_mathdataset_answer_generation', 'task1564_triviaqa_answer_generation' ]

            if not instance['Task'] in task_without_period:
                if not task_input[-1] in string.punctuation:
                    task_input += "."
                
            # Instruction
            definition = ""
            if add_task_definition:
                if self.placement_based_corruption == "end_window_garbage_in_begin":
                    definition = garbage_text + " " + instance["Definition"][0].strip()
                else:
                    definition = instance["Definition"][0].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # Add positive examples.
            pos_examples = []
            pos_examples_num = 0
            for i , pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):

                pos_example_str = ''
                if self.placement_based_corruption != 'input_empty':

                    pos_example_str += pos_example['input'].strip()

                    # if self.placement_based_corruption == "end_window_garbage_in_mid" and pos_examples_num == 1:
        
                    #     # Append garbage text to the second demo
                    #     pos_example_str += pos_example['input'].strip() + " " + garbage_text
            
                    # else:
                    #     # Concatenate input strings
                    #     pos_example_str += pos_example['input'].strip()

                    if not instance['Task'] in task_without_period:
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                  
        
                if self.placement_based_corruption != 'label_empty':
                    pos_example_output = pos_example['output'].strip()

                    pos_example_str += " " # added a space between input and output in demostrations
                    pos_example_str += pos_example_output
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."

                if self.placement_based_corruption == "end_window_garbage_in_mid" and pos_examples_num == 1:
        
                    # Append garbage text to the second demo
                    pos_example_str += " " + garbage_text

                pos_example_str += "\n\n" 

                pos_examples_num += 1

                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # Combine all elements
            if self.placement_based_corruption == 'instr_after_demo':
                source =  "".join(pos_examples) + definition + task_input 
            else:    
                source = definition + "".join(pos_examples) + task_input 

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

            # label
            labels = instance['Instance']['output']  # its a string of list of string
            labels = ast.literal_eval(labels)
            label.append(labels[0])
            

            model_inputs = {
                "id": id,
                "Task": task,
                "Categories": categories,
                "Reasoning": reasoning,
                "inputs": sources,
                "label":label
            }

        return model_inputs
