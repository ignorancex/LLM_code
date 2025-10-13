from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .response_feedback_agent import ResponseFeedbackAgent
from .summarize_agent import SummarizeAgent
from .revision_agent import RevisionAgent
import logging

class MultiAgents():
    def __init__(self, config):
        self.config = config

        self.model = AutoModelForCausalLM.from_pretrained(config['model']['agent_model_path'], torch_dtype=torch.float16, low_cpu_mem_usage=True,).eval().to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['agent_model_path'], padding_side='left')

        self.response_feedback_agent = ResponseFeedbackAgent(self.config, self.model, self.tokenizer)
        self.summarize_agent = SummarizeAgent(self.config, self.model, self.tokenizer)
        self.revision_agent = RevisionAgent(self.config, self.model, self.tokenizer)

    def attack(self, data, responses, prompt_list):
        
        questions = [d['origin_question'] for d in data]

        assert len(questions) == len(responses)

        qa_list = list(zip(questions, responses))

        multi_angle = self.config['attack_config']['multi_angle']

        for id, angle in enumerate(multi_angle):
            
            qa_feedback_list = self.response_feedback_agent.process(angle, qa_list)

            qa_summarize = self.summarize_agent.process(angle, qa_feedback_list, prompt_list[id])
            steer_prompt = self.revision_agent.process(angle, qa_summarize, prompt_list[id])
    
            prompt_list[id] = steer_prompt

        
        return prompt_list