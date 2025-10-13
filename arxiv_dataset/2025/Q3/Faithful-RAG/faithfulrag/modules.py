from typing import List, Tuple, Dict, Union, Optional
import sys
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from datasets import load_dataset,Dataset
import json
import string
import os
import transformers
from collections import Counter
from collections import defaultdict
import logging
from datetime import datetime
from .prompts import PromptGenerator
import torch
from rouge_score import rouge_scorer
import inflect
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from .util import FormatConverter
from .llm import LLMBackend
from omegaconf import DictConfig
from util import logger


class FactMiningModule:
    """Fact mining module with support for multiple LLM backends"""
    
    def __init__(
        self, 
        backend_type: str, 
        model_name: str,
        **backend_config
    ):
        """
        Initialize the fact mining module
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Initialize LLM backend
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        # Initialize prompt generators
        self.prompt_generator = PromptGenerator(
            llm_type=backend_type,
            task="normal"
        )
        self.prompt_generator_extract = PromptGenerator(
            llm_type=backend_type,
            task="extract"
        )
        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Fact extraction pattern
        self.fact_pattern = r'\d+\.\s([^\d]+(?:\s+[^\d]+)*)'
    
    async def generate_knowledges(
        self, 
        dataset: Dataset, 
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Generate factual knowledge for each item in the dataset
        
        Args:
            dataset: Input dataset
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing facts for each item
        """
        # Generate prompts for knowledge extraction
        prompts = [
            self.prompt_generator_extract.generate_factual_knowledge(
                user_query=item['question']
            )
            for item in dataset
        ]
        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        
        # Parse and return facts
        return [
            {
                'id': item['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]
    
    async def generate_self_context(
        self,
        dataset: Dataset,
        knowledges: Optional[Union[Dict, List[Dict]]] = None,
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Generate self-context for each item in the dataset
        
        Args:
            dataset: Input dataset
            knowledges: Optional factual knowledge to incorporate
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing context for each item
        """
        # Generate prompts for context generation
        prompts = []
        for item in dataset:
            if knowledges is None:
                prompt = self.prompt_generator.generate_context_directly_prompt(
                    user_query=item['question']
                )
            else:
                # Find matching knowledge for this item
                knowledge = next(
                    (k for k in knowledges if k['id'] == item['id']), 
                    None
                )
                if knowledge is None:
                    logger.warning(f"No knowledge found for item {item['id']}")
                    prompt = self.prompt_generator.generate_context_directly_prompt(
                        user_query=item['question']
                    )
                else:
                    prompt = self.prompt_generator.generate_context_by_factual_knowledge(
                        user_query=item['question'],
                        factual_knowledge=knowledge['facts']
                    )
            prompts.append(prompt)
        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating self-contexts...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        
        # Return contexts
        return [
            {'id': item['id'], 'context': result}
            for item, result in zip(dataset, raw_results)
        ]
    
    async def extract_facts(
        self,
        contexts: Union[List[Dict], Dict],
        **sampling_kwargs
    ) -> List[Dict]:
        """
        Extract facts from given contexts
        
        Args:
            contexts: List of contexts or single context dictionary
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            List of dictionaries containing extracted facts
        """
        # Normalize input to list
        if isinstance(contexts, dict):
            contexts = [contexts]
        
        # Generate prompts for fact extraction
        prompts = [
            self.prompt_generator_extract.generate_context_extract(
                user_context=ctx['context']
            )
            for ctx in contexts
        ]
        
        # Generate responses using LLM backend
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Extracting facts...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_extract.system_prompt,
            **merged_params
        )
        
        # Parse and return facts
        return [
            {
                'id': ctx['id'],
                'facts': re.findall(self.fact_pattern, result)
            }
            for ctx, result in zip(contexts, raw_results)
        ]

class ContextualAlignmentModule:
    def __init__(self,
                 similarity_model:str):
        self.similarity_model = SentenceTransformer(similarity_model)

    def chunk_text(self,paragraph: str, chunk_size: int = 20) -> List[str]:
        sentences = nltk.sent_tokenize(paragraph) 
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def calculate_similarity(
        self,
        paragraph: str,
        str_list: List[str],
        top_k: int = 5,
        chunk_size: int = 50
    ):
        chunks = self.chunk_text(paragraph, chunk_size=chunk_size)

        chunk_embeddings = self.similarity_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        str_list_embeddings = self.similarity_model.encode(str_list, convert_to_tensor=True, show_progress_bar=False)

        results = []

        for i, string in enumerate(str_list):
            cosine_scores = util.cos_sim(str_list_embeddings[i], chunk_embeddings)
            
            top_results = sorted(
                enumerate(cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True
            )[:top_k]

            top_chunks = [(chunks[idx], score) for idx, score in top_results]
            results.append((string, top_chunks))

        return results

    def get_topk_contextual_chunks(self, all_chunks: List[Dict], chunk_topk=5):
        all_topk_chunks = []
        for chunk in all_chunks:
            topk_chunks = []
            sorted_chunks = sorted(chunk['chunks'], key=lambda x: x['score'], reverse=True)
            seen_chunks = set()
            # pick unique chunks
            for sub_chunk in sorted_chunks:
                if sub_chunk['chunk'] not in seen_chunks:
                    topk_chunks.append(sub_chunk)
                    seen_chunks.add(sub_chunk['chunk'])
                if len(topk_chunks) == chunk_topk:
                    break
            all_topk_chunks.append({'id': chunk['id'], 'topk_chunks': topk_chunks})
        return all_topk_chunks

    def get_contextual_chunks(self,facts:List[Dict],dataset:Dataset,sent_topk=5,chunck_size=20):
        all_chunks = []
        for item,fact in zip(dataset,facts):
            if len(fact['facts']) == 0:
                print('No facts found')
                continue
            paragraph = FormatConverter.remove_brackets_and_content(item['context'])
            results = self.calculate_similarity(paragraph, fact['facts'], top_k=sent_topk, chunk_size=chunck_size)
            chunks = []
            for _,match in results:
                for chunk,score in match:
                    chunks.append({'chunk':chunk,'score':score})
            all_chunks.append({'id':fact['id'],'chunks':chunks})
        return all_chunks


class SelfThinkModule:
    """Self-thinking module for answer prediction with multiple LLM backends"""
    
    def __init__(
        self, 
        backend_type: str, 
        model_name: str,
        **backend_config
    ):
        """
        Initialize the self-thinking module
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            backend_config: Backend-specific configuration parameters
        """
        # Initialize LLM backend
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        # Initialize prompt generators
        self.prompt_generator_qa_cot = PromptGenerator(
            llm_type=backend_type,
            task="qa-cot"
        )
        self.prompt_generator_qa = PromptGenerator(
            llm_type=backend_type,
            task="qa"
        )
        
        # Default sampling parameters
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
    
    async def predict_answer_normal_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
        """
        Predict answers using normal chain-of-thought reasoning
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge for each item
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Generate prompts
        prompts = []
        for item in dataset:
            # Find matching facts for this item
            fact_str = next(
                (' '.join([chunk['chunk'] for chunk in d['topk_chunks']]) 
                 for d in facts if d['id'] == item['id']), 
                None
            )
            
            prompts.append(
                self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=fact_str
                )
            )
        
        # Generate responses
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating answers with normal cot")
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_qa_cot.system_prompt,
            **merged_params
        )
        
        # Return predictions
        return {item['id']: res for item, res in zip(dataset, results)}
    
    async def predict_answer_scheduled_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
        """
        Predict answers using scheduled chain-of-thought reasoning
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge for each item
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Generate prompts
        prompts = []
        for item in dataset:
            # Find matching facts for this item
            fact_str = next(
                (' '.join([chunk['chunk'] for chunk in d['topk_chunks']]) 
                 for d in facts if d['id'] == item['id']), 
                None
            )
            
            prompts.append(
                self.prompt_generator_qa_cot.generate_qa_prompt_schedule_cot(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=fact_str
                )
            )
        
        # Generate responses
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating answers with scheduled cot...")
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_qa_cot.system_prompt,
            **merged_params
        )
        
        # Return predictions
        return {item['id']: res for item, res in zip(dataset, results)}
    
    async def predict_answer_wo_cot(
        self,
        dataset: Dataset,
        facts: List[Dict],
        **sampling_kwargs
    ) -> Dict[str, str]:
        """
        Predict answers without chain-of-thought reasoning
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge for each item
            sampling_kwargs: Generation parameters to override defaults
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Generate prompts
        prompts = []
        for item in dataset:
            # Find matching facts for this item
            fact_str = next(
                (' '.join([chunk['chunk'] for chunk in d['topk_chunks']]) 
                 for d in facts if d['id'] == item['id']), 
                None
            )
            
            prompts.append(
                self.prompt_generator_qa.generate_qa_prompt(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=fact_str
                )
            )
        
        # Generate responses
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating answers with w/o cot...")
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_qa.system_prompt,
            **merged_params
        )
        
        # Return predictions
        return {item['id']: res for item, res in zip(dataset, results)}
