import asyncio
from typing import Dict, List, Optional
from datasets import Dataset
from tqdm import tqdm
from .evaluate import (
    exact_match_score, 
    acc_score, 
    f1_score, 
    metric_max_over_ground_truths
)
from .modules import (
    FactMiningModule,
    ContextualAlignmentModule,
    SelfThinkModule
)
from .util import FormatConverter

class FaithfulRAG:
    """Faithful Retrieval-Augmented Generation pipeline"""
    
    def __init__(
        self,
        backend_type: str,
        model_name: str,
        similarity_model: str,
        mining_sampling_params: Optional[Dict] = None,
        generation_sampling_params: Optional[Dict] = None,
        **backend_config
    ):
        """
        Initialize the FaithfulRAG pipeline
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            similarity_model: Sentence similarity model name
            mining_sampling_params: Parameters for fact mining generation
            generation_sampling_params: Parameters for answer generation
            backend_config: Backend-specific configuration parameters
        """
        self.backend_type = backend_type
        self.model_name = model_name
        self.similarity_model = similarity_model
        
        # Set default sampling parameters if not provided
        self.mining_sampling_params = mining_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        self.generation_sampling_params = generation_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Initialize modules
        self.fact_mining_module = FactMiningModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        self.contextual_alignment_module = ContextualAlignmentModule(
            similarity_model=similarity_model
        )
        
        self.self_think_module = SelfThinkModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
    
    async def get_self_facts(
        self, 
        dataset: Dataset, 
        fact_mining_type: str = "default",
        **mining_params
    ) -> List[Dict]:
        """
        Generate self-consistent facts for the dataset
        
        Args:
            dataset: Input dataset
            fact_mining_type: Type of fact mining ("default")
            mining_params: Override parameters for fact mining
            
        Returns:
            List of self-consistent facts dictionaries
        """
        # Use provided parameters or defaults
        params = {**self.mining_sampling_params, **mining_params}
        
        if fact_mining_type == "default":
            # Generate initial knowledge facts
            knowledges = await self.fact_mining_module.generate_knowledges(
                dataset, **params
            )
            
            # Generate self-context using the knowledge
            self_context = await self.fact_mining_module.generate_self_context(
                dataset, knowledges=knowledges, **params
            )
            
            # Extract self-consistent facts from the context
            return await self.fact_mining_module.extract_facts(
                self_context, **params
            )
        else:
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")
    
    def get_topk_chunks(
        self, 
        dataset: Dataset, 
        self_facts: List[Dict],
        sent_topk: int = 5,
        chunk_topk: int = 5,
        chunk_size: int = 20
    ) -> List[Dict]:
        """
        Retrieve top-k contextual chunks for each fact
        
        Args:
            dataset: Input dataset
            self_facts: Self-consistent facts
            sent_topk: Number of top sentences to retrieve
            chunk_topk: Number of top chunks to return
            chunk_size: Size of context chunks
            
        Returns:
            List of dictionaries with top-k chunks
        """
        # Get contextual chunks
        contextual_chunks = self.contextual_alignment_module.get_contextual_chunks(
            self_facts, dataset, sent_topk, chunk_size
        )
        
        # Get top-k chunks
        return self.contextual_alignment_module.get_topk_contextual_chunks(
            contextual_chunks, chunk_topk
        )
    
    async def get_predictions(
        self,
        dataset: Dataset, 
        facts: List[Dict],
        generation_type: str = "normal_cot",
        **generation_params
    ) -> Dict[str, str]:
        """
        Generate predictions for the dataset
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge with top-k chunks
            generation_type: Type of generation ("normal_cot", "scheduled_cot", "wo_cot")
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Use provided parameters or defaults
        params = {**self.generation_sampling_params, **generation_params}
        
        if generation_type == "normal_cot":
            params['response_format'] = {"type": "json_object"}
            return await self.self_think_module.predict_answer_normal_cot(
                dataset, facts, **params
            )
        elif generation_type == "scheduled_cot":
            params['response_format'] = {"type": "json_object"}
            return await self.self_think_module.predict_answer_scheduled_cot(
                dataset, facts, **params
            )
        elif generation_type == "wo_cot":
            return await self.self_think_module.predict_answer_wo_cot(
                dataset, facts, **params
            )
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")
    
    def evaluate(
        self, 
        dataset: Dataset, 
        predictions: Dict[str, str],
        cot_format: bool = False,
        detailed_output: bool = False
    ) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            dataset: Input dataset with ground truth
            predictions: Generated predictions
            detailed_output: Whether to include per-item details
            
        Returns:
            Evaluation results dictionary
        """
        prediction_details = []
        total_em = total_acc = total_f1 = 0
        num_items = 0
        
        for item in tqdm(dataset, desc="Evaluating"):
            prediction = predictions.get(item['id'], "")
            # if prediction is in JSON format, extract the 'answer' field
            if cot_format:
                prediction = FormatConverter.extract_answer(prediction)
            ground_truth = item['answer']
            
            # Calculate metrics
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
            acc_score_val = metric_max_over_ground_truths(
                acc_score, prediction, ground_truth)
            f1_score_val = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)
            
            # Accumulate totals
            total_em += em_score
            total_acc += acc_score_val
            total_f1 += f1_score_val
            num_items += 1
            
            # Store details if requested
            if detailed_output:
                prediction_details.append({
                    "id": item['id'],
                    "question": item['question'],
                    "answer": ground_truth,
                    "prediction": prediction,
                    "exact_match": em_score,
                    "acc": acc_score_val,
                    "f1": f1_score_val
                })
        
        # Calculate averages
        avg_em = 100.0 * total_em / num_items if num_items > 0 else 0
        avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
        avg_f1 = 100.0 * total_f1 / num_items if num_items > 0 else 0
        
        # Prepare result
        result = {
            "num_items": num_items,
            "exact_match": avg_em,
            "acc": avg_acc,
            "f1": avg_f1
        }
        
        if detailed_output:
            result["details"] = prediction_details
            
        return result