import random
import string
from typing import Dict, List, Any
from PIL import Image

from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

class RandomAgent(BaseAgent):
    """
    RandomAgent is a reference implementation that demonstrates the expected interface
    for the CRAG-MM benchmark.
    
    This agent returns random strings as responses and serves as a baseline for understanding
    the agent interface. In a real implementation, you would replace this with a model that
    generates meaningful responses based on the query, images, and conversation history.
    
    The agent interface is designed to work with the CRAG-MM evaluation framework,
    which evaluates agents on both single-turn and multi-turn conversation tasks.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
    """
    
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        """
        Initialize the RandomAgent.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        """
        super().__init__(search_pipeline)
        print("Initializing RandomAgent - reference implementation")
    
    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        # Fixed batch size of 16 for this reference implementation
        # You may adjust this based on your agent's capabilities
        return 16

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate responses for a batch of queries.
        
        This is the main method called by the evaluator. It processes multiple
        queries in parallel for efficiency. For multi-turn conversations,
        the message_histories parameter contains the conversation so far.
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query. 
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        # For illustration, this random agent just generates random strings
        # A real implementation would:
        # 1. Use self.search_pipeline to retrieve relevant information if needed
        # 2. Process each query along with its corresponding image (if available)
        # 3. Consider the conversation history for multi-turn tasks
        # 4. Generate meaningful responses based on retrieved information
        
        return [
            "".join(
                random.choice(string.ascii_letters + " ")
                for _ in range(random.randint(2, 16))
            )
            for _ in range(len(queries))
        ]
