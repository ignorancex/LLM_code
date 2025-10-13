# TODO: Use fire to parse arguments

import json
from typing import Self
from bespokelabs import curator
from pydantic import BaseModel, Field
from datasets import Dataset
import random

class InternalCoTMathChatSync(BaseModel):
    thought: str = Field(description="The internal reasoning for the assistant's response")

class InternalCoTMathChatSyncCurator(curator.LLM):
    response_format = InternalCoTMathChatSync
    
    def __init__(self, system_prompt_path: str, user_prompt_path: str, **kwargs_super):
        super().__init__(**kwargs_super)
        with open(system_prompt_path, 'r') as f:
            self.system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            self.user_prompt = f.read()
    
    def prompt(self, row):
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_prompt.format(
                    conversation=row["conversations"][:-1],
                    latest_assistant_message=row["conversations"][-1]
                )
            }
        ]
        
    def parse(self, row, response):
        return {
            "thought": response.thought
        }
        
class PrepareDataMathChatSync():
    def __init__(self, file_path):
        self.depth_map = {}
        
        with open(file_path,"r") as f:
            self.data = json.load(f)
            
    def prune_depth_six(self) -> Self:
        """
        There are 100443 conversations with depth 6.
        This is much more than any other depth.
        We reduce the number of depth 6 conversations to 30000.
        """
        
        indices_to_remove = [idx for idx, conversation in enumerate(self.data) if int(len(conversation["conversations"])/2) == 6][30000:]

        for idx in indices_to_remove[::-1]:
            self.data.pop(idx)
        
        return self
        
    def get_size(self) -> int:
        return len(self.data)
    
    def get_sampling_count(self, depth: int, num_samples: int) -> int:
        """
        Returns the number of samples to sample for a given depth.
        """
        
        return max(min(200, len(self.depth_map[depth])), int((len(self.depth_map[depth]) / self.get_size())*num_samples))
            
    def fill_depth_map(self) -> None:
        """
        Depth map is that mapping of depth to the indices of conversations with that depth.
        This function fills the depth map for all depths.
        """
        
        # Initialize depth map
        self.depth_map = {}
        
        # Fill depth map
        for idx, conversation in enumerate(self.data):
            if int(len(conversation["conversations"])/2) not in self.depth_map:
                self.depth_map[int(len(conversation["conversations"])/2)] = []
            self.depth_map[int(len(conversation["conversations"])/2)].append(idx)
    
    def random_sample(self, num_samples: int) -> Self:
        # Set seed for reproducibility
        random.seed(42)
        random.shuffle(self.data)
        
        # Ensure that the depth map is filled
        self.fill_depth_map()
        
        indices_to_remove = []
        
        # Sample the data
        for depth, indices in self.depth_map.items():
            sampling_count = self.get_sampling_count(depth, num_samples)
            indices_to_remove.extend(indices[sampling_count:])
        
        # Remove the indices
        for idx in sorted(indices_to_remove, reverse=True):
            self.data.pop(idx)
            
        return self

    def get_data(self) -> Dataset:
        augmented_data = []
        for conversation in self.data:
            for idx, utterance in enumerate(conversation["conversations"]):
                if utterance["from"] == "gpt":
                    augmented_data.append({
                        "conversations": conversation["conversations"][:idx+1],
                    })
        return Dataset.from_list(augmented_data)
    
    def set_data(self, thoughts: Dataset) -> None:
        next_thought_idx = 0
        for conversation in self.data:
            for idx in range(1, len(conversation["conversations"]), 2):
                conversation["conversations"][idx]["reasoning"] = thoughts[next_thought_idx]["thought"]
                next_thought_idx += 1
    
def main():
    # Prepare data
    data = PrepareDataMathChatSync("./math_chat_sync.json").prune_depth_six().random_sample(8000)
    
    # Prepare curator
    curator = InternalCoTMathChatSyncCurator(
        system_prompt_path="./prompts/internal-reasoning-generator-math-chat-sync-system.txt",
        user_prompt_path="./prompts/internal-reasoning-generator-math-chat-sync-user.txt",
        model_name="gpt-4.1-mini-2025-04-14",
        backend="openai",
        generation_params={
            "temperature": 0.1,
        }
    )
    
    # Generate reasoning
    thoughts = curator(data.get_data())
    
    with open('thoughts.json', 'w') as f:
        json.dump(thoughts.to_list(), f)
    
    # Set data
    data.set_data(thoughts)
        
    # Save data
    with open("math_chat_sync_reasoning.json", "w") as f:
        json.dump(data.data, f, ensure_ascii=False)
        
if __name__ == "__main__":
    main()