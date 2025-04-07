from torch.utils.data import Dataset


# BaseDataset class
class BaseDataset(Dataset):
    def __init__(self, dataset, args, tokenizer, split):
        super(BaseDataset, self).__init__()
        
        # Set the arguments
        self.dataset = dataset
        self.args = args
        self.tokenizer = tokenizer
        self.split = split

    # Function to return the length of the dataset
    def __len__(self):
        return len(self.instructions)
    
    # Function to get an item from the dataset
    def __getitem__(self, index):
        
        # Get the instruction, rationale, response and response type
        instruction = self.instructions[index]
        rationale = self.rationales[index]
        gt_rationale = self.gt_rationales[index]
        response = self.responses[index]
        response_type = self.response_types[index]

        # Add the end of string token to the rationale and response wherever necessary
        if response_type == "response" or response_type == "rationale_conditioned_response":
            response += self.tokenizer.eos_token
        elif response_type == "rationale" or response_type == "rationale_refinement":
            gt_rationale += self.tokenizer.eos_token

        # Return the instruction, rationale, response and response type
        return instruction, rationale, gt_rationale, response, response_type

    # Function to collate the items in the dataset
    def collate_fn(self, items):
        batch = {
            "instructions": [x[0] for x in items],
            "rationales": [x[1] for x in items],
            "gt_rationales": [x[2] for x in items],
            "responses": [x[3] for x in items],
            "response_types": [x[4] for x in items]
        }
        return batch
        