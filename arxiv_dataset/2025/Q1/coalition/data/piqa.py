from torch.utils.data import Dataset

class PIQARationaleGenerationDataset(Dataset):
    def __init__(self, dataset, args, tokenizer, split="train"):
        super(PIQARationaleGenerationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.split = split

        self.instructions, self.rationales, self.gt_rationales, self.responses, self.response_types = self.load_dataset()

    def load_dataset(self):

        instructions, rationales, gt_rationales, responses, response_types = [], [], [], [], []

        prompt = "Select the correct option to complete the goal.\Goal: {goal}\nOption 1: {sol1} Option 2: {sol2}"
        for index in range(len(self.dataset[self.split])):
            data = self.dataset[self.split][index]
            instructions.append(prompt.format(sentence=data["goal"], option1=data["sol1"], option2=data["sol2"]))
            gt_rationales.append("")
            rationales.append("")
            answer = data["sol2"] if int(data["answer"]) == 1 else data["sol2"]
            responses.append(answer)
            response_types.append("rationale")

        return instructions, rationales, gt_rationales, responses, response_types

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

        # Return the instruction, rationale, response and response type
        return instruction, rationale, gt_rationale, response, response_type

    def collate_fn(self, items):
        batch = {
            "instructions": [x[0] for x in items],
            "rationales": [x[1] for x in items],
            "gt_rationales": [x[2] for x in items],
            "responses": [x[3] for x in items],
            "response_types": [x[4] for x in items]
        }
        return batch
    
class PIQARationaleRefinementDataset(Dataset):
    def __init__(self, dataset, args, tokenizer, split="train"):
        super(PIQARationaleRefinementDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.split = split

        self.instructions, self.rationales, self.gt_rationales, self.responses, self.response_types = self.load_dataset()

    def load_dataset(self):

        instructions, rationales, gt_rationales, responses, response_types = [], [], [], [], []
        for index in range(len(self.dataset["instructions"])):
            instructions.append(self.dataset['instructions'][index])
            gt_rationales.append(self.dataset["gt_rationales"][index])
            rationales.append(self.dataset["rationales"][index])
            responses.append(self.dataset['responses'][index])
            response_types.append("rationale_refinement")

        return instructions, rationales, gt_rationales, responses, response_types

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

        # Return the instruction, rationale, response and response type
        return instruction, rationale, gt_rationale, response, response_type

    def collate_fn(self, items):
        batch = {
            "instructions": [x[0] for x in items],
            "rationales": [x[1] for x in items],
            "gt_rationales": [x[2] for x in items],
            "responses": [x[3] for x in items],
            "response_types": [x[4] for x in items]
        }
        return batch