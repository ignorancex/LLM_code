from .base import BaseDataset
from torch.utils.data import Dataset

# COTDataset class
class CoTDataset(BaseDataset):
    def __init__(self, dataset, args, tokenizer, split):
        super(CoTDataset, self).__init__(dataset, args, tokenizer, split)

        self.dataset = dataset

        # Load the dataset
        self.instructions, self.rationales, self.gt_rationales, self.responses, self.response_types = self.load_dataset()
    
    # Function to load the dataset
    def load_dataset(self):
        instructions, rationales, gt_rationales, responses, response_types = [], [], [], [], []
        for index in range(len(self.dataset["instructions"])):
            instructions.append(self.dataset['instructions'][index])
            gt_rationales.append(self.dataset['gt_rationales'][index])
            rationales.append(self.dataset['rationales'][index])
            responses.append(self.dataset['responses'][index])
            response_types.append(self.dataset['response_types'][index])

        return instructions, rationales, gt_rationales, responses, response_types



class CoTRationaleGenerationDataset(Dataset):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

        self.instructions, self.rationales, self.gt_rationales, self.responses, self.response_types = self.load_dataset()

    def load_dataset(self):
        instructions, rationales, gt_rationales, responses, response_types = [], [], [], [], []
        for index in range(len(self.dataset)):
            data = self.dataset[index]
            instructions.append(data['source'])
            gt_rationales.append(data['rationale'])
            rationales.append("")
            responses.append(data['target'])
            response_types.append("rationale_refinement")

        return instructions, rationales, gt_rationales, responses, response_types

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.instructions[index], self.rationales[index], self.gt_rationales[index], self.responses[index], self.response_types[index]

    def collate_fn(self, items):
        batch = {
            "instructions": [x[0] for x in items],
            "rationales": [x[1] for x in items],
            "gt_rationales": [x[2] for x in items],
            "responses": [x[3] for x in items],
            "response_types": [x[4] for x in items]
        }
        return batch