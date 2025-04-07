
from torch.utils.data import Dataset


class DPOCurationDataset(Dataset):
    def __init__(self, dataset, args, split="train"):
        super(DPOCurationDataset, self).__init__()
        self.args = args
        self.dataset = dataset
        self.split = split

        self.instructions, self.responses, self.rationales1, self.rationales2, self.refined_rationales1, self.refined_rationale2, self.response_types = self.load_dataset()

    def load_dataset(self):
        return self.dataset["instructions"], self.dataset["responses"], self.dataset["turn1_rationale1"], \
            self.dataset["turn1_rationale2"], self.dataset["turn2_rationale1"], self.dataset["turn2_rationale2"], self.dataset["response_types"]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        instruction = self.instructions[index]
        response = self.responses[index]
        rationale1 = self.rationales1[index]
        rationale2 = self.rationales2[index]
        refined_rationale1 = self.refined_rationales1[index]
        refined_rationale2 = self.refined_rationale2[index]
        response_type = self.response_types[index]

        return instruction, response, rationale1, rationale2, refined_rationale1, refined_rationale2, response_type

    def _collate_fn(self, items):
        batch = {
            "instructions": [x[0] for x in items],
            "responses": [x[1] for x in items],
            "rationales1": [x[2] for x in items],
            "rationales2": [x[3] for x in items],
            "refined_rationales1": [x[4] for x in items],
            "refined_rationales2": [x[5] for x in items],
            "response_types": [x[6] for x in items]
        }
        return batch