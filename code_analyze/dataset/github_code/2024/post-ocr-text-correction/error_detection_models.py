from data_utils import PUNCTUATION_ALL
from torch import nn
from transformers import BertModel, DistilBertModel, XLMRobertaModel, XLMModel, RobertaModel, DebertaV2Model
import fasttext

class ErrorDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2

        ## change below two for the respective pretrained LLMs
        # self.pretrained_model = BertModel.from_pretrained("distilbert-base-multilingual-cased")
        # self.pretrained_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
        # self.pretrained_model = RobertaModel.from_pretrained("iarfmoose/roberta-base-bulgarian")
        # self.pretrained_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        # self.pretrained_model = XLMModel.from_pretrained("xlm-mlm-100-1280")
        self.pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.embedding_dim = 768
        
        self.classifier = nn.Linear(self.embedding_dim, self.num_labels)

    def forward(self, input, attention_mask):
        outputs = self.pretrained_model(input, attention_mask)
        sequence_output = outputs.last_hidden_state # 64,100,768

        return self.classifier(sequence_output)
    

class CladaErrorDetectionModel:
    def __init__(self, config) -> None:
        self.config = config
        self.clada_dictionary = set(line.strip(PUNCTUATION_ALL + '\n').lower() for line in open(self.config.get_clada_dictionary(), encoding='utf-16', mode="r"))
    
    def forward(self, input):
        return input not in self.clada_dictionary # Labels: 0 - correct, 1 - incorrect
    
class FastTextErrorDetectionModel:
    def __init__(self, config) -> None:
        self.config = config
        self.model = None

    def forward(self, input):
        if self.model is None:
            raise Exception("Fast Text Model is none. Need to train it first.")
        
        lbl, p = self.model.predict(input)
        predicted_lbl = int(lbl[0].replace("__label__", ""))

        return predicted_lbl
    
    def train(self, fast_text_file, epoch=1000):
        self.model = fasttext.train_supervised(fast_text_file, epoch=epoch)