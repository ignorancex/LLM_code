import torch
import nltk.data

from config import ConfigManager
from tqdm import tqdm
from error_detection_model_runner import ErrorDetectorRunner
from error_corrector_runner import ErrorCorrectionRunner
from dataset_parser import DatasetParser
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class ErrorCorrectionEvaluator:

    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Available device: {}".format(self.device))

        self.edrunner = ErrorDetectorRunner(self.config)
        self.tokenizer = self.edrunner.tokenizer
        self.model = self.edrunner.model
        self.dataset_parser = DatasetParser(config)

        nltk.download('punkt')
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.error_corrector = ErrorCorrectionRunner(config=config,
            enable_diagonal_attention_loss=False,
            enable_coverage=False,
            enable_copy=False)

        self.correction_util = self.error_corrector.correction_util

        self.batch_size = 64

        self.error_corrector.load()
        self.features = self.error_corrector.features

    def evaluate(self):
        test_words, _, test_gs_words, _ = self.dataset_parser.read_test_dataset()
        wrong = []
        gs = []

        for x, y in tqdm(zip(test_words, test_gs_words)):
            _, l = self.edrunner.inference(x)

            if len(l) != len(y):
                continue

            for tok, g, lb in zip(x, y, l):
                if lb == 1 and tok != '[UNK]':
                    wrong.append(tok)
                    gs.append(g)

        i, t = [], []
        for w, gs in zip(wrong, gs):
            if len(w) >= self.features.max_encoder_sequence_length or len(gs) >= self.features.max_decoder_sequence_length:
                continue

            target = self.correction_util.start_character + gs + self.correction_util.end_character

            i.append(w)
            t.append(target)

        loader = self.create_test_dataloader(i, t)
        self.error_corrector.corrector.eval_loader(loader)

    def create_test_dataloader(self, input_test, target_test):
        inputs_test_transformed = self.error_corrector.encoder_text_processor.to_ids(input_test)
        targets_test_transformed = self.error_corrector.decoder_text_processor.to_ids(target_test)

        input_test_torch = torch.from_numpy(inputs_test_transformed)
        target_test_torch= torch.from_numpy(targets_test_transformed)

        test_loader = TensorDataset(input_test_torch, target_test_torch)
        test_sampler = RandomSampler(test_loader)
        test_dataloader = DataLoader(test_loader, pin_memory=True, num_workers=2, sampler=test_sampler, batch_size=self.batch_size)

        return test_dataloader

def main():
    config = ConfigManager()
    evaluator = ErrorCorrectionEvaluator(config)
    evaluator.evaluate()

if __name__ == '__main__':
    main()