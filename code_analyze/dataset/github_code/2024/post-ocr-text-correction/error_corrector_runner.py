from data_utils import load_clada, TextProcessor
from error_corrector_seq2seq import Seq2SeqErrorCorrector
from sklearn.model_selection import train_test_split
import torch
import nltk
from config import ConfigManager
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from correction_utils import CorrectionUtil
from sklearn.model_selection import train_test_split

class ErrorCorrectionRunner:

    def __init__(self, config, enable_diagonal_attention_loss, enable_coverage, enable_copy, suffix='') -> None:
        self.config = config
        self.enable_diagonal_attention_loss = enable_diagonal_attention_loss
        self.enable_coverage = enable_coverage
        self.enable_copy = enable_copy

        self.correction_util = CorrectionUtil(self.config)

        self.input, self.target, self.features = self.correction_util.read_train_correction_data()

        self.encoder_text_processor = TextProcessor(self.features.max_encoder_sequence_length, self.features.num_characters, self.features.char_to_id)
        self.decoder_text_processor = TextProcessor(self.features.max_decoder_sequence_length, self.features.num_characters, self.features.char_to_id)

        self.suffix = suffix
        self.units = 256
        self.diag_loss = 3

        self.clada_set = load_clada(self.config.get_clada_dictionary())
        
        nltk.download('punkt')
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs, batch_size):
        # self.input = ['кгато', 'хриету', 'с’съкрушение', 'поусмихпа', 'руенякъ', 'чвек', 'с’съкрушение', 'поусмихпа', 'руенякъ', 'хриету', 'с’съкрушение', 'млеко', 'руенякъ']
        # self.target = ['когато', '\tхристу\n', '\tсъкрушение\n', '\tпоусмихна\n', '\tруснякъ\n', '\човек\n', '\tсъкрушение\n', '\tпоусмихна\n', '\tруснякъ\n', '\tхристу\n', '\tсъкрушение\n', '\мляко\n', '\tруснякъ\n']

        inputs_transformed = self.encoder_text_processor.to_ids(self.input[:15])
        targets_transformed = self.decoder_text_processor.to_ids(self.target[:15])

        self.corrector = Seq2SeqErrorCorrector(
                                number_tokens = self.features.num_characters, 
                                encoder_units = self.units, 
                                encoder_text_processor = self.encoder_text_processor,
                                decoder_text_processor = self.decoder_text_processor,
                                decoder_units = self.units * 2,
                                max_encoder_seq_length = self.features.max_encoder_sequence_length,
                                max_decoder_seq_length = self.features.max_decoder_sequence_length,
                                enable_diagonal_attention_loss = self.enable_diagonal_attention_loss,
                                enable_coverage = self.enable_coverage, 
                                enable_copy = self.enable_copy,
                                start_character = self.correction_util.start_character,
                                pad_character = self.correction_util.pad_character,
                                end_character = self.correction_util.end_character,
                                use_beam=False,
                                config = self.config,
                                device = self.device,
                                diag_loss_length = self.diag_loss)
        
        print(self.encoder_text_processor.char_to_id)
        print(self.decoder_text_processor.char_to_id)

        input_text_torch = torch.from_numpy(inputs_transformed)#.to(device)
        target_text_torch= torch.from_numpy(targets_transformed)#.to(device)

        input_text_x_train, input_text_x_val, input_text_y_train, input_text_y_val = train_test_split(input_text_torch, target_text_torch, test_size=0.1)

        print(input_text_x_train.shape)
        print(input_text_x_val.shape)
        print(input_text_y_train.shape)
        print(input_text_y_val.shape)

        train_data = TensorDataset(input_text_x_train, input_text_y_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader_simple_test = DataLoader(train_data, pin_memory=True, num_workers=2, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(input_text_x_val, input_text_y_val)
        val_sampler = RandomSampler(val_data)
        val_dataloader_simple_test = DataLoader(val_data, pin_memory=True, num_workers=2, sampler=val_sampler, batch_size=batch_size)

        print("Encoder tokens {}".format(self.corrector.number_tokens))
        print("Decoder tokens {}".format(self.corrector.number_tokens))

        losses = self.corrector.train(train_dataloader_simple_test, val_dataloader_simple_test, epochs=epochs, eval_every = 10)

        return losses

    def save(self):
        print("Saving the model to {}".format(self.config.get_error_corrector_model()))

        self.corrector.save_model(self.config.get_error_corrector_model(), self.suffix)

    def load(self):
        self.corrector = Seq2SeqErrorCorrector(
                                number_tokens = self.features.num_characters, 
                                encoder_units = self.units, 
                                encoder_text_processor = self.encoder_text_processor,
                                decoder_text_processor = self.decoder_text_processor,
                                decoder_units = self.units * 2,
                                max_encoder_seq_length = self.features.max_encoder_sequence_length,
                                max_decoder_seq_length = self.features.max_decoder_sequence_length,
                                enable_diagonal_attention_loss = self.enable_diagonal_attention_loss,
                                enable_coverage = self.enable_coverage, 
                                enable_copy = self.enable_copy,
                                start_character = self.correction_util.start_character,
                                pad_character = self.correction_util.pad_character,
                                end_character = self.correction_util.end_character,
                                use_beam=False,
                                config = self.config,
                                device = self.device, 
                                diag_loss_length = self.diag_loss)
        
        print("Loading model from {}".format(self.config.get_error_corrector_model() + self.suffix))
        self.corrector.load_model(self.config.get_error_corrector_model(), self.suffix)
    
def main():
    config = ConfigManager()
    error_corrector = ErrorCorrectionRunner(config=config,
        enable_diagonal_attention_loss=False,
        enable_coverage=False,
        enable_copy=False)
    
    # error_corrector.train()
    # error_corrector.test()
    # error_corrector.save()
    error_corrector.load()
    print(error_corrector.corrector.predict_text(['облекло'])) # облѣкло 

if __name__ == '__main__':
    main()
