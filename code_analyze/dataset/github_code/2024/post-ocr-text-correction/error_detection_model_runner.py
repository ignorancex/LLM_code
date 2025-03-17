import torch
import numpy as np
import nltk.data

from tqdm import tqdm
from error_detection_models import ErrorDetectionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from config import ConfigManager
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DistilBertTokenizer, get_linear_schedule_with_warmup

from dataset_parser import DatasetParser
import os

class ErrorDetectorRunner:

    def __init__(self, config, is_train=False):
        self.config = config
        self.bert_tokenizer_directory = config.get_error_detection_tokenizer()
        self.bert_model_path = config.get_error_detection_model()
        self.max_norm_levenshtein_distance = 0.5
        self.max_sequence_length = 100
        self.batch_size = 32
        self.punctuation = "!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~"
        self._is_train = is_train

        self.dataset_parser = DatasetParser(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Available device: {}".format(self.device))
        print("Tokenizer directory is {}".format(self.bert_tokenizer_directory))
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = ErrorDetectionModel()

        # Load the model if not in train mode
        if not self._is_train:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_directory)
            checkpoint = torch.load(self.bert_model_path + '/best_model', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.to(self.device)

        nltk.download('punkt')
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


    def train(self):
        print(self._is_train)
        if not self._is_train:
            raise Exception("Sorry, you are not in train mode. Please set train=True when creating the runner.")
        
        print(self.config.get_train_dataset())
        train_words, train_aligned_words, train_gs_words, train_labels = self.dataset_parser.read_train_dataset()

        train_inputs, train_labels, train_masks, _ = self.dataset_parser.tokenize_dataset(train_words[:100], train_labels[:100], self.max_sequence_length, self.tokenizer)

        train_inputs, validation_inputs, train_masks,\
            validation_masks, train_labels, validation_labels = train_test_split(train_inputs, train_masks, train_labels, test_size=0.1, random_state=42)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, pin_memory=True, num_workers=2, sampler=train_sampler, batch_size=self.batch_size)

        val_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, pin_memory=True, num_workers=2, sampler=val_sampler, batch_size=self.batch_size)

        eval_every = 1

        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if "bias" in n],
                'weight_decay_rate': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if "bias" not in n],
                'weight_decay_rate': 0.0
            }
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        epochs = 10
        max_grad_norm = 1.0
        best_f1 = 0.0

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_values = []
        training_acc_values = []
        f1_score_values = []

        all_predictions = []
        all_true_labels = []
        all_masks = []

        for epoch in range(epochs):
            print("Running epoch: {}".format(epoch))
            self.model.train()
            total_loss = 0
            true_labels = []
            masks = []
            logits_list = []

            for step, batch in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()

                logits = self.model(b_input_ids, attention_mask=b_input_mask)
                loss_fct = CrossEntropyLoss()
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.model.num_labels)
                active_labels = torch.where(
                    active_loss, b_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(b_labels)
                )
                loss = loss_fct(active_logits, active_labels)
                
                logits = logits.detach()
                logits_list.append(logits)
                true_labels.extend(b_labels)
                masks.extend(b_input_mask)

                loss.backward()

                total_loss += loss.item()
                clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

                self.optimizer.step()
                scheduler.step()

            
            predictions = [list(p) for logits in logits_list for p in np.argmax(logits.to("cpu").numpy(), axis=2)]
            true_labels = [tl.to("cpu").numpy() for tl in true_labels]
            b_input_mask_list = [tl.to("cpu").numpy() for tl in masks]
            
            avg_train_loss = total_loss / len(train_dataloader)
            print("Training loss: {}".format(avg_train_loss))

            pred_tags = [p_i for p, l, a in zip(predictions, true_labels, b_input_mask_list) for p_i, l_i, a_i in zip(p, l, a) if a_i]
            valid_tags = [l_i for l, a in zip(true_labels, b_input_mask_list) for l_i, a_i in zip(l, a) if a_i]

            acc_train = accuracy_score(pred_tags, valid_tags)
            prec_train = precision_score(pred_tags, valid_tags)
            rec_train = recall_score(pred_tags, valid_tags)
            f1_score_train = f1_score(pred_tags, valid_tags)
            print("Training Accuracy: {}".format(acc_train))
            print("Training Precision: {}".format(prec_train))
            print("Training Recall: {}".format(rec_train))
            print("F1 score: {}".format(f1_score_train))

            training_acc_values.append(acc_train)
            loss_values.append(avg_train_loss)
            f1_score_values.append(f1_score_train)

            all_predictions.append(predictions)
            all_true_labels.append(true_labels)
            all_masks.append(b_input_mask_list)

            if epoch % eval_every == 0:
                curr_f1 = self.evaluate(val_dataloader)
                if curr_f1 > best_f1:
                    print("Saving new model with F1 score {}".format(curr_f1))
                    self.save()
                    best_f1 = curr_f1

    def evaluate(self, loader=None):
        is_test = loader is None
        eval_type = "Test" if is_test else "Validation"

        if is_test:
            test_words, test_aligned_wods, test_gs_words, test_labels_word = self.dataset_parser.read_test_dataset()
            test_inputs, test_labels, test_masks, tokenized_text = self.dataset_parser.tokenize_dataset(test_words, test_labels_word, self.max_sequence_length, self.tokenizer)

            test_data = TensorDataset(test_inputs, test_labels, test_masks)
            test_sampler = SequentialSampler(test_data)
            loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

        self.model.eval()
        total_loss = 0

        predictions = []
        true_labels = []
        true_masks = []

        for batch in tqdm(loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_labels, input_mask = batch

            with torch.no_grad():
                logits = self.model(input_ids, input_mask)

            label_ids = input_labels.to('cpu').numpy()

            loss_function = CrossEntropyLoss()
            active_loss = input_mask.view(-1) == 1
            active_logits = logits.view(-1, self.model.num_labels)
            active_labels = torch.where(active_loss, input_labels.view(-1), torch.tensor(loss_function.ignore_index).type_as(input_labels))
            loss = loss_function(active_logits, active_labels).item()

            total_loss += loss
            logits = logits.detach().cpu().numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
            true_masks.extend(input_mask.to('cpu').numpy())

        total_loss = total_loss / len(loader)
        print("{} loss: {}".format(eval_type, total_loss))

        pred_labels_new = []
        true_labels_new = []

        for pred_label, true_label, input_mask in zip(predictions, true_labels, true_masks):
            for y_pred, y_true, t_mask in zip(pred_label, true_label, input_mask):
                if t_mask:
                    pred_labels_new.append(y_pred)
                    true_labels_new.append(y_true)

        accuracy_res = accuracy_score(pred_labels_new, true_labels_new)
        precision_res = precision_score(pred_labels_new, true_labels_new)
        recall_res = recall_score(pred_labels_new, true_labels_new)
        f1_score_res = f1_score(pred_labels_new, true_labels_new)

        print("{} accuracy is: {}".format(eval_type, accuracy_res))
        print("{} precision is: {}".format(eval_type, precision_res))
        print("{} recall is: {}".format(eval_type, recall_res))
        print("{} F1 score is: {}".format(eval_type, f1_score_res))

        return f1_score_res

    def save(self):
        tokenizer_output_directory =  self.bert_tokenizer_directory
        model_output_directory =  self.bert_model_path

        if not os.path.exists(model_output_directory):
            os.makedirs(model_output_directory)

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            },
            model_output_directory + '/best_model'
        )

        self.tokenizer.save_pretrained(tokenizer_output_directory)

    def inference(self, test_sentence):
        # print(test_sentence)
        tokenized_sentence = self.tokenizer.encode(test_sentence)[:100]

        #print(self.tokenizer.decode(tokenized_sentence))
        attention_mask = np.zeros(100)
        attention_mask[:len(tokenized_sentence)] = 1
        tokenized_sentence += [0] * (100 - len(tokenized_sentence))
        input_ids = torch.tensor([tokenized_sentence]).cpu()
        attention_mask = torch.tensor([attention_mask]).cpu()

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        
        label_indices = np.argmax(logits.to('cpu').numpy(), axis=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

        classified_tokens, classified_labels = [], []
        dash = False
        for token, label_idx in zip(tokens, label_indices[0]):
            if token == '[PAD]' or token == '[SEP]' or token == '[CLS]' or token in self.punctuation:
                continue

            if token.startswith("##"):
                classified_tokens[-1] = classified_tokens[-1] + token[2:]
                if not classified_labels[-1] and label_idx:
                    classified_labels[-1] = label_idx
            elif token == '-' and len(classified_tokens) > 0:
                classified_tokens[-1] = classified_tokens[-1] + token
                dash = True              
            else:
                if dash and len(classified_tokens) > 0:
                    classified_tokens[-1] = classified_tokens[-1] + token
                    if not classified_labels[-1] and label_idx:
                        classified_labels[-1] = label_idx

                    dash = False
                else:
                    classified_labels.append(label_idx)
                    classified_tokens.append(token)

        return classified_tokens, classified_labels

def main():
    config = ConfigManager()
    error_detector = ErrorDetectorRunner(config, is_train=False)
    # error_detector.train()
    error_detector.evaluate()
    # error_detector.save()

if __name__ == '__main__':
    main()