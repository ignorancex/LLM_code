import sys, os
import numpy as np
from transformers import Seq2SeqTrainer
from model.postprocess import PostProcessFuns

class Trainer(object):
    def __init__(self, model, training_args, model_args, data_args,
                    train_dataset, eval_dataset, test_shards,
                    tokenizer, data_collator, logger):

        self.model_name_or_path = model_args.model_name_or_path
        self.predict_with_generate = training_args.predict_with_generate
        self.output_dir = training_args.output_dir
        self.max_train_samples = data_args.max_train_samples
        self.max_val_samples = data_args.max_val_samples
        self.num_beams = data_args.num_beams
        self.val_max_target_length = data_args.val_max_target_length
        self.max_test_samples = data_args.max_test_samples

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_shards = test_shards
        self.logger = logger

        self.tokenizer = tokenizer
        self.postproces_obj = PostProcessFuns(self.tokenizer, data_args)

        self.trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.postproces_obj.compute_metrics if self.predict_with_generate else None,)

    def train(self, last_checkpoint):
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(self.model_name_or_path):
            checkpoint = self.model_name_or_path
        else:
            checkpoint = None

        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (self.max_train_samples if self.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
    

    def eval(self):
        self.logger.info("*** Evaluate ***")

        metrics = self.trainer.evaluate(
            max_length=self.val_max_target_length, num_beams=self.num_beams, metric_key_prefix="eval")
        max_val_samples = self.max_val_samples if self.max_val_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(self.eval_dataset))

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)


    def predict(self):
        self.logger.info("*** Test ***")
        if self.trainer.is_world_process_zero():
            if self.predict_with_generate:
                output_test_preds_file = os.path.join(self.output_dir, "test_generations.txt")
                writer_pred = open(output_test_preds_file, "w")
                output_test_labels_file = os.path.join(self.output_dir, "test_labels.txt")
                writer_label = open(output_test_labels_file, "w")

        for test_dataset in self.test_shards:
            test_results = self.trainer.predict(
                test_dataset,
                metric_key_prefix="test",
                max_length=self.val_max_target_length,
                num_beams=self.num_beams,)
            metrics = test_results.metrics
            max_test_samples = self.max_test_samples if self.max_test_samples is not None else len(test_dataset)
            metrics["test_samples"] = min(max_test_samples, len(test_dataset))

            self.trainer.log_metrics("test", metrics)
            self.trainer.save_metrics("test", metrics)

            if self.trainer.is_world_process_zero():
                if self.predict_with_generate:
                    test_preds = self.tokenizer.batch_decode(
                            test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    lbs = np.where(test_results.label_ids != -100, test_results.label_ids, self.tokenizer.pad_token_id)
                    test_labels = self.tokenizer.batch_decode(
                            lbs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    test_preds = [pred.strip() for pred in test_preds]
                    test_labels = [label.strip() for label in test_labels]
                    #output_test_preds_file = os.path.join(self.output_dir, "test_generations.txt")
                    #with open(output_test_preds_file, "w") as writer:
                    writer_pred.write("\n".join(test_preds)+"\n")
                    writer_label.write("\n".join(test_labels)+"\n")

