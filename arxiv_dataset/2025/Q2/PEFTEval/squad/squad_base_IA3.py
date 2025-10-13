import torch
import peft
import logging
from torch.utils.data import DataLoader
from transformers import squad_convert_examples_to_features
from transformers import get_linear_schedule_with_warmup
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers import RobertaForQuestionAnswering, RobertaTokenizer, AdamW, AutoTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate


batch_size = 32
model_name_or_path = "./Roberta_base"
model_save_path = "./result_with_loss_2/checkpoint/squad_base_IA3_4e-3/"

logging.basicConfig(filename='./result_with_loss_2/loss/squad_base_IA3_4e-3.log', level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

def evaluate_compute(example_indices, eval_features, outputs, all_results):

    for i, example_index in enumerate(example_indices):
        eval_feature = eval_features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        start_logits = outputs.start_logits[i].cpu().numpy().tolist()
        end_logits = outputs.end_logits[i].cpu().numpy().tolist()
        result = SquadResult(unique_id, start_logits, end_logits)
        all_results.append(result)
    return all_results

def result_compute(eval_examples, eval_features, all_results, tokenizer):
    final_predictions = compute_predictions_logits(
        all_examples=eval_examples,
        all_features=eval_features,
        all_results=all_results,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file=None,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=True,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer
    )
    results = squad_evaluate(eval_examples, final_predictions)
    return results

def features_convert(examples):
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=386,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt"
    )
    return features, dataset

device = torch.device("cuda:7")
tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)

processor = SquadV1Processor()
train_examples = processor.get_train_examples('./SQUAD/squad1.1')
train_features, train_dataset = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=386,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="pt"
)

eval_examples = processor.get_dev_examples('./SQUAD/squad1.1')
eval_features, eval_dataset = features_convert(eval_examples)

as_examples = processor.get_dev_examples('./SQUAD/squad_Adversarial_Testsets/AS')
as_features, as_dataset = features_convert(as_examples)

aa_examples = processor.get_dev_examples('./SQUAD/squad_Adversarial_Testsets/AA')
aa_features, aa_dataset = features_convert(aa_examples)

aae_examples = processor.get_dev_examples('./SQUAD/squad_Adversarial_Testsets/AAE')
aae_features, aae_dataset = features_convert(aae_examples)

aac_examples = processor.get_dev_examples('./SQUAD/squad_Adversarial_Testsets/AAC')
aac_features, aac_dataset = features_convert(aac_examples)

ana_examples = processor.get_dev_examples('./SQUAD/squad_Adversarial_Testsets/ANA')
ana_features, ana_dataset = features_convert(ana_examples)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
as_dataloader = DataLoader(as_dataset, batch_size=batch_size, shuffle=False)
aa_dataloader = DataLoader(aa_dataset, batch_size=batch_size, shuffle=False)
aae_dataloader = DataLoader(aae_dataset, batch_size=batch_size, shuffle=False)
aac_dataloader = DataLoader(aac_dataset, batch_size=batch_size, shuffle=False)
ana_dataloader = DataLoader(ana_dataset, batch_size=batch_size, shuffle=False)

peft_config = peft.IA3Config(task_type="QUESTION_ANS", inference_mode=False)

model = RobertaForQuestionAnswering.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

lr = 4e-3
num_epochs = 50

optimizer = AdamW(model.parameters(), lr=lr)

total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


model.train()
model.to(device)


for epoch in range(num_epochs):
    batch_idx = 0
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        start_positions = batch[3].to(device)
        end_positions = batch[4].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                        end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        batch_idx = batch_idx + 1

    print("epoch " + str(epoch) + " loss: ", loss)
    model.save_pretrained(model_save_path + str(epoch))

    print("**********Epoch " + str(epoch) + " Test************")
    model.eval()
    all_results = []
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, eval_features, outputs, all_results)

        results = result_compute(eval_examples, eval_features, all_results, tokenizer)
        print("Eval Results:", results)

    all_results = []
    with torch.no_grad():
        for batch in as_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, as_features, outputs, all_results)

        results = result_compute(as_examples, as_features, all_results, tokenizer)
        print("AS Results:", results)

    all_results = []
    with torch.no_grad():
        for batch in aa_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, aa_features, outputs, all_results)

        results = result_compute(aa_examples, aa_features, all_results, tokenizer)
        print("AA Results:", results)

    all_results = []
    with torch.no_grad():
        for batch in aae_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, aae_features, outputs, all_results)

        results = result_compute(aae_examples, aae_features, all_results, tokenizer)
        print("AAE Results:", results)

    all_results = []
    with torch.no_grad():
        for batch in aac_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, aac_features, outputs, all_results)

        results = result_compute(aac_examples, aac_features, all_results, tokenizer)
        print("AAC Results:", results)

    all_results = []
    with torch.no_grad():
        for batch in ana_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            example_indices = batch[3]
            all_results = evaluate_compute(example_indices, ana_features, outputs, all_results)

        results = result_compute(ana_examples, ana_features, all_results, tokenizer)
        print("ANA Results:", results)

