
import os
import logging
from argparse import ArgumentParser
import jsonlines
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from blueprint_filters import BlueprintQAFilter
from tqdm import tqdm
import torch

from utils import get_summaries

logger = logging.getLogger(__name__)

class QAPlanGenModel:
    def __init__(self, cuda_device, qa_model_dir, qa_batch_size, 
                 qg_batch_size, num_context_sentences_rb, num_context_sentences_lb, use_round_trip_filter, 
                 use_rheme_filter, use_coverage_filter) -> None:
        
        self.qg_tokenizer = AutoTokenizer.from_pretrained('Salesforce/mixqg-large')
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/mixqg-large").to(cuda_device)
        self.cuda_device = cuda_device
        self.qg_batch_size = qg_batch_size
        self.num_context_sentences_rb = num_context_sentences_rb
        self.num_context_sentences_lb = num_context_sentences_lb
        
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        self.use_round_trip_filter = use_round_trip_filter
        self.use_rheme_filter = use_rheme_filter
        self.use_coverage_filter = use_coverage_filter

        if use_rheme_filter or use_coverage_filter or use_round_trip_filter:
            self.blueprint_qa_filter = BlueprintQAFilter(qa_model_dir, cuda_device, qa_batch_size)
        else:
            self.blueprint_qa_filter = None

    def get_question_context(self, summary_sents, idx):
        if self.num_context_sentences_lb != -1:
            lb = max(0, idx - self.num_context_sentences_lb)
        else:
            lb = 0
        
        if self.num_context_sentences_rb != -1:
            rb = min(len(summary_sents) - 1, idx + self.num_context_sentences_rb)
        else:
            rb = len(summary_sents) - 1
        return " ".join([summary_sents[i] for i in range(lb, rb+1)])
        

    def get_questions(self, answers, context_sentences):
        assert len(answers) == len(context_sentences), \
            f"Number of answers and context sentences do not match: {len(answers)} vs {len(context_sentences)}."
        
        input_texts = []
        for pp_answers, context in zip(answers, context_sentences):
            input_texts += [f"{ans} \\n {context}" for ans in pp_answers]
        
        if not input_texts:
            # Return empty list, same shape as answers
            return answers

        questions = []
        num_batches = len(input_texts) // self.qg_batch_size
        for batch_idx in range(num_batches+1):
            batch_input_texts = input_texts[batch_idx * self.qg_batch_size: (batch_idx + 1) * self.qg_batch_size]
            if not batch_input_texts:
                break
            
            input_ids = self.qg_tokenizer(batch_input_texts, padding=True, return_tensors="pt")["input_ids"]
            output_ids = self.qg_model.generate(input_ids.to(self.cuda_device), max_length=64, num_beams=1)
            questions += self.qg_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # questions = []
        # start_idx = 0
        # answer_lens = [len(ans) for ans in answers]
        # for ans_len in answer_lens:
        #     questions.append(questions_batch[start_idx: start_idx + ans_len])
        #     start_idx += ans_len

        return questions
    
    def format_qa_plan(self, questions, answers):
        qa_list = []
        for question, answer in zip(questions, answers):
            formatted_qa = f"Q: {question} A: {answer}"
            qa_list.append(formatted_qa)
        return "\n".join(qa_list)
    
    def remove_duplicates(self, questions, answers):
        qa_set = set()
        q_nodup = []
        a_nodup = []
        for question, answer in zip(questions, answers):
            qa_tuple = (question, answer)
            if qa_tuple not in qa_set:
                qa_set.add(qa_tuple)
                q_nodup.append(question)
                a_nodup.append(answer)
        return q_nodup, a_nodup

    def generate_qa_plans(self, summary_sents):
        # Extract entities and get question contexts
        answers = []
        question_contexts = []
        for i, summary_sent in enumerate(summary_sents):
            entities = [ent.text for ent in self.spacy_nlp(summary_sent).ents]
            answers.append(entities)
            qc = self.get_question_context(summary_sents, i)
            question_contexts.append(qc)

        # Generate questions
        questions = self.get_questions(answers, question_contexts)
        answers = [ans for ans_list in answers for ans in ans_list]
        questions, answers = self.remove_duplicates(questions, answers)
        summary = " ".join(summary_sents)

        if self.use_round_trip_filter:
            questions, answers = self.blueprint_qa_filter.round_trip_filter(summary, questions, answers)
        
        if self.use_rheme_filter:
            rh_questions, rh_answers, _ = self.blueprint_qa_filter.rheme_filter(summary, questions, answers)
            if rh_questions: # Check if there are any questions left
                questions, answers = rh_questions, rh_answers
        
        if self.use_coverage_filter:
            questions, answers = self.blueprint_qa_filter.coverage_filter(summary, questions, answers)
        
        # Format QA plans and return
        formatted_plan = self.format_qa_plan(questions, answers)
        return formatted_plan


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--qa_model_dir", type=str, default=None)
    parser.add_argument("--qa_batch_size", type=int, default=32)
    parser.add_argument("--qg_batch_size", type=int, default=16)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--num_context_sentences_lb", type=int, default=-1)
    parser.add_argument("--num_context_sentences_rb", type=int, default=-1)
    parser.add_argument("--use_round_trip_filter", action='store_true')
    parser.add_argument("--use_rheme_filter", action='store_true')
    parser.add_argument("--use_coverage_filter", action='store_true')
    parser.add_argument("--save_every", type=int, default=500)
    # parser.add_argument("--checkpoint_path", type=str, default="")
    args = parser.parse_args()

    qa_plan_gen_model = QAPlanGenModel(
        cuda_device=args.cuda_device,
        qa_model_dir=args.qa_model_dir,
        qa_batch_size=args.qa_batch_size,
        qg_batch_size=args.qg_batch_size,
        num_context_sentences_lb=args.num_context_sentences_lb,
        num_context_sentences_rb=args.num_context_sentences_rb,
        use_round_trip_filter=args.use_round_trip_filter,
        use_rheme_filter=args.use_rheme_filter,
        use_coverage_filter=args.use_coverage_filter,
    ) 
    
    # num_completed_plans = 0
    # partial_fine_plans = []
    # if args.checkpoint_path:
    #     with jsonlines.open(args.checkpoint_path, 'r') as f:
    #         for l in f:
    #             partial_fine_plans.append(l)
    #     num_completed_plans = len(partial_fine_plans)
    for split in ['train', 'validation', 'test']:
        logger.info(f"Processing {args.dataset} {split} split.")

        # Each item contains a list of four summaries split into sentences
        summaries_sents = get_summaries(args.dataset, split)
        blueprint_plans = []
        for i, summary_sents in tqdm(enumerate(summaries_sents), total=len(summaries_sents)):
            # if i < num_completed_plans:
            #     plan_dict['fine_grained_plan'] = partial_fine_plans[i]['fine_grained_plan']
            # else:
            ent_qa_plan = qa_plan_gen_model.generate_qa_plans(summary_sents)
            blueprint_plans.append({'idx': i, 
                                    'blueprint_qa_plan': ent_qa_plan})
            
            # if i != 0 and i % args.save_every == 0 and i > num_completed_plans:
            #     write_path = os.path.join(args.output_dir,
            #                               f"{args.dataset}_{args.split}_lb_{args.num_context_sentences_lb}_{i}_qa_plans.jsonl")
            #     with jsonlines.open(write_path, 'w') as f:
            #         for plan_dict in plans_and_summaries[:i+1]:
            #             f.write(plan_dict)
            #     logger.info(f'Saved checkpoint with {i} plans to {args.output_dir}.')
            torch.cuda.empty_cache()
        
        write_path = os.path.join(args.output_dir, 
                                f"{args.dataset}_{split}_lb_{args.num_context_sentences_lb}_bpqa_plans.jsonl")
        with jsonlines.open(write_path, 'w') as f:
            for plan_dict in blueprint_plans:
                f.write(plan_dict)
    
    logger.info(f'Saved QA plans to {args.output_dir}.')
