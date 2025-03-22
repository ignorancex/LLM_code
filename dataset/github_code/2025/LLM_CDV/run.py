import os
import argparse
import heapq
import math
import json
import concurrent.futures
from utils.answers import simulate
from utils.question import generate_question
from utils.check import check_semantic
from utils.tools import num_tokens_from_string, logger as utils_logger

logger = utils_logger
FLAG = 0

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap) if self.heap else None

    def __len__(self):
        return len(self.heap)

class Node:
    def __init__(self, model='gpt-4o', question=None, ori_question=None, choices=[], ground_truth=None, depth=1, rate=1):
        self.question = question
        self.model = model
        self.depth = depth
        self.ori_question = ori_question
        self.choices = choices
        self.ground_truth = ground_truth
        length = num_tokens_from_string(question)/num_tokens_from_string(ori_question) if ori_question else 1
        self.value = self.metric(depth=depth, length=length, rate=rate)

    def __lt__(self, other):
        return self.value > other.value

    def metric(self, depth=1, length=1, rate=1, K=2, M=3, N=1):
        try:
            return math.exp(K / rate) * (depth ** -N)
        except Exception as e:
            logger.error(f"Metric calculation error: {e}")
            return 0

def add_child(node, priority_queue, args):
    good_number = 0
    good_questions = []
    logger.info("="*100)
    logger.info(f"Processing question:\n{node.question}\nDepth: {node.depth}\nValue: {node.value}")

    wrong_options = [c for c in node.choices if c != node.ground_truth]
    new_questions = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                generate_question,
                model=node.model,
                question=node.question,
                wrong_option=opt,
                choices=node.choices,
                ground_truth=node.ground_truth
            ): opt for opt in wrong_options
        }

        for future in concurrent.futures.as_completed(future_map):
            try:
                generated = future.result()
                if generated and num_tokens_from_string(generated[0]['question']) < 10*num_tokens_from_string(node.question):
                    new_questions.extend(generated)
            except Exception as e:
                logger.error(f"Generation error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        future_map = {
            executor.submit(
                process_single_question,
                nq, node, args
            ): nq for nq in new_questions
        }

        for future in concurrent.futures.as_completed(future_map):
            try:
                is_good, qdata, child = future.result()
                if is_good:
                    good_number += 1
                    good_questions.append(qdata)
                if child:
                    priority_queue.push(child)
            except Exception as e:
                logger.error(f"Processing error: {e}")

    return good_number, good_questions

def process_single_question(new_question, node, args):
    global FLAG

    # Semantic check for DK mode
    if not check_semantic(
        model=node.model,
        question=new_question['question'],
        ori_question=node.ori_question,
        choices=new_question.get('choices'),
        ground_truth=node.ground_truth
    ):
        logger.warning(f"Semantic check failed:\n{new_question['question']}")
        return False, None, None

    # Simulation
    result = simulate(
        question=node.question,
        model=node.model,
        choices=node.choices,
        ground_truth=node.ground_truth,
        simulate_times=args.simulate_times
    )

    logger.info(f"Simulation result: {result} | Ground truth: {node.ground_truth}")
    
    if result <= 0.0:
        qdata = {
            'ori_question': node.ori_question,
            "question": new_question['question'],
            "choices": new_question.get('choices', []),
            "ground_truth": node.ground_truth,
            "rate": result
        }
        return True, qdata, None

    child = Node(
        model=node.model,
        question=new_question['question'],
        ori_question=node.ori_question,
        choices=new_question.get('choices', node.choices),
        ground_truth=node.ground_truth,
        depth=node.depth + 1,
        rate=result
    )
    return False, None, child

def process_entry(entry, args):
    global FLAG
    priority_queue = PriorityQueue()
    good_questions = []

    root = Node(
        model=args.model,
        question=entry["question"],
        ori_question=entry["question"],
        choices=entry.get("choices", []),
        ground_truth=entry.get("answer")
    )
    priority_queue.push(root)

    max_try, tries_remaining, add_tag = args.max_try, 5, 1
    FLAG = 0

    while len(priority_queue) > 0 and max_try > 0:
        node = priority_queue.pop()
        current_good, current_questions = add_child(node, priority_queue, args)
        good_questions.extend(current_questions)
        max_try -= 1

        if current_good:
            tries_remaining = 5
            add_tag = 1
        else:
            tries_remaining -= 1

        if len(good_questions) >= args.max_good:
            break

    return good_questions

def main():
    parser = argparse.ArgumentParser(description="DK Mode Question Processor")
    parser.add_argument('--input-file', type=str, default='data/MMLU.json')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--concurrency', type=int, default=10)
    parser.add_argument('--max-try', type=int, default=10)
    parser.add_argument('--max-good', type=int, default=3)
    parser.add_argument('--simulate-times', type=int, default=5)
    
    args = parser.parse_args()

    # Load and process data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_path = f'enhanced/{args.model}/{os.path.basename(args.input_file).split(".")[0]}_enhanced.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for entry in data:
        if 'good_questions' in entry:
            continue
            
        origin_rate = simulate(
            question=entry['question'],
            model=args.model,
            choices=entry.get('choices', []),
            ground_truth=entry.get('answer'),
            simulate_times=args.simulate_times
        )
        if origin_rate == 0:
            entry['good_questions'] = [{
                "ori_question": entry['question'],
                "question": entry['question'],
                "choices": entry.get('choices', []),
                "ground_truth": entry.get('answer'),
                "rate": 0
            }]
            logger.info(f"Origin question is not answerable:\n{entry['question']}")
        else:
            entry['good_questions'] = process_entry(entry, args)
            logger.info(f"Good questions generated: {len(entry['good_questions'])}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()