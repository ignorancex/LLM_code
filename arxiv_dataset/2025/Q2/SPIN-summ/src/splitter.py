import os
import argparse
import json
import math
import numpy as np

from datasets import load_metric
from rouge_score import rouge_scorer

def splitter(n, s):
    pieces = s.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def split_to_part(text, n_parts):
    pieces = text.split()
    len_pieces = len(pieces)
    step = math.ceil(len_pieces / n_parts)
    rest = step * n_parts - len_pieces
    return [" ".join(pieces[i:i+step]) for i in range(0, len_pieces + rest, step)]

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="")
    parser.add_argument("--scenario", type=str, help="")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown = read_args()

    metrics = ['rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    rouge = load_metric("rouge")

    train_data = os.path.join(args.data_root, 'train.json')
    val_data = os.path.join(args.data_root, 'val.json')
    test_data = os.path.join(args.data_root, 'test.json')

    data_prefixes = ['train', 'val', 'test']
    data_paths = [train_data, val_data, test_data]

    task_output_dir = os.path.join(args.data_root, 'spin', args.scenario)

    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)

    for data_path, prefix in zip(data_paths, data_prefixes):
        print(f"Splitting {prefix} on {data_path}")
        output = []
        count = 0

        with open(data_path, 'r') as fp:
            while (line := fp.readline()):
                count += 1
                print(f'Processing row: {count}')
                row = json.loads(line)
                document = row['document']
                summary = row['summary']
                items = []

                for piece in splitter(4096, document):
                    #piece = piece.strip()

                    item = {
                        "article_id": row['article_id'],
                        "document": piece,
                        "abstract": summary,
                        #"summary": summaries[id],
                        "document_len": len(piece.split()),
                        #"summary_len": len(summaries[id].split())
                    }

                    items.append(item)

                print(f"Document len: {row['document_len']}, splitted into: {len(items)}")
                summaries = split_to_part(summary, len(items))
                print(f"Summary len: {row['summary_len']}, splitted into: {len(summaries)}")
                
                for idx, item in enumerate(items):
                    max_rougeL = 0
                    max_summary = None

                    for summary in summaries:
                        result = scorer.score(summary, item['document'])
                        #result = rouge.compute(predictions=item['document'], references=summary)
                        #print(result)
                        
                        metric_index = 0

                        if args.scenario == 1:
                            metric_index = 1
                        elif args.scenario == 2:
                            metric_index = 0

                        if result['rougeL'][metric_index] > max_rougeL:
                            max_rougeL = result['rougeL'][metric_index]
                            max_summary = summary

                    if max_summary is None:
                        max_summary = summaries[np.random.randint(0, len(summaries))]

                    item['summary'] = max_summary
                    item['summary_len'] = len(max_summary.split())

                    #print(f'removing summary: {max_summary}')
                    summaries.remove(max_summary)
                    #print(f'len summaries: {len(summaries)}')
                    
                output.extend(items)

        fname = os.path.join(task_output_dir, f'{prefix}.json')

        with open(fname, 'w') as fp:
            for item in output:
                fp.write(json.dumps(item) + '\n')
                #json.dump(output, fp, indent=2)

        print(f"Finished writing {prefix} split to {task_output_dir}")

if __name__ == "__main__":
    main()
