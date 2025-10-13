import os
import json
import argparse

import setproctitle
from tqdm import tqdm

from openai import OpenAI
from make_data import SYSTEM_PROMPT, TEMPLATE, QUESTIONS
from utils import model_arg_dict, read_jsonl, write_jsonl, writer_jsonl
from run_openai import run_chat, run_batch, process_data


def prepare_data(args):
    results_dir = './data'
    output_name = os.path.join(results_dir, f'k{str(args.k)}---val{str(args.val)}.jsonl')
    data = read_jsonl(output_name)

    def make_single(item, args):
        idx = item['idx']
        question = QUESTIONS[args.question_type].replace('{p1}', item[f"{args.chain_type}_lastname"])
        target = item[f"{args.chain_type}_{args.question_type}_val"]
        tmp_template = TEMPLATE.replace(
            '{names}', item['names']).replace(
            '{context}', item[f'{args.chain_type}_chain']).replace(
            '{question}', question).replace(
            '{num_names}', str(len(item['names'].split(', ')))
        )
        return {'idx': idx, 'question': tmp_template, 'target': target}

    data = [make_single(item, args) for item in data]
    processed = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": d['question']}]
        for d in data
    ]

    return processed, data


def run_hf(client, data, messages, output_name, **kwargs):
    model_name = kwargs['model_name']
    output = []

    exists_, writer_ = writer_jsonl(output_name)
    exists_ids = [i['idx'] for i in exists_]

    for idx, (message, d) in tqdm(enumerate(zip(messages, data))):
        if d['idx'] in exists_ids:
            entry = exists_[exists_ids.index(idx)]
            writer_.write(json.dumps(entry) + '\n')
            writer_.flush()
            continue
        # try:
        completion = client.chat.completions.create(
            model=model_arg_dict[model_name],
            messages=message,
            temperature=0.6, top_p=0.95
        )
        generated = completion.choices[0].message.content
        d['generated'] = generated
        # output.append(d)
        writer_.write(json.dumps(d) + '\n')
        writer_.flush()

    # write_jsonl(output, output_name)
    writer_.close()


def main(args):
    processed, data = prepare_data(
        args=args,
    )
    os.makedirs(args.results_dir, exist_ok=True)
    output_name = os.path.join(args.results_dir, f'{args.output_name}.jsonl')

    if args.model_name in ['gpt-4o', 'gpt-4.1-2025-04-14', 'gpt-4o-2024-08-06', 'gpt-4.1-mini-2025-04-14']:
        # batch
        client = OpenAI(api_key=args.openai_apikey)
        if args.tool:
            print("\n\n ### Tool activated ### \n\n")
            run_chat(client=client, data=data, messages=processed, output_name=output_name, model_name=args.model_name, tool=True)
        else:
            processed = process_data(args.model_name, processed)
            run_batch(client=client, data=data, messages=processed, output_name=output_name)
    elif args.model_name in ['o3', 'o3-mini', 'o3-2025-04-16', 'o3-mini-2025-01-31']:
        client = OpenAI(api_key=args.openai_apikey)
        run_chat(client=client, data=data, messages=processed, output_name=output_name, model_name=args.model_name)
    else:
        client = OpenAI(
            base_url="http://localhost:8123/v1",
            api_key="needlechain",
        )
        run_hf(client=client, data=data, messages=processed, output_name=output_name, model_name=args.model_name)


if __name__ == '__main__':
    print(':(')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='llama3.1')
    parser.add_argument('--openai_apikey', default='OpenAI API key')
    parser.add_argument('--chain_type', default='forward', choices=['forward', 'parallel', 'backward', 'chaotic'])
    parser.add_argument('--question_type', default='single', choices=['single', 'total'])
    parser.add_argument('--val', default=1600, choices=[160, 1600, 16000])
    parser.add_argument('--k', default=5)
    parser.add_argument('--tool', default=False, action='store_true')
    parser.add_argument('--output_name', default='tmp', help="""d""")
    parser.add_argument('--results_dir', default='./results')

    temporal_args = parser.parse_args()
    setproctitle.setproctitle(f'mmmm inference')

    main(temporal_args)

    print(':)')
