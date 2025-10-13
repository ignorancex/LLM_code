import argparse
from os.path import join
from tqdm import tqdm
from setproctitle import setproctitle

from openai import OpenAI

from utils import *


def process_messages(custom_id, message, model_name):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model_name,
                 "messages": message,
                 "max_tokens": 16384}
    }


def process_data(model_name, data_list):
    data = []
    messages = []
    for idx, item in enumerate(data_list):
        message = process_messages(
            custom_id=f"request-{idx}",
            message=item,
            model_name=model_name
        )
        messages.append(message)
    return messages


def run_batch(client, data, messages, output_name, **kwargs):
    name_msg = output_name + '---msg'
    name_batch = output_name + '---batch'
    name_original = output_name + '---original'
    name_output = output_name

    if not os.path.exists(name_batch):
        write_jsonl(data,     name_original)
        write_jsonl(messages, name_msg)

        batch_input_file = client.files.create(
            file=open(name_msg, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        batch_created = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        write_json(batch_created.model_dump(), name_batch)
        print(f'batch created: {name_batch}')
    else:
        batch = read_json(name_batch)
        try:
            batch_id = client.batches.retrieve(batch['id'])
        except:
            print('batch exist.. but different API key')
            return

        if batch_id.status == 'completed':
            print(f'completed: {name_batch}')
            if not os.path.exists(name_output):
                responses = client.files.content(batch_id.output_file_id).text
                input_file = read_jsonl(name_original)

                temporal_filename = 'temporal_fileeeeeee.jsonl'
                write_file(responses, temporal_filename)
                responses = read_jsonl(temporal_filename)
                responses = [item['response']['body']['choices'][0]['message']['content'] for item in responses]

                input_file = [{'generated': response, **others} for response, others in zip(responses, input_file)]
                write_jsonl(input_file, name_output)
                if os.path.exists(temporal_filename):
                    os.remove(temporal_filename)

        elif batch_id.status == 'expired':
            print(f'expired: {name_batch}')
        else:
            print(f'{batch_id.status}: {name_batch} - {batch_id.request_counts}')
            # client.batches.cancel(batch['id'])


def run_chat(client, data, messages, output_name, **kwargs):
    if os.path.exists(output_name):
        _exist = read_jsonl(output_name)
        _exist_ids = [i['idx'] for i in _exist]  # CAUTION: "idx" key sould be included in data
        for pos, instance in enumerate(data):
            if instance['idx'] in _exist_ids:
                data[pos] = _exist[_exist_ids.index(instance['idx'])]

    result_writer = open(output_name, 'w')

    for entry, message in tqdm(zip(data, messages)):
        if entry.get('generated', None) is not None:
            result_writer.write(json.dumps(entry) + '\n')
            result_writer.flush()
            continue
        success = False
        while not success:
            if kwargs.get("tool", False):
                completion = client.responses.create(
                    model=kwargs['model_name'],
                    tools=[
                        {
                            "type": "code_interpreter",
                            "container": {"type": "auto"}
                        }
                    ],
                    instructions=message[0]['content'],
                    input=message[1]['content'],
                    temperature=1,
                )
                # generation = completion.output[0].content[0].text
                generation = completion.output_text
            else:
                completion = client.chat.completions.create(
                    model=kwargs['model_name'],
                    messages=message,
                    temperature=1,
                )
                generation = completion.choices[0].message.content
            entry['generated'] = generation
            success = True

        result_writer.write(json.dumps(entry) + '\n')
        result_writer.flush()

    result_writer.close()
