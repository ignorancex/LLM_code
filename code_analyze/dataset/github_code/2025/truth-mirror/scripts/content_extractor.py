# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os.path as path
from tqdm import tqdm
from utils import load_json, save_json, full_lang
from extractor_base import ExtractorBase


TASK_PROMPT_EXTRACT_CONTENT_OUTLINE = \
'''Outline the main points of the text in {lang} to get a clear and concise picture of the content:
{generation}
'''

TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY = \
'''Simplify the text in {lang} to make it clear and concise while preserving its meaning:
{generation}
'''


class ContentExtractor(ExtractorBase):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.task_prompts = {
            'outline': TASK_PROMPT_EXTRACT_CONTENT_OUTLINE,
            'simplify': TASK_PROMPT_EXTRACT_CONTENT_SIMPLIFY
        }

    def _get_task_prompt(self, dataset, generation):
        kwargs = {'lang': full_lang(self.lang), 'generation': generation}
        template = self.task_prompts[self.args.prompt]
        return template, template.format(**kwargs)

    def _extract_item(self, dataset, item):
        id = item['id']
        generation = item['generation']
        # call model
        model_name = self.model
        sys_prompt = self.sys_prompt
        task_template, task_prompt = self._get_task_prompt(dataset, generation)
        temperature = self.temperature
        topp = self.top_p
        validate_fn = self._get_validate_fn(item, model_name)
        content_field = 'content'
        # check existence
        if content_field in item and validate_fn(item[content_field]):
            return item
        # call API
        try:
            content = self.model_proxy.prompt_generate(model_name, sys_prompt, task_prompt,
                                identifier=id, temperature=temperature, top_p=topp,
                                validate_fn=validate_fn)
            item[content_field] = content
        except Exception as e:
            print(e)
        return item

    def extract_content(self, dataset):
        # load data
        data_file = path.join(self.args.data_path, f'{dataset}.json')
        data = load_json(data_file)
        # refine the language expression
        data = [self._extract_item(dataset, item) for item in tqdm(data)]
        save_json(data_file, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./benchmark/hart")
    parser.add_argument('--datasets', type=str, default='essay.dev')
    parser.add_argument('--prompt', type=str, default="simplify", choices=['simplify', 'outline'])
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')
    for dataset in args.datasets:
        print(f'Extract content for {dataset} ...')
        extractor = ContentExtractor(args, dataset)
        extractor.extract_content(dataset)

