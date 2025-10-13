# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import cProfile
import io
import os
import json
import pstats
import random
from collections import defaultdict
import sys
import time
from pstats import SortKey
from fastchat.model import get_conversation_template
import torch
from tqdm import tqdm

class Benchmark():
    def __init__(self,
                 log_dir=None,
                 eos=None,
                 eop=None,
                 device='cuda:0'):
        self.log_dir = log_dir
        self.eos = eos
        self.eop = eop  # end token id of prompt, ignore if end token id of prompt is not a fixed id
        self.device = device

        self.model = None
        self.tokenizer = None

        self.prompts = []
        self.answers = []
        self.ids = []

        self.warmup_prompts = []
        self.warmup_answers = []
        self.warmup_ids = []

        self.stop_words = [',', '.', ' ', '\n', '，', ',']
        self.stop_ids = None

        self.logger = None
        if self.log_dir is not None:
            self.logger = open(self.log_dir, 'w+')

    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        raise NotImplementedError()

    def save_answers(self, src_dir, dst_dir, max_new_tokens=256, batch_size=1, prompt_name='prompt', answer_name='answer', max_count=None, use_lookahead=False, start_id=None, end_id=None):
        # 加载warmup
        lines = open(src_dir).readlines()
        prompts = []
        # answers = []
        user_ids = []
        if start_id is not None and end_id is not None:
            lines = lines[start_id:end_id]
        for d in lines:
            d = json.loads(d)
            # 因为我存的只是一部分prompt，所以要加入template
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], d[prompt_name])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            # answers.append(d[answer_name])
            user_ids.append(d['question_id'])
            if max_count is not None and len(prompts) >= max_count:
                break

        jsons = []
        # 如果没有answer，需要生成
        qaids = self.generate(prompts, max_new_tokens=max_new_tokens, use_lookahead=use_lookahead,
                              decoding_length=64, branch_length=12, batch_size=batch_size, choose_topk=1)
        for i, (p, a, ids, pids) in enumerate(qaids): # queries, output_texts, output_id_list, input_id_list
            jsons.append(json.dumps({'user_id': user_ids[i],'prompt': p, 'answer': a, 'answer_ids': ids, 'prompt_ids': pids})) # prompt ,answer, outputtext, output text id

        # 如果已经有answer了，直接转换为id
        # ts = time.time()
        # for i in range(len(prompts)):
        #     r = answers[i]
        #     ids = self.tokenizer(r).input_ids[1:] # 多了一个id=1的token 句首token
        #     pids = self.tokenizer(prompts[i]).input_ids[1:]
        #     jsons.append(json.dumps({'prompt': prompts[i], 'answer': r, 'ids': ids, 'pids':pids, 'user':user_ids[i]})) # prompt ,answer, answer token id, prompt token id

        #     if (i + 1) % 10000 == 0:
        #         print(f'convert ans to id:{i + 1}, elapse:{round(time.time() - ts, 1)}s')

        # 保存answer
        with open(dst_dir, 'w') as f:
            f.write('\n'.join(jsons))

    def load_prompts(self, prompt_dir=None, warmup_prompt_dir=None, test_dir=None,
                     max_length=2048, prompt_num=None, warmup_num=None):
        prompts = []
        answers = []
        user_ids = []
        self.test_group = int(test_dir.split('_')[-1].split('.')[0])
        print('test group', self.test_group)
        if os.path.exists(test_dir):
            data = open(test_dir, 'r').readlines()
            print('Loading test data from {}'.format(test_dir), len(data))
        else:
            print(prompt_dir)
            data = open(prompt_dir, 'r').readlines()
            remain_data = []
            for i, line in enumerate(data):
                # print(line)
                line_dict = json.loads(line)
                # print(line_dict)
                user_id = line_dict['question_id']
                if user_id not in self.user_class:
                    print('not in class', user_id)
                    continue
                user_class = self.user_class[user_id]
                if user_class == self.test_group:
                    remain_data.append(line)
            print('group', self.test_group, 'num', len(remain_data))
            if prompt_num is not None and len(remain_data) >= prompt_num:
                remain_data = random.sample(remain_data, prompt_num)
            data = remain_data
            # print(len(test_dir))
            with open(test_dir, 'w') as f:
                f.write(''.join(remain_data))
            print('Saving test data to {}'.format(test_dir))
        full_data = []
        for i, line in enumerate(data):
        # for i, line in enumerate(open(prompt_dir, 'r')):
            if prompt_num is not None and i >= prompt_num:
                break
            line = json.loads(line)
            if isinstance(line, str):
                # str中存在转义字符时第一次会返回str，需要再处理一遍
                line = json.loads(line)
            # 因为我存的只是一部分prompt，所以要加入template
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], line['prompt'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            user_id = line['question_id']
            full_data.append((prompt, line.get('answer', None), user_id, self.user_class[user_id]))
        # full_data.sort(key=lambda x: x[-1])

        prev_group = full_data[0][-1]
        switch_idx = []
        for i, data in enumerate(full_data):
            prompt, ans, user_id, group= data
            prompts.append(prompt)
            answers.append(ans)
            user_ids.append(user_id)
            if prev_group != group:
                switch_idx.append(i)
            prev_group = group
        print('switch idx', switch_idx)
        # int_user = [int(user) for user in user_ids]
        print(prompts[0])
        # print('--------------max user num', max(int_user), len(user_ids))
        # 加载prompt
        self.prompts = [x for x in prompts if len(x)<=max_length or len(self.tokenizer(x).input_ids)<=max_length]
        self.answers = answers
        self.user_ids = user_ids
        self.switch_idx = switch_idx
        print("all prompt length: ", len(self.prompts))
        # print("prompt sample: ", self.prompts[0], self.answers[0])
        print(warmup_prompt_dir)

        if warmup_prompt_dir is not None:
            prompts = []
            answers = []
            ids = []
            user_ids = []
            for i, line in enumerate(open(warmup_prompt_dir, 'r')):
                if warmup_num is not None and i >= warmup_num:
                    break
                # print(line)
                line_dict = json.loads(line.strip())
                if isinstance(line_dict, str):
                    # str中存在转义字符时第一次会返回str，需要再处理一遍
                    line_dict = json.loads(line_dict)
                # 这里因为save answer中存下来的prompt就是完整的 所以不用template
                prompts.append(line_dict['prompt'])
                answers.append(line_dict.get('answer', None))
                ids.append(line_dict.get('answer_ids', None))
                user_ids.append(line_dict.get('user_id', None))
            # int_user = [int(user) for user in user_ids]
            # print('--------------max user num', max(int_user), len(user_ids))
            # 加载warmup prompt
            self.warmup_prompts = [x for x in prompts if len(x)<=max_length or len(self.tokenizer(x).input_ids)<=max_length]
            self.warmup_answers = answers
            self.warmup_ids = ids
            self.warmup_user_ids = user_ids
            print("all warmup length: ", len(self.warmup_prompts))
            # print("warmup sample: ", self.warmup_prompts[0], self.warmup_answers[0], self.warmup_ids[0])

    def tokenize(self, prompt, max_length=256):
        if isinstance(prompt, list):
            inputs = self.tokenizer(prompt,
                                    padding=True,
                                    truncation=False,
                                    return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = None
        return input_ids, position_ids, attention_mask

    def chat(self, prompt, max_length=2048, max_new_tokens=256, use_lookahead=False, decoding_length=64, branch_length=8,
             decoding_mode='hier', debug_lookahead=False, max_query_length=2, choose_topk=1, repetition_penalty=1.0,
             choose_topp=0.0):
        if use_lookahead and decoding_length > 1 and branch_length > 0:
            max_gen_length = max_new_tokens + decoding_length + 1
        else:
            max_gen_length = max_new_tokens
        input_ids, position_ids, attention_mask = self.tokenize(prompt, max_length=max_gen_length)
        tokenizer = self.tokenizer
        model = self.model

        decoding_kwargs = {"use_lookahead": use_lookahead,
                        "debug_lookahead": debug_lookahead,
                        "decoding_mode": decoding_mode,
                        "decoding_length": decoding_length,
                        "branch_length": branch_length,
                        "max_query_length": max_query_length,
                        "stop_words": self.stop_ids,
                        'choose_topk': choose_topk,
                        'choose_topp': choose_topp}
        assert self.eos is not None
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 pad_token_id=self.eos,
                                 eos_token_id=self.eos,
                                 use_cache=True,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 decoding_kwargs=decoding_kwargs,
                                 repetition_penalty=repetition_penalty,
                                 return_dict_in_generate=True
                                 )
        output_ids = outputs.sequences
        kwargs = outputs.kwargs if hasattr(outputs, 'kwargs') else {}
        input_length = input_ids.size(-1)
        output_ids = output_ids[:, input_length:].tolist()
        # output_ids = output_ids.tolist()
        output_texts = []
        output_id_list = []
        for token_ids in output_ids:
            output_id_list.append(token_ids)
            text = tokenizer.decode(token_ids)
            output_texts.append(text)
        input_id_list = input_ids.tolist()
        # input_texts = tokenizer.batch_decode(input_ids)
        return prompt, input_id_list, output_id_list, output_texts, kwargs

    # 填充检索池
    def warm_up(self, ids, users, branch_length=8, eop=None, max_cache=2000):
        lookahead_cache_list = self.model.lookahead_cache_list
        # ts = time.time()
        # for i, ids_ in enumerate(ids):
        #     if ids_ is None:
        #         continue
        #     # 只放answer id
        #     lookahead_cache = lookahead_cache_list[self.user_class[users[i]]]
        #     lookahead_cache.put([eop] + ids_ if eop else ids_, branch_length=branch_length + 1, mode='output', idx=-1)
        #
        #     if (i + 1) % 5000 == 0:
        #         print(f'warmup:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        #         for j, lookahead_cache in enumerate(lookahead_cache_list):
        #             print(f"Cache {j} Mem size: ", len(lookahead_cache.mem))
        #         # lookahead_cache.save_mem(f'datastore_{i + 1}.json')
        #         # print('store cache: ', f'datastore_{i + 1}.json')

        lookahead_cache_data = [[] for _ in range(len(lookahead_cache_list))]
        for i, ids_ in enumerate(ids):
            if ids_ is None:
                continue
            lookahead_cache_data[self.user_class[users[i]]].append(i)
        ts = time.time()
        for j, data in enumerate(lookahead_cache_data):
            if j == self.test_group:
                if max_cache is not None and len(data) > max_cache:
                    selected_cache_data = random.sample(data, max_cache)
                else:
                    selected_cache_data = data
                for i in selected_cache_data:
                    ids_ = ids[i]
                    lookahead_cache_list[j].put([eop] + ids_ if eop else ids_, branch_length=branch_length + 1,
                                                mode='output', idx=-1)
                print(f'warmup data:{len(selected_cache_data)}, elapse:{round(time.time() - ts, 1)}s')
                print(f"Cache {j} Mem size: ", len(lookahead_cache_list[j].mem))

    def generate(self, qs, use_lookahead=True, max_new_tokens=256, decoding_length=64, branch_length=8, batch_size=16, choose_topk=1):
        chat_count = len(qs)
        qas = []
        ts = time.time()
        for i in tqdm(range((chat_count - 1) // batch_size + 1)):
            queries = qs[i * batch_size:(i + 1) * batch_size]
            input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(queries,
                                                        max_new_tokens=max_new_tokens,
                                                        use_lookahead=use_lookahead,
                                                        decoding_length=decoding_length,
                                                        branch_length=branch_length,
                                                        choose_topk=choose_topk)
            for j in range(len(queries)):
                qas.append((queries[j], output_texts[j], output_id_list[j], input_id_list[j]))
            if (i + 1) % 1000 == 0:
                print(f'generate:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
        return qas

    # def batch_chat(self, qs, max_new_tokens=256, decoding_length=64, branch_length=8, decoding_mode='hier',
    #                debug_lookahead=False, erase=True, batch_size=1, max_query_length=2):
    #     total_out_tokens = [0, 0]
    #     total_times = [0, 0]
    #     lookahead_cache = self.model.lookahead_cache
    #     if erase:
    #         lookahead_cache.fresh()
    #     chat_count = len(qs)
    #     for i in range(chat_count // batch_size):
    #         query = qs[i * batch_size:(i + 1) * batch_size]
    #         speeds = []
    #         for j, use_lookahead in enumerate([False, True]):
    #             in_char = 0
    #             in_token = 0
    #             out_char = 0
    #             out_token = 0
    #             ts = time.time()
    #             # 调用chat
    #             input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(query,
    #                                     max_new_tokens=max_new_tokens,
    #                                     use_lookahead=use_lookahead,
    #                                     decoding_length=decoding_length,
    #                                     branch_length=branch_length,
    #                                     decoding_mode=decoding_mode,
    #                                     debug_lookahead=debug_lookahead,
    #                                     max_query_length=max_query_length)
    #             in_char += sum([len(x) for x in input_texts])
    #             in_token += sum([len(x) for x in input_id_list])
    #             out_char += sum([len(x) for x in output_texts])
    #             out_token += sum([len(x) for x in output_id_list])
    #             t = (time.time() - ts)
    #             speed_char = out_char / t
    #             speed_token = out_token / t
    #             speeds.append(speed_token)
    #             total_out_tokens[j] += out_token
    #             total_times[j] += t
    #             bs = len(query)
    #             dls = kwargs.get('dls', [])
    #             dl = sum(dls[bs:]) / len(dls[bs:]) if len(dls) > bs else 0.0
    #             edls = kwargs.get('edls', [])
    #             edl = sum(edls[bs:]) / len(edls[bs:]) if len(edls) > bs else 0.0
    #             pt = kwargs.get('fts', [0])[0]
    #             gts = kwargs.get('fts', [0])[1:]
    #             gt = sum(gts)/max(len(gts),1)
    #             print(f"1/{bs} Robot:{output_texts[0]}")
    #             prefix = 'lookahead:' + ('On ' if use_lookahead else 'Off')
    #             speedup = speeds[-1] / speeds[0] if use_lookahead else 0.0
    #             print(
    #                 f"{prefix} mode:{decoding_mode} idx:{i} "
    #                 f"input:{in_char:.1f}/{in_token:.1f} output:{out_char:.1f}/{out_token:.1f} "
    #                 f"edl:{edl:.3f}/{dl:.3f}/{pt:.3f}/{gt:.3f} time:{t:.3f} speed:{speed_token:.1f} speedup:{speedup:.3f}\n")
    #     org_speed = total_out_tokens[0] / total_times[0]
    #     opt_speed = total_out_tokens[1] / total_times[1]
    #     speedup = opt_speed / org_speed
    #     print(f'speed:{org_speed:.2f}->{opt_speed:.2f} speedup:{speedup:.3f}')

    # 性能测试
    def perf_check(self, queries, queries_users=None, warmup_ids=None, warmup_users=None, warmup_count=None, max_new_tokens=256,
                   sizes=(32, 64), lens=(4, 8, 12), decoding_mode='hier', batch_size=1, max_node_rate=16,
                   max_query_length=2, choose_topk=1, repetition_penalty=1.0, choose_topp=0.0):
        wc = warmup_count if warmup_count is not None else 0
        log_str = f'\nmode:{decoding_mode} bs:{batch_size} queries:{len(queries)} warmup:{wc} sizes:{sizes} lens:{lens} topk:{choose_topk}'
        print(log_str)
        # if batch_size > 1:
        #     queries = sorted(queries, key=lambda x: len(x))
        speeds = []
        outputs = {}
        lookahead_cache_list = self.model.lookahead_cache_list
        for i, decoding_length in enumerate(sizes):
            for j, branch_length in enumerate(lens):
                if decoding_length < branch_length * batch_size:
                    continue
                use_lookahead = decoding_length > 1 and branch_length > 0 and warmup_count > 0
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                dls = []
                edls = []
                pts = []
                gts = []
                qts = []

                # 对于每一个不同的decoding_length和branch_length，要fresh cache
                if use_lookahead:
                    for lookahead_cache in lookahead_cache_list:
                        lookahead_cache.fresh()
                        lookahead_cache.max_output_node = max_node_rate * decoding_length
                        lookahead_cache.max_node = 2*max_node_rate * decoding_length

                    # # 直接加载datastore
                    # lookahead_cache.load_mem(f'datastore_{len(warmup_ids)}.json')
                    # print('Cache load: ', f'datastore_{len(warmup_ids)}.json')

                    # # # 如果没有datastore， 先生成cache
                    if warmup_ids is not None:
                        self.warm_up(warmup_ids, warmup_users, branch_length=branch_length, eop=self.eop,
                                     max_cache=warmup_count)
                        
                    # lookahead_cache_list.save_mem(f'datastore_{len(warmup_ids)}.json')
                    # print("Store Cache")

                    # print("Warmup Done")
                # exit()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=None)
                ts = time.time()
                n_b = len(queries) // batch_size
                times = []
                self.model.lookahead_cache = lookahead_cache_list[self.test_group]
                # self.model.lookahead_cache = lookahead_cache_list[self.user_class[queries_users[0]]]
                for k in tqdm(range(n_b)):
                    qs_ = queries[k * batch_size:(k + 1) * batch_size]
                    # if k in self.switch_idx:
                    #     self.model.lookahead_cache = lookahead_cache_list[self.user_class[queries_users[k]]]
                    # self.model.lookahead_cache = lookahead_cache_list[self.user_class[queries_users[k]]]
                    ts_ = time.time()
                    input_texts, input_id_list, output_id_list, output_texts, kwargs = self.chat(qs_,
                                max_new_tokens=max_new_tokens,
                                use_lookahead=use_lookahead,
                                decoding_length=decoding_length,
                                branch_length=branch_length,
                                decoding_mode=decoding_mode,
                                max_query_length=max_query_length,
                                choose_topk=choose_topk,
                                repetition_penalty=repetition_penalty,
                                choose_topp=choose_topp)
                    
                    # print("Cache Mem size: ", len(lookahead_cache.mem))
                    if k == 1:
                        print('input text', input_texts)
                        print('output_texts:', output_texts)
                        for id_list in kwargs['final_list']:
                            text = [self.tokenizer.decode(i) for i in id_list]
                            print(text)
                        exit()
                    te_ = time.time()
                    times.append(te_ - ts_)
                    in_char += sum([len(x) for x in qs_])
                    in_token += sum([len(x) for x in input_id_list])
                    out_char += sum([len(x) for x in output_texts])
                    out_token += sum([len(x) for x in output_id_list])
                    bs = len(qs_)
                    # print(kwargs) # dls edls fts qts
                    # for tmp_k,tmp_v in kwargs.items():
                    #     print(tmp_k, len(tmp_v))

                    dls_ = kwargs.get('dls', [])
                    dls.extend(dls_[bs:] if len(dls_) > bs else [])
                    edls_ = kwargs.get('edls', [])
                    edls.extend(edls_[bs:] if len(edls_) > bs else []) # effective decoding lengths
                    
                    pts.append(kwargs.get('fts', [0])[0]) # prefil time
                    gts.extend(kwargs.get('fts', [0])[1:]) # decoding time

                    qts.extend(kwargs.get('qts', [0])) # retrieve time
                    if (k + 1) % (100 // batch_size) == 0:
                        elapse = time.time() - ts
                        speed = out_token / elapse
                        avg_in_token = float(in_token) / (k + 1) / batch_size
                        avg_out_token = float(out_token) / (k + 1) / batch_size
                        dl = sum(dls) / max(len(dls), 1)
                        edl = sum(edls) / max(len(edls), 1)
                        pt = sum(pts) / max(len(pts), 1)
                        gt = sum(gts) / max(len(gts), 1)
                        qt = sum(qts) / max(len(qts), 1)
                        log_str = f'group num: {self.group_num} mode:{decoding_mode} step:{k + 1} ' \
                                  f'decoding:{decoding_length}/{branch_length} bs:{batch_size} ' \
                                  f'elapse:{elapse:.1f}s in:{avg_in_token:.1f} out:{avg_out_token:.1f} ' \
                                  f'edl:{edl:.3f}/{dl:.3f}/{pt:.3f}/{gt:.3f}/{qt:.3f} speed:{speed:.1f}token/s ' \
                                  f'All prefill time: {sum(pts):.3f}s All decoding time: {sum(gts):.3f}s All retrieval time: {sum(qts):.3f}s All time :{elapse:.1f}s'
                        print(log_str)
            
                n_repeat = len(queries)
                in_char /= n_repeat
                in_token /= n_repeat
                out_char /= n_repeat
                out_token /= n_repeat
                # 平均每个query所需时间
                t = (time.time() - ts) / n_repeat
                speed = out_token / t
                speeds.append(speed)
                outputs[(decoding_length, branch_length)] = speed
                dl = sum(dls) / max(len(dls), 1)
                edl = sum(edls) / max(len(edls), 1)
                pt = sum(pts) / max(len(pts), 1) # prefil time
                gt = sum(gts) / max(len(gts), 1) # decoding time
                qt = sum(qts) / max(len(qts), 1) # retrieve time
                ms = torch.cuda.memory_stats()
                mem = ms['reserved_bytes.large_pool.peak'] / 1e9
                speedup = speeds[-1] / speeds[0]
                times = [round(x, 3) for x in times]
                log_str = f"group num: {self.group_num} mode:{decoding_mode} bs:{batch_size} " \
                          f"decoding_length:{decoding_length} branch_length:{branch_length} " \
                          f"query:{len(queries)} warmup:{wc} topk:{choose_topk} " \
                          f"input:{in_token:.1f} output:{out_token:.1f} edl num: {len(edls)}" \
                          f"edl:{edl:.3f}/{dl:.3f}/{pt:.3f}/{gt:.3f}/{qt:.3f} time:{t:.3f} " \
                          f"speed:{speed:.1f} mem:{mem:.3f} " \
                          f"All prefill time: {sum(pts):.3f}s All decoding time: {sum(gts):.3f}s All retrieval time: {sum(qts):.3f}s All time: {t*n_repeat:.3f}"
                print(log_str)
    
                if self.logger is not None:
                    self.logger.write(log_str + '\n')
                    self.logger.flush()

        return outputs

    def perf_check_trie(self, lookahead_cache, warmup_ids, input_ids, output_ids, max_node_rate=16, decoding_length=64, branch_length=24, edl=8):
        lookahead_cache.max_output_node=decoding_length*max_node_rate
        lookahead_cache.fresh()

        for i, ids_ in enumerate(warmup_ids):
            lookahead_cache.put(ids_, branch_length=branch_length + 1, mode='output', idx=0, final=False)

        count = len(input_ids)
        put_count = 0
        put_time = 0.0
        get_count = 0
        get_time = 0.0
        for i in range(count):
            in_ids = input_ids[i]
            out_ids = output_ids[i]

            put_count += len(in_ids)
            ts = time.time()
            lookahead_cache.put(in_ids, branch_length=branch_length + 1, mode='input', idx=0, final=False)
            put_time += time.time()-ts

            ts = time.time()
            for j in range(0, len(out_ids) - 1, edl):
                get_count += 1
                lookahead_cache.bat_get([out_ids[j:j + 2]], decoding_length=decoding_length,
                                        branch_length=branch_length, decoding_cursors=[j], mode='mix',
                                        indices=[0], decoding_mode='hier')
            get_time += time.time()-ts

            put_count += len(out_ids)
            ts = time.time()
            for j in range(0, len(ids_) - 1, edl):
                lookahead_cache.stream_put(out_ids[j:j+edl], branch_length=branch_length + 1, mode='output', idx=0, final=False)
            lookahead_cache.stream_put([], branch_length=branch_length + 1, mode='output', idx=0, final=True)
            put_time += time.time()-ts

        single_put_time = put_time/max(put_count,1)
        sample_put_time = put_time/max(count,1)

        single_get_time = get_time/max(get_count,1)
        sample_get_time = get_time/max(count,1)

        print(f'\nparam:{max_node_rate}/{decoding_length}/{branch_length} sample:{count} put:{put_count}/{put_time:.2f}/{single_put_time*1e3:.2f}/{sample_put_time*1e3:.2f} get:{get_count}/{get_time:.2f}/{single_get_time*1e3:.2f}/{sample_get_time*1e3:.2f}\n')

    def naive_profile(self, qs, use_lookahead=False, count=64, sortby=0):
        pr = cProfile.Profile()
        pr.enable()
        for q in qs:
            self.chat(q, use_lookahead=use_lookahead)
        pr.disable()
        s = io.StringIO()
        if sortby == 0:
            sortby = SortKey.TIME
        else:
            sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def naive_profile_trie(self, lookahead_cache, warmup_ids, max_node_rate=16, decoding_length=64, branch_length=24, edl=8, put_count=10000, get_count=100,
                        count=64, sortby=0):
        pr = cProfile.Profile()
        pr.enable()
        self.perf_check_trie(lookahead_cache, warmup_ids, max_node_rate=max_node_rate, decoding_length=decoding_length, branch_length=branch_length,
                             edl=edl, put_count=put_count, get_count=get_count)
        pr.disable()
        s = io.StringIO()
        if sortby == 0:
            sortby = SortKey.TIME
        else:
            sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(count)
        print(s.getvalue())

    def torch_profile(self, use_lookahead=False):

        # pip install torch_tb_profiler
        # tensorboard --logdir=./prof
        # http://localhost:6006/#pytorch_profiler

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof'),
            record_shapes=True,
            with_stack=True)
        prof.start()
        for p in self.prompts:
            prof.step()
            self.chat(p, use_lookahead=use_lookahead)
        prof.stop()

    def to_words(self, token_ids):
        if isinstance(token_ids, list):
            tokens = []
            for i in token_ids:
                tokens.append(self.tokenizer._convert_id_to_token(i))
            print(tokens)
        else:
            print(self.tokenizer._convert_id_to_token(token_ids))

    def to_ids(self, tokens):
        return self.tokenizer._convert_token_to_id(tokens)

    def grid_search(self, chat_count=100, warmup_count=10000):

        ps = self.prompts
        warmup_ids = self.warmup_ids
        outputs = self.perf_check(ps[:chat_count],
                                  warmup_ids=warmup_ids[:warmup_count],
                                  sizes=[16 * x for x in [1, 2, 4, 8, 16]],
                                  lens=[4 * x for x in range(1, 11)],
                                  batch_size=1)

        opt_size, opt_len = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[0][0]
        for rate in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            self.perf_check(ps[:chat_count], warmup_ids=ids[:warmup_count], sizes=[opt_size],
                            lens=[opt_len], max_node_rate=rate)

    def batch_grid_search(self, chat_count=100, warmup_count=10000):

        ps = self.prompts
        warmup_ids = self.warmup_ids
        decoding_mode = 'hier'
        for bs in [2, 4, 6, 8]:
            outputs = self.perf_check(ps[:chat_count],
                                      warmup_ids=warmup_ids[:warmup_count],
                                      sizes=[16 * x - bs for x in [8, 16]],
                                      lens=[4, 8, 12, 16],
                                      batch_size=bs,
                                      decoding_mode=decoding_mode)

            opt_size, opt_len = sorted(outputs.items(), key=lambda x: x[1], reverse=True)[0][0]
            self.perf_check(ps[:chat_count], warmup_ids=warmup_ids[:warmup_count], sizes=[opt_size],
                            lens=[opt_len], batch_size=bs)