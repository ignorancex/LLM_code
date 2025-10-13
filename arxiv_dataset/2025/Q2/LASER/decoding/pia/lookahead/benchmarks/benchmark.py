# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from __future__ import print_function

import cProfile
import io
import json
import os.path
import pstats
import random
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
                 device='cuda:0',
                 use_int8=False):
        self.log_dir = log_dir
        self.eos = eos
        self.eop = eop  # end token id of prompt, ignore if end token id of prompt is not a fixed id
        self.device = device
        self.use_int8 = use_int8

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

    def initialize(self, model_dir=None, token_dir=None, use_int8=False, **kwargs):
        raise NotImplementedError()

    def save_answers(self, src_dir, dst_dir, max_new_tokens=256, batch_size=1, prompt_name='prompt', data_type=None,
                     warmup_prompt_dir=None, warmup_num=None, max_count=None, use_lookahead=False, start_id=None,
                     end_id=None, sample=False, choose_topk=1, choose_topp=0.0, max_length=2048,
                     database_dir=None, branch_length=12, decoding_length=64):
        # 加载warmup
        if warmup_prompt_dir is not None and use_lookahead:
            prompts = []
            answers = []
            ids = []
            for i, line in enumerate(open(warmup_prompt_dir, 'r', encoding="utf-8")):
                if warmup_num is not None and i >= warmup_num:
                    break
                # print(line)
                line = json.loads(line, strict=False)
                if isinstance(line, str):
                    # str中存在转义字符时第一次会返回str，需要再处理一遍
                    line = json.loads(line)
                # 这里因为save answer中存下来的prompt就是完整的 所以不用template
                prompts.append(line['prompt'])
                answers.append(line.get('answer', None))
                ids.append(line.get('answer_ids', None))
            # 加载warmup prompt
            self.warmup_prompts = [x for x in prompts if len(x)<=max_length or len(self.tokenizer(x).input_ids)<=max_length]
            self.warmup_answers = answers
            self.warmup_ids = ids
            print("all warmup length: ", len(self.warmup_prompts))
            if len(self.warmup_ids) > 0:
                lookahead_cache = self.model.lookahead_cache
                database_name = f'{database_dir}/{data_type}_datastore_{len(self.warmup_ids)}.json'
                if os.path.exists(database_name):
                    # # 直接加载datastore
                    lookahead_cache.load_mem(database_name)
                    print('Cache load: ', database_name)
                else:
                    # # # 如果没有datastore， 先生成cache
                    self.warm_up(self.warmup_ids, branch_length=branch_length, eop=self.eop)
                    lookahead_cache.save_mem(database_name)
                    print("Store Cache:", database_name)

                print("Cache Mem size: ", len(lookahead_cache.mem))
        #     else:
        #         use_lookahead = False
        # else:
        #     use_lookahead = False


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
                              decoding_length=decoding_length, branch_length=branch_length, batch_size=batch_size,
                              choose_topk=choose_topk, choose_topp=choose_topp, sample=sample)
        for i, (p, a, ids, pids) in enumerate(qaids): # queries, output_texts, output_id_list, input_id_list
            jsons.append(json.dumps({
                'user_id': user_ids[i],
                'prompt': p,
                'answer': a,
                'answer_ids': ids,
                'prompt_ids': pids
            }, ensure_ascii=False)) # prompt ,answer, outputtext, output text id


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
        # ramdom select and save
        if os.path.exists(test_dir):
            data = open(test_dir, 'r').readlines()
            print('Loading test data from {}'.format(test_dir))
        else:
            data = open(prompt_dir, 'r').readlines()
            if prompt_num is not None and len(data) >= prompt_num:
                data = random.sample(data, prompt_num)
            with open(test_dir, 'w') as f:
                f.write(''.join(data))
            print('Saving test data to {}'.format(test_dir))

        for i, line in enumerate(data):
            if prompt_num is not None and i >= prompt_num:
                break
            # print(line)
            line = json.loads(line)
            # 因为我存的只是一部分prompt，所以要加入template
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], line['prompt'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

            answers.append(line.get('answer', None))
        print(prompts[0])
        # 加载prompt
        self.prompts = [x for x in prompts if len(x)<=max_length or len(self.tokenizer(x).input_ids)<=max_length]
        self.answers = answers
        print("all prompt length: ", len(self.prompts))
        # print("prompt sample: ", self.prompts[0], self.answers[0])

        if warmup_prompt_dir is not None:
            prompts = []
            answers = []
            ids = []
            for i, line in enumerate(open(warmup_prompt_dir, 'r', encoding="utf-8")):
                if warmup_num is not None and i >= warmup_num:
                    break
                line = json.loads(line)
                if isinstance(line, str):
                    # str中存在转义字符时第一次会返回str，需要再处理一遍
                    line = json.loads(line)
                # 这里因为save answer中存下来的prompt就是完整的 所以不用template
                prompts.append(line['prompt'])
                answers.append(line.get('answer', None))
                ids.append(line.get('answer_ids', None))
            # 加载warmup prompt
            self.warmup_prompts = [x for x in prompts if len(x)<=max_length or len(self.tokenizer(x).input_ids)<=max_length]
            self.warmup_answers = answers
            self.warmup_ids = ids
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
             decoding_mode='hier', debug_lookahead=False, max_query_length=2, choose_topk=1, sample=False,
             repetition_penalty=1.0, choose_topp=0.0):
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
                                 do_sample=sample,
                                 repetition_penalty=repetition_penalty,
                                 decoding_kwargs=decoding_kwargs,
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
    def warm_up(self, ids, branch_length=8, eop=None):
        ts = time.time()
        lookahead_cache = self.model.lookahead_cache
        for i, ids_ in enumerate(ids):
            if ids_ is None:
                continue
            # 只放answer id
            lookahead_cache.put([eop] + ids_ if eop else ids_, branch_length=branch_length + 1, mode='output', idx=-1)

            if (i + 1) % 5000 == 0:
                print(f'warmup:{i + 1}, elapse:{round(time.time() - ts, 1)}s')
                print("Cache Mem size: ", len(lookahead_cache.mem))
                # lookahead_cache.save_mem(f'datastore_{i + 1}.json')
                # print('store cache: ', f'datastore_{i + 1}.json')

    def generate(self, qs, use_lookahead=True, max_new_tokens=256, decoding_length=64,
                 branch_length=8, batch_size=16, choose_topk=1, choose_topp=0.0, sample=False):
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
                                                        choose_topk=choose_topk,
                                                        choose_topp=choose_topp,
                                                        sample=sample)
            if i == 0:
                print(input_texts)
                print(output_texts)
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
    def perf_check(self, queries, warmup_ids=None, max_new_tokens=256, sizes=(32, 64), lens=(4, 8, 12), data_type=None,
                   decoding_mode='hier', batch_size=1, max_node_rate=16, max_query_length=2, choose_topk=1,
                   database_dir='./database/', repetition_penalty=1.0, choose_topp=0.0):
        wc = len(warmup_ids) if warmup_ids is not None else 0
        log_str = f'\nmode:{decoding_mode} bs:{batch_size} queries:{len(queries)} warmup:{wc} sizes:{sizes} lens:{lens} topk:{choose_topk}'
        print(log_str)
        # if batch_size > 1:
        #     queries = sorted(queries, key=lambda x: len(x))
        speeds = []
        outputs = {}
        lookahead_cache = self.model.lookahead_cache
        for i, decoding_length in enumerate(sizes):
            for j, branch_length in enumerate(lens):
                if decoding_length < branch_length * batch_size:
                    continue
                use_lookahead = decoding_length > 1 and branch_length > 0 and len(warmup_ids) > 0
                in_char = 0
                in_token = 0
                out_char = 0
                out_token = 0
                dls = []
                edls = []
                pts = []
                gts = []
                qts = []
                print('--------------------------Without lookahead----------------------------')
                # 对于每一个不同的decoding_length和branch_length，要fresh cache
                if use_lookahead:
                    lookahead_cache.fresh()
                    lookahead_cache.max_output_node = max_node_rate * decoding_length
                    lookahead_cache.max_node = 2*max_node_rate * decoding_length
                    database_file = f'{database_dir}/{data_type}_datastore_{len(warmup_ids)}.json'
                    if os.path.exists(database_file):
                        # # 直接加载datastore
                        lookahead_cache.load_mem(database_file)
                        print('Cache load: ', database_file)
                    else:
                        # # # 如果没有datastore， 先生成cache
                        if warmup_ids is not None:
                            self.warm_up(warmup_ids, branch_length=branch_length, eop=self.eop)
                        lookahead_cache.save_mem(database_file)
                        print("Store Cache:", database_file)

                    print("Cache Mem size: ", len(lookahead_cache.mem))

                    # print("Warmup Done")
                # exit()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=None)
                ts = time.time()
                n_b = len(queries) // batch_size
                times = []
                for k in tqdm(range(n_b)):
                    qs_ = queries[k * batch_size:(k + 1) * batch_size]
                    ts_ = time.time()
                    (input_texts, input_id_list, output_id_list, output_texts,
                     kwargs) = self.chat(qs_,
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
                    if k == 0:
                        print(output_texts)
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
                        log_str = f'mode:{decoding_mode} step:{k + 1} ' \
                                  f'decoding:{decoding_length}/{branch_length} bs:{batch_size} ' \
                                  f'elapse:{elapse:.1f}s in:{avg_in_token:.1f} out:{avg_out_token:.1f} ' \
                                  f'edl:{edl:.3f}/{dl:.3f}/{pt:.3f}/{gt:.3f}/{qt:.3f} speed:{speed:.1f}token/s ' \
                                  f'All prefill time: {sum(pts):.3f}s All decoding time: {sum(gts):.3f}s ' \
                                  f'All retrieval time: {sum(qts):.3f}s All time :{elapse:.1f}s'
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
                log_str = f"mode:{decoding_mode} bs:{batch_size} " \
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