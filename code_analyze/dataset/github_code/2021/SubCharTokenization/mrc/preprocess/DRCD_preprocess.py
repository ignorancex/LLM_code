import collections
import copy
import json
import os

from tqdm import tqdm

from ..tools.langconv import Converter
from .utils import (
    _improve_answer_span,
    _check_is_max_context,
    _convert_examples_to_features,
)

SPIECE_UNDERLINE = '▁'

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def json2features(input_file, output_files, tokenizer, is_training=False, max_query_length=64,
                  max_seq_length=512, doc_stride=128):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
        train_data = train_data['data']

    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’' or c == '─' or c == ':':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = copy.deepcopy(para['context'])
            # 转简体
            context = Traditional2Simplified(context)
            # context中的中文前后加入空格
            context_chs = _tokenize_chinese_chars(context)
            context_fhs = _tokenize_chinese_chars(para['context'])

            doc_tokens = []
            ori_doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            for ic, c in enumerate(context_chs):
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                        ori_doc_tokens.append(context_fhs[ic])
                    else:
                        doc_tokens[-1] += c
                        ori_doc_tokens[-1] += context_fhs[ic]
                    prev_is_whitespace = False
                if c != SPIECE_UNDERLINE:
                    char_to_word_offset.append(len(doc_tokens) - 1)

            assert len(context_chs) == len(context_fhs)
            for qas in para['qas']:
                qid = qas['id']
                ques_text = Traditional2Simplified(qas['question'])
                ans_text = Traditional2Simplified(qas['answers'][0]['text'])
                start_position_final = None
                end_position_final = None

                if is_training:
                    start_position = qas['answers'][0]['answer_start']
                    end_position = start_position + len(ans_text) - 1

                    while context[start_position] == " " or context[start_position] == "\t" or \
                            context[start_position] == "\r" or context[start_position] == "\n":
                        start_position += 1

                    start_position_final = char_to_word_offset[start_position]
                    end_position_final = char_to_word_offset[end_position]

                    if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                        start_position_final += 1
                    actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                    cleaned_answer_text = "".join(whitespace_tokenize(ans_text))

                    if actual_text != cleaned_answer_text:
                        print(actual_text, 'V.S', cleaned_answer_text)
                        mis_match += 1

                examples.append({'doc_tokens': doc_tokens,
                                 'ori_doc_tokens': ori_doc_tokens,
                                 'orig_answer_text': ans_text,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    print('examples num:', len(examples))
    print('mis match:', mis_match)
    os.makedirs('/'.join(output_files[0].split('/')[0:-1]), exist_ok=True)
    json.dump(examples, open(output_files[0], 'w'))

    # to features
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']]  # 原来token到新token的映射，这是新token的起点
            if example['end_position'] < len(example['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': example_index,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1

    print('features num:', len(features))
    json.dump(features, open(output_files[1], 'w'))


def read_drcd_examples(input_file, is_training, convert_to_simplified, two_level_embeddings):
    with open(input_file, 'r') as f:
        train_data = json.load(f)
    train_data = train_data['data']

    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’' or c == '─' or c == ':':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c in [" ", "\t", "\r", "\n", SPIECE_UNDERLINE, '\u3000', '\u2009'] or ord(c) == 0x202F:
            return True
        return False

    # to examples
    examples = []
    mis_match = 0
    for article in tqdm(train_data):
        for para in article['paragraphs']:
            context = copy.deepcopy(para['context'])
            if two_level_embeddings:
                # Remove weird whitespace
                context = context.replace('\u200b', '')
                context = context.replace(u'\xa0', u'')
                # Adjust answer position accordingly
                for i, qas in enumerate(para['qas']):
                    ans_text = qas['answers'][0]['text']
                    ans_start = qas['answers'][0]['answer_start']
                    if ans_text != context[ans_start:ans_start + len(ans_text)]:
                        lo = None
                        for offset in range(-3, 4):
                            lo = ans_start + offset
                            if context[lo:lo+len(ans_text)] == ans_text:
                                break
                        para['qas'][i]['answers'][0]['answer_start'] = lo
            # 转简体
            if convert_to_simplified:
                context = Traditional2Simplified(context)
            # context中的中文前后加入空格
            context_chs = _tokenize_chinese_chars(context)
            doc_tokens = []
            # ori_doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context_chs:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                if c != SPIECE_UNDERLINE:
                    char_to_word_offset.append(len(doc_tokens) - 1)

            # Generate one example for each question
            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']
                if convert_to_simplified:
                    ques_text = Traditional2Simplified(ques_text)
                    ans_text = Traditional2Simplified(ans_text)
                start_position_final = None
                end_position_final = None

                # Get start and end position
                start_position = qas['answers'][0]['answer_start']
                end_position = start_position + len(ans_text) - 1

                while context[start_position] == " " or context[start_position] == "\t" or \
                        context[start_position] == "\r" or context[start_position] == "\n":
                    start_position += 1

                start_position_final = char_to_word_offset[start_position]
                end_position_final = char_to_word_offset[end_position]

                if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                    start_position_final += 1
                actual_text = "".join(doc_tokens[start_position_final:(end_position_final + 1)])
                cleaned_answer_text = "".join(whitespace_tokenize(ans_text))

                if actual_text != cleaned_answer_text:
                    print(actual_text, 'V.S', cleaned_answer_text)
                    mis_match += 1

                examples.append({'doc_tokens': doc_tokens,
                                 'orig_answer_text': ans_text,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    return examples, mis_match


def convert_examples_to_features(*args, **kwargs):
    return _convert_examples_to_features(*args, **kwargs)
