import json
import random
import numpy as np
import re
from datetime import datetime, timedelta
from src.utilities import config
from src.utilities.util import ed


class Simulator(object):
    def __init__(self, max_turn, name, type='phone', version_name='all'):
        self.max_turn = max_turn
        self.name = name
        self.type = type
        if self.type == 'id_number':
            with open('src/corpus/area.json', 'r', encoding='utf-8') as f:
                self.area_list = json.load(f)
        elif self.type == 'phone':
            self.turn2 = 0
        if type == 'id_number':
            if version_name == 'all':
                templates_file = 'src/corpus/templates_idnumber.json'
            elif version_name == 'train':
                templates_file = 'src/corpus/train_templates_id.json'
            elif version_name == 'dev':
                templates_file = 'src/corpus/dev_templates_id.json'
            elif version_name == 'test':
                templates_file = 'src/corpus/test_templates_id.json'
        else:
            if version_name == 'all':
                templates_file = 'src/corpus/templates.json'
            elif version_name == 'train':
                templates_file = 'src/corpus/train_templates.json'
            elif version_name == 'dev':
                templates_file = 'src/corpus/dev_templates.json'
            elif version_name == 'test':
                templates_file = 'src/corpus/test_templates.json'

        with open(templates_file, 'rb') as f:
            self.templates = json.load(f)
        self.idx_coll = None
        self.turn = 0
        self.turn2 = 0
        self.manager_act_space = len(config.sys_action_id2name)
        if self.name == 'all':
            self.act2semantic = config.intent_tag_id2name
            self.semantic2act = config.intent_tag_names
            self.act_space = len(self.semantic2act)
        else:
            self.act2semantic = ['offer', 'inform', 'affirm', 'deny', 'update', 'ack', 'finish', 'ask_state', 'bye', 'other']
            self.semantic2act = {w: i for i, w in enumerate(self.act2semantic)}
            self.act_space = len(self.semantic2act)
        self.goal_value = None
        self.pattern = None
        self.response_type = None
        self.slot_mem = []
        self.informed_value = []
        if type == 'phone':
            self.slot_error_prob = 0.4
        else:
            self.slot_error_prob = 0.2
        self.similar_map = {
            '8': ["吧"],
            '7': ['期', "启"],
            '4': ["是"],
            '1': ['幺']
        }

    def init_state(self):
        pass

    def generate_first_turn(self):
        pass

    def _generate_goal(self):
        if self.type == 'phone':
            self._generate_goal_phone()
        elif self.type == 'id_number':
            self._generate_goal_idnumber()
        else:
            pass

    def _generate_goal_idnumber(self):
        self.total_length = 18
        id_number = random.choice(self.area_list)
        start, end = datetime.strptime("1950-01-01", "%Y-%m-%d"), datetime.strptime("2005-12-30", "%Y-%m-%d")
        birth_days = datetime.strftime(start + timedelta(random.randint(0, (end - start).days + 1)), "%Y%m%d")
        id_number += str(birth_days)
        id_number += str(random.randint(10, 99))
        id_number += str(random.randrange(random.choice([0, 1]), 10, step=2))
        check_sum = 0
        for i in range(0, 17):
            check_sum += ((1 << (17 - i)) % 11) * int(id_number[i])
        check_digit = (12 - (check_sum % 11)) % 11
        check_digit = str(check_digit) if check_digit < 10 else 'X'
        id_number += check_digit

        pattern = ([6, 4, 4, 4], [4, 4, 4, 6], [5, 5, 5, 3], [4,6,4,4],[4,4,6,4],[3,3,4,4,4])
        p = np.array([0.3, 0.15, 0.15, 0.1, 0.15, 0.15])
        index = np.random.choice(len(pattern), p=p.ravel())
        self.pattern = pattern[index]
        self.goal_value = []
        start = 0
        for pos in self.pattern:
            self.goal_value.append(id_number[start: start + pos])
            start += pos

    def generate_fix_goal(self):
        self.total_length = 11
        self.pattern = [3,3,5]
        self.goal_value = ['188', '106', '22951']

    def _generate_goal_phone(self):
        self.total_length = 11
        prelist = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                   "147", "150", "151", "152", "153", "155", "156", "157", "158", "159",
                   "186", "187", "188", "189"]
        phone = random.choice(prelist) + "".join(random.choice("0123456789") for i in range(8))
        pattern = ([3, 4, 4], [4, 4, 3], [3, 3, 5], [5, 6], [5, 4, 2], [11])
        p = np.array([0.5, 0.199, 0.199, 0.1, 0.001, 0.001])
        index = np.random.choice(len(pattern), p=p.ravel())
        self.pattern = pattern[index]
        goal_value = []
        pos = 0
        if self.turn2 % 6 == 0:
            p = 3 + np.random.choice(3, p=np.array([0.6, 0.35, 0.05]).ravel())
            if p == 5:
                self.pattern = [3, 3, 5]
                for length in self.pattern[:-1]:
                    goal_value.append(phone[pos: pos + length])
                    pos += length
                goal_value.append(5 * str(random.randint(0, 9)))
            elif p == 4:
                flag_ = True
                for length in self.pattern:
                    if flag_ and pos != 0 and length >= 4:
                        tmp = 4 * str(random.randint(0, 9))
                        p = phone[pos:pos + (length - 4)] + tmp
                        goal_value.append(p)
                        flag_ = False
                    else:
                        goal_value.append(phone[pos: pos + length])
                    pos += length
            else:
                flag_ = True
                for length in self.pattern:
                    if flag_ and random.random() > 0.5 and pos != 0:
                        tmp = 3 * str(random.randint(0, 9))
                        p = phone[pos:pos + (length - 3)] + tmp
                        goal_value.append(p)
                        flag_ = False
                    else:
                        goal_value.append(phone[pos: pos + length])
                    pos += length
                if flag_:
                    tmp_i = -2 if len(self.pattern) > 2 else -1
                    value = goal_value[tmp_i]
                    len_ = len(value)
                    tmp = 3 * str(random.randint(0, 9))
                    value = value[:(len_ - 3)] + tmp
                    goal_value[tmp_i] = value
        else:
            for length in self.pattern:
                goal_value.append(phone[pos: pos + length])
                pos += length
        self.turn2 += 1
        if self.turn2 % 400 == 0:
            goal_value = [phone]
            self.pattern = [11]
        self.goal_value = goal_value

    def clear_sample(self):
        pass

    # Error Model
    def make_error(self, act_num, last_act, slot_value):
        """

        :param act_num:
        :param last_act:
        :param slot_value:
        :return: act_num, slot_value
        """
        # intent error
        if act_num == self.semantic2act['other']:
            act_num = random.sample(['ask_repeat', 'restart', 'bad_signal', 'wait'], 1)[0]
            act_num = config.intent_tag_names[act_num]
            slot_value = None

        # slot_error
        elif slot_value:
            slot_value = [i for i in slot_value]
            if random.random() < 0.3:
                # add new or delete one
                ind_ = random.randint(0, len(slot_value) - 1) if random.random() < 0.7 else len(slot_value) - 1
                num = random.randint(0, 9)
                if random.random() < 0.7:
                    slot_value.pop(ind_)
                else:
                    slot_value.insert(ind_, str(num))
            else:
                if len(slot_value) == 3:
                    slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                elif len(slot_value) in [4, 5]:
                    if random.random() < 0.7:
                        slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                    else:
                        slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                        slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                else:
                    slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                    slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
                    slot_value[random.randint(0, len(slot_value) - 1)] = str(random.randint(0, 9))
            slot_value = ''.join(slot_value)
        return act_num, slot_value

    def process_slot(self, slot_value):
        result = []
        if len(slot_value) == 0:
            return result
        else:
            if len(self.slot_mem) in [0, 1] or self.informed_value[1] == '0':
                result.append((slot_value, 0))
                return result
            for ind, v in enumerate(self.informed_value):
                if v in slot_value and v != '0':
                    result.append((v, ind))
                else:
                    break
            if len(''.join(v[0] for v in result)) != len(slot_value):
                result = []
                if len(slot_value) < 6:
                    result.append((slot_value, len(self.slot_mem)-1))
                elif len(slot_value) < 10:
                    if self.informed_value[0] == slot_value[len(self.informed_value[0]):]:
                        result.append((self.informed_value[0], 0))
                        result.append((slot_value[len(self.informed_value[0]):], 1))
                    elif len(self.informed_value) > 2 and self.informed_value[1] == slot_value[:len(self.informed_value[1])]:
                        result.append((slot_value[:len(self.informed_value[1])], 0))
                        result.append((self.informed_value[1], 1))
                if len(result) == 0:
                    start = 0
                    for ind, v in enumerate(self.informed_value):
                        if start + len(v) < len(slot_value) and v != '0':
                            if ind == len(self.informed_value)-1:
                                result.append((slot_value[start:], ind))
                            else:
                                result.append((slot_value[start: start+len(v)], ind))
                        else:
                            result.append((slot_value[start:], ind))
                            break
                        start += len(v)
            return result
    # NLG
    def give_response(self, semantic, slot_value, inform_turn, fake_slot_value=None, wrong_list=None, compare=False):
        tmp_value = None
        multi_inform_tag = None
        slot_tag = []
        kind = 'normal'

        if semantic in ['choose_first', 'choose_last']:
            tmp_value = ''.join(v[0] for v in slot_value)
            sen_intent_tags = config.intent_tag_names['inform']
            idx = random.randint(0, len(self.templates['user'][semantic]) - 1)
            template = self.templates['user'][semantic][idx]
            sentence = re.sub('X', tmp_value, template)
            slot_tag = self.bio_tag(template, tmp_value, None)
            kind = 'compare'
        elif semantic not in ['update', 'inform', 'restart', 'inform_all']:
            sen_intent_tags = config.intent_tag_names[semantic]
            idx = random.randint(0, len(self.templates['user'][semantic]) - 1)
            sentence = self.templates['user'][semantic][idx]
            slot_tag = [config.slot_tag_names['O']] * len(sentence)
            if semantic in ['how_signal', 'good_signal', 'bad_signal']:
                kind = 'signal'
            else:
                kind = 'normal'
        elif semantic == 'restart':
            sen_intent_tags = config.intent_tag_names[semantic]
            kind = 'restart'
            if slot_value:
                if type(slot_value) == tuple:
                    value = slot_value[0]
                else:
                    value = ''.join(v[0] for v in slot_value)
                idx = random.randint(0, len(self.templates['user']['restart_inform']) - 1)
                template = self.templates['user']['restart_inform'][idx]
                sentence = re.sub('X', value, template)
                slot_tag = self.bio_tag(template, value, None)
            else:
                idx = random.randint(0, len(self.templates['user']['restart']) - 1)
                template = self.templates['user']['restart'][idx]
                sentence = template
                slot_tag = self.bio_tag(template, None, None)

        elif semantic == 'inform_all':
            sen_intent_tags = config.intent_tag_names['inform']
            idx = random.randint(0, len(self.templates['user']['inform_all']) - 1)
            template = self.templates['user']['inform_all'][idx]
            sentence = re.sub('X', ''.join(v[0] for v in slot_value), template)
            slot_tag = self.bio_tag(template, ''.join(v[0] for v in slot_value), None)
            kind = 'inform_all'
        else:
            sen_intent_tags = config.intent_tag_names[semantic]
            if type(slot_value) == tuple:
                tmp_value, sub_ = self.sub_nn(slot_value[0])
                slot_value = (tmp_value, slot_value[1])
            else:
                tmp_value = ''.join(v[0] for v in slot_value)
                tmp_value, sub_ = self.sub_nn(tmp_value)
            if semantic == 'update':
                kind = 'update'
                if compare:
                    value = self.goal_value[slot_value[0][1]] + self.goal_value[slot_value[1][1]]
                    idx = random.randint(0, len(self.templates['user']['update']) - 1)
                    template = self.templates['user']['update'][idx]
                    sentence = re.sub('X', value, template)
                    slot_tag = self.bio_tag(template, value, None)
                    kind = 'compare'
                elif fake_slot_value:
                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                    template = self.templates['user']['update_special'][idx]
                    kind = 'update_special'
                    if len(wrong_list) != 1:
                        if random.random() < 0.9:
                            while 'Y' not in template:
                                idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                template = self.templates['user']['update_special'][idx]
                            value = ''.join(v[0] for v in slot_value)
                            sentence = re.sub('X', value, template)
                            sentence = re.sub('Y', fake_slot_value, sentence)
                            slot_tag = self.bio_tag(template, value, fake_slot_value)
                            kind = 'update_special'
                        else:
                            sentence = ''
                            slot_tag = []
                            for value, index in wrong_list:
                                if index == 0:
                                    action = 'update_front'
                                elif index == len(self.pattern) - 1:
                                    action = 'update_last'
                                else:
                                    action = 'update_middle'
                                idx = random.randint(0, len(self.templates['user'][action]) - 1)
                                template = self.templates['user'][action][idx]
                                sentence_ = re.sub('X', self.goal_value[index], template)
                                sentence_ = re.sub('Y', value, sentence_)
                                sentence_ = re.sub('N', str(len(value)), sentence_)
                                slot_tag += [1] + self.bio_tag(template, self.goal_value[index], value).copy()
                                sentence += '，' + sentence_
                                kind = 'update_two_part'
                            sentence = sentence[1:]
                            slot_tag = slot_tag[1:]
                    elif sub_:
                        if type(slot_value) == list:
                            value = ''.join(v[0] for v in slot_value)
                        else:
                            value = slot_value[0]
                        sentence = re.sub('X', value, template)
                        sentence = re.sub('Y', fake_slot_value, sentence)
                        slot_tag = self.bio_tag(template, value, fake_slot_value)
                        kind = 'update_jige'
                    else:
                        if tmp_value:
                            if (len(fake_slot_value) == len(tmp_value) + 1 or len(wrong_list[0][0]) == len(tmp_value) + 1) and random.random() < 0.5:
                                if len(wrong_list[0][0]) == len(tmp_value) + 1:
                                    diff_value, pos = self.diff_one(wrong_list[0][0], tmp_value)
                                else:
                                    diff_value, pos = self.diff_one(fake_slot_value, tmp_value)
                                if len(re.findall(diff_value, fake_slot_value)) < 2:
                                    idx = random.randint(0, len(self.templates['user']['negate']) - 1)
                                    template = self.templates['user']['negate'][idx]
                                    sentence = re.sub('X', diff_value, template)
                                    slot_tag = self.bio_tag(template, diff_value, None)
                                    kind = 'negate'
                                else:
                                    count = 1
                                    for i in range(pos):
                                        if fake_slot_value[i] == diff_value:
                                            count += 1
                                    idx = random.randint(0, len(self.templates['user']['negate_n']) - 1)
                                    template = self.templates['user']['negate_n'][idx]
                                    sentence = re.sub('X', diff_value, template)
                                    sentence = re.sub('N', str(count), sentence)
                                    slot_tag = self.bio_tag(template, diff_value, fake_slot_value)
                                    kind = 'negate_n'
                            elif (len(fake_slot_value) == len(tmp_value) - 1 or len(wrong_list[0][0]) == len(tmp_value)-1) and random.random() < 0.5:
                                if len(wrong_list[0][0]) == len(tmp_value)-1:
                                    diff_value, pos = self.diff_one(tmp_value, wrong_list[0][0])
                                else:
                                    diff_value, pos = self.diff_one(tmp_value, fake_slot_value)
                                if pos == len(tmp_value) - 1 and len(fake_slot_value) == len(tmp_value) - 1:
                                    idx = random.randint(0, len(self.templates['user']['last_miss']) - 1)
                                    template = self.templates['user']['last_miss'][idx]
                                    sentence = re.sub('X', tmp_value[-1], template)
                                    slot_tag = self.bio_tag(template, tmp_value[-1], None)
                                    kind = 'last_miss'
                                elif random.random() < 0.5 and pos != 0:
                                    idx = random.randint(0, len(self.templates['user']['after_miss']) - 1)
                                    template = self.templates['user']['after_miss'][idx]
                                    i = 1
                                    while pos - i >= 0 and len(re.findall(tmp_value[pos - i:pos], fake_slot_value)) != 1:
                                        i += 1
                                    if pos - i < 0:
                                        idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                        template = self.templates['user']['update_special'][idx]
                                        sentence = re.sub('X', tmp_value, template)
                                        sentence = re.sub('Y', wrong_list[0][0], sentence)
                                        slot_tag = self.bio_tag(template, tmp_value, wrong_list[0][0])
                                        kind = 'update_special'
                                    else:
                                        sentence = re.sub('X', diff_value, template)
                                        sentence = re.sub('Y', tmp_value[pos - i:pos], sentence)
                                        slot_tag = self.bio_tag(template, diff_value, tmp_value[pos - i:pos])
                                        kind = 'after_miss'
                                else:
                                    idx = random.randint(0, len(self.templates['user']['before_miss']) - 1)
                                    template = self.templates['user']['before_miss'][idx]
                                    i = 2
                                    while pos + i < len(tmp_value) and len(re.findall(tmp_value[pos + 1: pos + i], fake_slot_value)) != 1:
                                        i += 1
                                    if pos + i >= len(tmp_value):
                                        idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                        template = self.templates['user']['update_special'][idx]
                                        sentence = re.sub('X', tmp_value, template)
                                        sentence = re.sub('Y', wrong_list[0][0], sentence)
                                        slot_tag = self.bio_tag(template, tmp_value,  wrong_list[0][0])
                                        kind = 'update_special'
                                    else:
                                        sentence = re.sub('X', diff_value, template)
                                        sentence = re.sub('Y', tmp_value[pos + 1: pos + i], sentence)
                                        slot_tag = self.bio_tag(template, diff_value, tmp_value[pos + 1:pos + i])
                                        kind = 'before_miss'
                            elif (len(wrong_list[0][0]) == len(self.goal_value[wrong_list[0][1]]) > 9 or \
                                  (len(wrong_list[0][0]) == len(self.goal_value[wrong_list[0][1]]) > 3 and random.random() < 0.7)):
                                fake_slot_value, value = self.sub_update(wrong_list[0][0], self.goal_value[wrong_list[0][1]])
                                while 'Y' not in template:
                                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                    template = self.templates['user']['update_special'][idx]
                                sentence = re.sub('X', value, template)
                                sentence = re.sub('Y', fake_slot_value, sentence)
                                slot_tag = self.bio_tag(template, value, fake_slot_value)
                                kind = 'sub_update'
                            elif len(fake_slot_value) > 5 and (len(fake_slot_value)==len(tmp_value) or len(wrong_list[0][0])==len(tmp_value)) and random.random() < 0.2:
                                if wrong_list[0][1] == 0:
                                    action = 'update_front'
                                elif wrong_list[0][1] == len(self.pattern) - 1:
                                    action = 'update_last'
                                else:
                                    action = 'update_middle'
                                idx = random.randint(0, len(self.templates['user'][action]) - 1)
                                template = self.templates['user'][action][idx]
                                sentence = re.sub('X', self.goal_value[wrong_list[0][1]], template)
                                sentence = re.sub('Y', wrong_list[0][0], sentence)
                                sentence = re.sub('N', str(len(wrong_list[0][0])), sentence)
                                slot_tag = self.bio_tag(template, self.goal_value[wrong_list[0][1]], wrong_list[0][0])
                                kind = 'update_sure'
                            elif self.diff_count(wrong_list[0][0], self.goal_value[wrong_list[0][1]]) > 2:
                                while 'Y' not in template:
                                    idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                    template = self.templates['user']['update_special'][idx]
                                sentence = re.sub('X', self.goal_value[wrong_list[0][1]], template)
                                sentence = re.sub('Y', wrong_list[0][0], sentence)
                                slot_tag = self.bio_tag(template, self.goal_value[wrong_list[0][1]], wrong_list[0][0])
                                kind = 'update_special'
                            else:
                                if len(fake_slot_value) < 6:
                                    idx = random.randint(0, len(self.templates['user']['update']) - 1)
                                    template = self.templates['user']['update'][idx]
                                    kind = 'update_normal'
                                sentence = re.sub('X', tmp_value, template)
                                sentence = re.sub('Y', wrong_list[0][0], sentence)
                                slot_tag = self.bio_tag(template, tmp_value, wrong_list[0][0])
                        else:
                            if len(wrong_list) != 1:
                                raise Exception('what???')
                            if len(wrong_list[0][0]) == len(self.goal_value[wrong_list[0][1]]) > 9:
                                fake_slot_value, value = self.sub_update(wrong_list[0][0], self.goal_value[wrong_list[0][1]])
                                kind = 'sub_update'
                            elif len(wrong_list[0][0]) == len(self.goal_value[wrong_list[0][1]]) > 3 and random.random() < 0.9:
                                fake_slot_value, value = self.sub_update(wrong_list[0][0], self.goal_value[wrong_list[0][1]])
                                kind = 'sub_update'
                            else:
                                value = slot_value[0]
                                kind = 'update_special'
                            idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                            template = self.templates['user']['update_special'][idx]
                            while 'Y' not in template:
                                idx = random.randint(0, len(self.templates['user']['update_special']) - 1)
                                template = self.templates['user']['update_special'][idx]
                            sentence = re.sub('X', value, template)
                            sentence = re.sub('Y', fake_slot_value, sentence)
                            slot_tag = self.bio_tag(template, value, fake_slot_value)


                else:
                    value = slot_value[0] if type(slot_value) == tuple else ''.join(v[0] for v in slot_value)
                    idx = random.randint(0, len(self.templates['user']['update']) - 1)
                    template = self.templates['user']['update'][idx]
                    sentence = re.sub('X', value, template)
                    slot_tag = self.bio_tag(template, value, None)
                    kind = 'update_normal'
            elif semantic == 'inform':
                kind = 'inform'
                if type(slot_value) == tuple:
                    value = slot_value[0]
                else:
                    value = ''.join(v[0] for v in slot_value)
                if inform_turn == 0:
                    action = 'inform_start'
                elif inform_turn == 1:
                    action = 'inform'
                else:
                    action = 'inform_last'

                idx = random.randint(0, len(self.templates['user'][action]) - 1)
                template = self.templates['user'][action][idx]
                if sub_:
                    sentence = re.sub('X', value, template)
                    slot_tag = self.bio_tag(template, value, None)
                    kind = 'inform_jige'
                else:
                    p = random.random()
                    if inform_turn != 0 and p < 0.15:
                        # 188, 88106
                        last_value = self.goal_value[slot_value[1] - 1]
                        value = last_value[random.randint(1, len(last_value) - 1):] + value
                        multi_inform_tag = (value, slot_value[1])
                        kind = 'multi_inform'
                        sentence = re.sub('X', value, template)
                        slot_tag = self.bio_tag(template, value, None)
                    elif p < 0.25:
                        # 17, 171
                        while len(re.findall('X', template)) != 1:
                            idx = random.randint(0, len(self.templates['user'][action]) - 1)
                            template = self.templates['user'][action][idx]
                        value = value[:random.randint(1, len(value) - 1)] + '，' + value
                        kind = 'tone_inform'
                        sentence = re.sub('X', value, template)
                        slot_tag = self.bio_tag(template, value, None)
                    elif p < 0.35:
                        # inform_update
                        _, fake_value = self.make_error(0, 0, value)
                        action = 'inform_update'
                        idx = random.randint(0, len(self.templates['user'][action]) - 1)
                        template = self.templates['user'][action][idx]
                        sentence = re.sub('X', value, template)
                        sentence = re.sub('Y', fake_value, sentence)
                        slot_tag = self.bio_tag(template, value, fake_value)
                        kind = 'inform_update'
                    else:
                        if random.random() < 0.1 and value[-1] in ['7', '8']:
                            value = value[:-1] + self.similar_map[value[-1]][0]
                            kind = 'asr_num_to_han'
                        if random.random() < 0.1 and '1' in value:
                            value = re.sub('1', self.similar_map['1'][0], value)
                            kind = 'asr_num_to_han'
                        if kind == 'inform' and len(re.findall('X', template)) != 1:
                            kind = '2X'
                        sentence = re.sub('X', value, template)
                        slot_tag = self.bio_tag(template, value, None)
        return sentence, slot_tag, sen_intent_tags, [multi_inform_tag, kind]

    def bio_tag(self, template, slot_value, fake_slot_value):
        char_label = []
        flag = True
        _flag = True
        for c in template:
            if c != 'X' and c != 'Y':
                char_label.append('O')
            elif c == 'Y':
                for v in fake_slot_value:
                    char_label.append('O')
            else:
                if _flag:
                    _flag = False
                    for v in slot_value:
                        if flag:
                            char_label.append('B-NUM')
                            flag = False
                        else:
                            char_label.append('I-NUM')
                else:
                    for v in slot_value:
                        char_label.append('O')
        sen_slot_tags = [config.slot_tag_names[l] for l in char_label]
        return sen_slot_tags

    def sub_nn(self, slot_str):
        sub_ = False
        for i in range(10):
            pattern = re.compile(str(i) + '{3,}')
            res = pattern.search(slot_str)
            if res is not None:
                len_ = len(res.group())
                if len_ == 5:
                    slot_str = (re.sub(pattern, f'，5个{i}，', slot_str))
                    sub_ = True
                elif len_ == 4:
                    if random.random() < 0.7:
                        slot_str = (re.sub(pattern, f'，4个{i}，', slot_str))
                        sub_ = True
                elif len_ == 3:
                    if random.random() < 0.3:
                        slot_str = (re.sub(pattern, f'，3个{i}，', slot_str))
                        sub_ = True
                break
        return slot_str, sub_

    def sub_update(self, slot_value, update_value):
        if len(slot_value) == len(update_value):
            if len(update_value) == 4:
                if slot_value[:1] == update_value[:1]:
                    return slot_value[1:], update_value[1:]
                elif slot_value[-1:] == update_value[-1:]:
                    return slot_value[:-1], update_value[:-1]
            elif len(update_value) == 5:
                r_ = random.random()
                if slot_value[:2] == update_value[:2]:
                    if r_ < 0.5:
                        return slot_value[1:], update_value[1:]
                    else:
                        return slot_value[2:], update_value[2:]
                elif slot_value[-2:] == update_value[-2:]:
                    if r_ < 0.5:
                        return slot_value[:-1], update_value[:-1]
                    else:
                        return slot_value[:-2], update_value[:-2]
            elif len(update_value) > 5:
                if slot_value[4:] == update_value[4:]:
                    return slot_value[:4], update_value[:4]
                elif slot_value[:-4] == update_value[:-4]:
                    return slot_value[-4:], update_value[-4:]
                else:
                    return slot_value, update_value
        if slot_value[0] == update_value[0]:
            return slot_value[1:], update_value[1:]
        elif slot_value[-1] == update_value[-1]:
            return slot_value[:-1], update_value[:-1]
        else:
            return slot_value, update_value

    def diff_one(self, value1, value2):
        assert len(value1) == len(value2) + 1
        flag = False
        diff = None
        pos = None
        for v1, v2, pos_ in zip(value1, value2, range(len(value1))):
            if v1 != v2:
                flag = True
                diff = v1
                pos = pos_
                break
        if flag:
            pass
        elif not flag and value1[:-1] == value2:
            diff = value1[-1]
            pos = len(value1) - 1
        else:
            raise Exception('can not find different in function diff_one')
        return diff, pos

    def diff_count(self, value1, value2):
        count = 0
        for v1, v2 in zip(value1, value2):
            if v1 != v2:
                count += 1
        return count

    def generator_slot_value(self):
        pos = len(self.slot_mem)
        slot_value = self.goal_value[pos] if pos < len(self.goal_value) else None
        return slot_value, pos

    def offer(self):
        self.act_semantic = 'offer'
        self.act_num = self.semantic2act['offer']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('offer', None, None)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def inform(self):
        informed_slot_value = self.generator_slot_value()
        # if informed_slot_value[0] == '0125':
        #     print('here')

        if len(self.slot_mem) == len(self.goal_value) - 1:
            self.inform_turn = 2
        elif len(self.slot_mem) == 0:
            self.inform_turn = 0
        else:
            self.inform_turn = 1

        # state Transition
        self.slot_mem.append(informed_slot_value[0])
        self.act_semantic = 'inform'
        self.act_num = self.semantic2act['inform']

        # error model
        if random.random() < self.slot_error_prob and len(informed_slot_value[0]) > 2:
            _, tmp_value = self.make_error(1, self.last_act, informed_slot_value[0])
            informed_slot_value = (tmp_value, informed_slot_value[1])
        self.informed_value[informed_slot_value[1]] = informed_slot_value[0]

        # NLG
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('inform', informed_slot_value, self.inform_turn)

        if len(informed_slot_value[0]) >= self.total_length - 1:
            self.inform_turn = 2
        if len(self.slot_mem) == len(self.goal_value):
            self.inform_turn = 2

        return sentence, slot_tag, sen_intent_tags, [informed_slot_value], multi_inform_tag

    def inform_all(self):
        informed_slot_value = [(value, index) for index, value in enumerate(self.goal_value)]
        self.inform_turn = 2

        # state Transition
        self.slot_mem = []
        self.informed_value = ['0'] * len(self.pattern)
        for value, index in informed_slot_value:
            self.slot_mem.append(value)
        self.act_semantic = 'inform'
        self.act_num = self.semantic2act['inform']

        # error model
        if random.random() < 0.9:
            pos = random.choice([1, 2])
            slot_value = informed_slot_value[pos][0]
            slot_value = [i for i in slot_value]
            # add new or delete one
            ind_ = random.randint(0, len(slot_value) - 1) if random.random() < 0.7 else len(slot_value) - 1
            num = random.randint(0, 9)
            if random.random() < 0.5:
                slot_value.pop(ind_)
            else:
                slot_value.insert(ind_, str(num))
            slot_value = ''.join(slot_value)
            informed_slot_value[pos] = (slot_value, informed_slot_value[pos][1])
        for i, v in informed_slot_value:
            self.informed_value[v] = i

        # NLG
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('inform_all', informed_slot_value, self.inform_turn)

        return sentence, slot_tag, sen_intent_tags, informed_slot_value, multi_inform_tag

    def affirm(self):
        self.act_semantic = 'affirm'
        self.act_num = self.semantic2act['affirm']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('affirm', None, self.inform_turn)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def deny(self, slot_value, wrong_list):

        self.fake_slot_value = list(slot_value)
        self.fake_slot_list = list(wrong_list)
        self.update_origin = wrong_list

        self.act_semantic = 'deny'
        self.act_num = self.semantic2act['deny']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('deny', None, self.inform_turn)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def update(self, slot_value, wrong_list, compare=False):
        if len(self.slot_mem) != 0:
            for v, ind in wrong_list:
                self.informed_value[ind] = self.goal_value[ind]
        # 两种情况，全部，和上一次的长度
        if compare:
            informed_slot_value = [(self.goal_value[index], index) for value, index in slot_value]
            self.update_origin = ''.join(v[0] for v in slot_value)
            sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('update', informed_slot_value, None, compare=True)
            return sentence, slot_tag, sen_intent_tags, informed_slot_value, multi_inform_tag
        if len(wrong_list) != 1:
            informed_slot_value = [(value, index) for index, value in enumerate(self.slot_mem)]
            self.update_origin = ''.join(v[0] for v in slot_value)
        else:
            if len(''.join(v[0] for v in slot_value)) >= self.total_length - 1:
                # 直接报全部
                if random.random() < 0.3 and len(self.slot_mem) != 0:
                    informed_slot_value = [(value, index) for index, value in enumerate(self.slot_mem)]
                    self.update_origin = ''.join(v[0] for v in slot_value)
                else:
                    # 报错误的部分
                    informed_slot_value = (self.goal_value[wrong_list[0][1]], wrong_list[0][1])
                    self.update_origin = ''.join(v[0] for v in slot_value)
            else:
                informed_slot_value = (self.goal_value[wrong_list[0][1]], wrong_list[0][1])
                self.update_origin = ''.join(v[0] for v in slot_value)

        self.act_semantic = 'update'
        self.act_num = self.semantic2act['update']

        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('update', informed_slot_value, self.inform_turn, self.update_origin, wrong_list)
        if type(informed_slot_value) == tuple:
            informed_slot_value = [informed_slot_value]
        return sentence, slot_tag, sen_intent_tags, informed_slot_value, multi_inform_tag

    def ack(self):
        self.act_semantic = 'ack'
        self.act_num = self.semantic2act['ack']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('ack', None, self.inform_turn)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def restart(self):
        self.act_semantic = 'restart'
        self.slot_mem = []
        self.informed_value = ['0'] * len(self.pattern)
        informed_slot_value = None
        self.inform_turn = 0
        if random.random() < 0.5:
            informed_slot_value = self.generator_slot_value()
            self.slot_mem.append(informed_slot_value[0])
            self.inform_turn = 1
            if len(self.slot_mem) == len(self.goal_value):
                self.inform_turn = 2
            self.informed_value[informed_slot_value[1]] = informed_slot_value[0]
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('restart', informed_slot_value, None)
        if informed_slot_value:
            informed_slot_value = [informed_slot_value]
        else:
            informed_slot_value = []
        return sentence, slot_tag, sen_intent_tags, informed_slot_value, multi_inform_tag

    def finish(self):
        self.act_semantic = 'finish'
        self.act_num = self.semantic2act['finish']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('finish', None, None)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def ask_state(self):
        self.act_semantic = 'ask_state'
        self.act_num = self.semantic2act['ask_state']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('ask_state', None, None)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def bye(self):
        self.act_semantic = 'bye'
        self.act_num = self.semantic2act['bye']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('bye', None, None)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def wait(self):
        self.act_semantic = 'wait'
        self.act_num = self.semantic2act['other']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('wait', None, self.inform_turn)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    def null(self):
        self.act_semantic = 'other'
        self.act_num = config.intent_tag_names['other']
        sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('other', None, None)
        return sentence, slot_tag, sen_intent_tags, [], multi_inform_tag

    # @staticmethod
    # def simulator_reward(terminal, adj, turn):
    #     if
    # @staticmethod
    # def global_reward(terminal, success):
    #     if not terminal:
    #         return -0.01
    #     elif success:
    #         return 1.0
    #     else:
    #         return -1.0
    #
    # @staticmethod
    # def simulator_reward(terminal, adj, shaping_reward):
    #     r = 0.01
    #     if not terminal:
    #         if adj == 1.0:
    #             r = shaping_reward
    #         elif adj < 0:
    #             r = shaping_reward*adj
    #         # elif adj == -20:
    #         #     r = -0.02*20
    #     return r
    #
    # @staticmethod
    # def manager_reward(terminal, adj, shaping_reward):
    #     r = 0
    #     if not terminal:
    #         if adj == 1.0:
    #             r = 0.01*adj
    #         elif adj < 0:
    #             r = 0.02*adj
    #     return r
