import random
import json
import re

from src.utilities.config_new import global_config as cfg


class RuleNLG:
    def __init__(self):
        with open('src/corpus/templates.json', 'rb') as f:
            self.templates = json.load(f)

    def give_response(self, act_num, slot_value, slot_mem):
        if act_num == cfg.sys_act_name2id['compare']:
            act_num = cfg.sys_act_name2id['implicit_confirm']
        # if act_num == len(cfg.sys_act_id2name) - 1:
        #     return act_num, slot_value, ''
        if act_num == cfg.sys_act_name2id['implicit_confirm'] and self.phone_rule(slot_mem) == 1:
            idx = random.randint(0, len(self.templates['system']['implicit_confirm_last']) - 1)
            sentence = re.sub('X', slot_value, self.templates['system']['implicit_confirm_last'][idx])
            return act_num, slot_value, sentence

        idx = random.randint(0, len(self.templates['system'][cfg.sys_act_id2name[act_num]]) - 1)
        if act_num == cfg.sys_act_name2id['implicit_confirm'] or act_num == cfg.sys_act_name2id['explicit_confirm']:
            sentence = re.sub('X', slot_value, self.templates['system'][cfg.sys_act_id2name[act_num]][idx])
        else:
            sentence = self.templates['system'][cfg.sys_act_id2name[act_num]][idx]
        return act_num, slot_value, sentence

    def phone_rule(self, slot_mem):
        phone_num = ''.join(slot for slot in slot_mem)
        if (len(phone_num)) < 11:
            return 0
        elif len(phone_num) == 11:
            return 1
        else:
            return -1