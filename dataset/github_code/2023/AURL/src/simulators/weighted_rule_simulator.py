import random
import re
import numpy as np

from src.simulators.simulator import Simulator
from src.utilities import config, util
from src.utilities.config_new import global_config as cfg
from src.dataLoader.user_dataset import get_user_act



class WeightedRuleSimulator(Simulator):
    def __init__(self, max_turn, name, type='phone', version_name='all'):
        """ initialize the Simulator

        :param templates_file: templates file
        :param slot_num: the max turns
        :param max_turn: the number of slots
        :param name: the name of simulator
        """
        super(WeightedRuleSimulator, self).__init__(max_turn, name, type, version_name)
        self.trainable = False
        self.act_num = None
        self.type = type
        self.act2semantic = ['offer', 'inform', 'affirm', 'deny', 'update', 'ack', 'finish', 'ask_state', 'bye', 'other']
        self.semantic2act = {w: i for i, w in enumerate(self.act2semantic)}
        self.multi_inform_prob = 0.1
        self.sub_update_prob = 0.1

    @staticmethod
    def _normal(mu=4, sigma=0.5):
        return int(np.random.normal(mu, sigma) + 0.5)

    def init_state(self):
        """
        initialize the environment
        :param idx:
        :return: sentence, sentence tags for slot filling and intent recognition
        """
        self.idx_coll = None
        # generate intent slots value
        self._generate_goal()
        # self.generate_fix_goal()
        # clear state
        self.slot_mem = []
        self.turn = 0
        self.inform_turn = 0
        self.manager_act = None
        self.last_act = None
        self.informed_value = ['0'] * len(self.pattern)
        self.fake_slot_value = None
        self.clear_sample()
        self.adj = 0
        self.last_slot = []
        if self.type == 'phone':
            self.slot_error_prob = 0.4
        else:
            self.slot_error_prob = 0.4

    def clear_sample(self):
        # clear state
        self.slot_mem = []
        self.turn = 0
        self.inform_turn = 0
        self.manager_act = None
        self.last_act = None
        self.fake_slot_value = None
        self.last_slot = []
        self.informed_value = ['0'] * len(self.pattern)

    def step(self, response, manager_act, slot_value, test=False):
        if isinstance(slot_value, str):
            slot_value = self.process_slot(slot_value)
        else:
        # if slot_value and slot_value[-1][1] >= len(self.goal_value):
            slot_value = self.process_slot(''.join(v[0] for v in slot_value))
        if manager_act == cfg.sys_act_name2id['ask_restart']:
            self.slot_error_prob = 0.02
        self.turn += 1
        assert 0 <= manager_act < self.manager_act_space
        user_slot_value = []
        # intent error
        if random.random() < 0.001 and manager_act < cfg.sys_act_name2id['bye']:
            act_num, _ = self.make_error(9, self.last_act, None)
            self.act_num = act_num
            self.act_semantic = list(config.intent_tag_names.keys())[act_num]
            if self.act_semantic == 'restart':
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.restart()
            else:
                sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response(self.act_semantic, None, None)
            return sentence, cfg.user_act_name2id[get_user_act(config.intent_tag_id2name[sen_intent_tags], multi_inform_tag[1])], user_slot_value

        # manager acts is request
        if manager_act == cfg.sys_act_name2id['request']:
            if len(self.slot_mem) == 0:
                temp = random.random()
                if temp > 0.1:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform()
                elif temp < 0.05:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.offer()
                elif temp < 0.09:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()
                else:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.wait()
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        # manager act is continue
        elif manager_act == cfg.sys_act_name2id['continue']:
            if self.inform_turn == 2 or len(self.slot_mem) == len(self.goal_value):
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.finish()
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform()

        # manager act is reqmore
        elif manager_act == cfg.sys_act_name2id['req_more']:
            if self.inform_turn != 2:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform()
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.finish()

        # manager act is Implicit-confirm
        elif manager_act == cfg.sys_act_name2id['implicit_confirm']:
            if len(slot_value) > 0:
                wrong_list = self.check_slot_correct(slot_value)
                # number is correct
                if len(wrong_list) == 0:
                    # provided complete
                    if self.inform_turn == 2:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.affirm()
                    # continue providing
                    else:
                        if random.random() > 0.1:
                            sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform()
                        else:
                            sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.affirm()

                # the number is wrong
                else:
                    if self.last_act == config.intent_tag_names['ask_state']:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.restart()
                    elif random.random() > 0.3:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.update(slot_value, wrong_list)
                    else:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.deny(slot_value, wrong_list)
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        # manager act is explicit-confirm
        elif manager_act == cfg.sys_act_name2id['explicit_confirm'] or manager_act == cfg.sys_act_name2id['compare']:
            if len(slot_value) > 0:
                wrong_list = self.check_slot_correct(slot_value)
                # affirm
                if len(wrong_list) == 0:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.affirm()
                else:
                    if random.random() > 0.1:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.update(slot_value, wrong_list)
                    else:
                        sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.deny(slot_value, wrong_list)
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        # manager act is ack
        elif manager_act == cfg.sys_act_name2id['ack']:
            if self.inform_turn in [0, 1]:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform()
            elif self.inform_turn == 2:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        # manager act is req_correct
        elif manager_act == cfg.sys_act_name2id['req_correct']:
            if len(self.slot_mem) == self.total_length:
                p = random.random()
                if p < 0.6 and self.fake_slot_value:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.update(self.fake_slot_value, self.fake_slot_list)
                elif p < 0.7:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.restart()
                else:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.inform_all()
            else:
                if self.fake_slot_value and len(self.slot_mem) != 0:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.update(self.fake_slot_value, self.fake_slot_list)
                else:
                    sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        elif manager_act == cfg.sys_act_name2id['compare']:
            # [(((188, 0), (106, 1)), ((188, 0), (88106, 1)), 0)]
            value = slot_value[0][0]
            wrong_list = self.check_slot_correct(value)
            if len(wrong_list) == 0:
                self.act_semantic = 'inform'
                self.act_num = self.semantic2act['inform']

                user_slot_value = list(slot_value[0][0])
                sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('choose_first', slot_value[0][0], None)
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.update(value, wrong_list, True)
        # bye
        elif manager_act == cfg.sys_act_name2id['bye']:
            sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.bye()
        elif manager_act == cfg.sys_act_name2id['ask_restart']:
            # self.slot_error_prob = 0.1
            p = random.random()
            if p < 0.5:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.restart()
            elif p < 0.65:
                self.slot_mem = []
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ask_state()
        else:
            # 重复，good_signal -> 重复上一次动作
            if manager_act == cfg.sys_act_name2id['good_signal']:
                self.act_semantic = 'ask_repeat'
                self.act_num = config.intent_tag_names['ask_repeat']
                sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response(self.act_semantic, None, None)

            # how_signal
            elif manager_act == cfg.sys_act_name2id['how_signal']:
                act_semantic = 'how_signal' if random.random() < 0.2 else 'good_signal'
                self.act_semantic = 'other'
                self.act_num = self.semantic2act[self.act_semantic]
                sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response(act_semantic, None, None)
            # bad_signal
            elif manager_act == cfg.sys_act_name2id['bad_signal']:
                self.act_semantic = 'other'
                self.act_num = self.semantic2act[self.act_semantic]
                sentence, slot_tag, sen_intent_tags, multi_inform_tag = self.give_response('how_signal', None, None)
            # null:
            else:
                sentence, slot_tag, sen_intent_tags, user_slot_value, multi_inform_tag = self.ack()

        if 0 <= self.act_num < config.intent_tag_names['bye']:
            self.last_act = self.act_num
            self.last_slot = user_slot_value
        if len(self.slot_mem) == len(self.goal_value):
            self.inform_turn = 2

        return sentence, cfg.user_act_name2id[get_user_act(config.intent_tag_id2name[sen_intent_tags], multi_inform_tag[1])], user_slot_value

    def multi_inform(self, slot_value):
        assert len(self.slot_mem) > 1
        add_value = self.slot_mem[-2]
        pos = random.randint(1, len(add_value))
        slot_value = add_value[-pos:] + slot_value
        return slot_value

    def check_slot_correct(self, slot_value):
        wrong = []
        for value, index in slot_value:
            if index < len(self.goal_value):
                if value != self.goal_value[index]: wrong.append((value, index))
            else:
                if wrong[-1][1] == len(self.goal_value) - 1:
                    tmp = wrong.pop()
                    tmp = (tmp[0] + ''.join(s[0] for s in slot_value[index:]), tmp[1])
                    wrong.append(tmp)
                else:
                    wrong.append((''.join(s[0] for s in slot_value[index:]), len(self.goal_value) - 1))
        return wrong

    def Rule_Tel(self, phone):
        reg = "1[3|4|5|7|8][0-9]{9}"
        if re.fullmatch(reg, phone) is not None:
            return True
        else:
            return False

    def terminal(self, manager_act, collected_slot, turn):
        terminal = False
        success = False
        if manager_act == cfg.sys_act_name2id['bye'] and ''.join(collected_slot) == ''.join(self.goal_value):
            terminal = True
            success = True
            return terminal, success
        elif manager_act == cfg.sys_act_name2id['bye'] or turn > 20:
            terminal = True
            success = False
            return terminal, success
        else:
            return terminal, success

    def terminal_test(self, manager_act, collected_slot, turn):
        terminal = False
        success = False
        if manager_act == cfg.sys_act_name2id['bye'] and ''.join(collected_slot) == ''.join(self.goal_value):
            terminal = True
            success = True
            return terminal, success
        elif turn > 20 or manager_act == cfg.sys_act_name2id['bye']:
            terminal = True
            success = False
            return terminal, success
        else:
            return terminal, success


if __name__ == '__main__':
    sim = WeightedRuleSimulator(cfg.max_turn, 'all')
    sim.init_state()
    sim.init_state()

    while 1:
        act = input('manager_act:')
        slot = input('slot:')
        utterance, _, intent_tag, user_slot, _ = sim.step(cfg.sys_act_name2id[act], slot)
        print(utterance, config.intent_tag_id2name[intent_tag], user_slot)
