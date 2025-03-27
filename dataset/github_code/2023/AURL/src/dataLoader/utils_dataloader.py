from src.utilities.util import *
from src.dataLoader.user_dataset import get_user_act
from src.utilities.config_new import global_config as cfg


class InputFeature(object):
    """A single set of feature of data"""
    def __init__(self,
                 user_utterance,
                 user_label,
                 user_state,
                 tag,
                 staff_utterance,
                 staff_label,
                 staff_state,
                 last_staff_state,
                 last_staff_label,
                 dialog_history,
                 goal):
        self.user_utterance = sub_ge(user_utterance)
        self.user_action, self.user_slot = self.convert_label_to_action_and_slot(user_label, user_state)
        self.user_action = get_user_act(user_label, tag) if tag else 'offer'
        self.user_state = user_state
        self.staff_utterance = staff_utterance
        self.staff_action, self.staff_slot = self.convert_label_to_action_and_slot(staff_label, staff_state)
        self.staff_state = staff_state
        self.staff_slot_idx = self.get_sys_index(staff_label, staff_state)
        self.last_staff_state = last_staff_state
        self.last_staff_action, self.last_staff_slot = self.convert_label_to_action_and_slot(last_staff_label, last_staff_state)
        self.dialog_history = dialog_history + user_utterance
        self.goal = goal
        self.confirm_id = self.get_confirm_id()
        self.reg_exp = self.get_re(''.join(last_staff_state))

    def getdict(self):
        return {
            "user_utterance": self.user_utterance,
            "user_action": self.user_action,
            "user_slot": self.user_slot,
            "staff_utterance": self.staff_utterance,
            "staff_action": self.staff_action,
            "staff_slot": self.staff_slot,
            "staff_state": self.staff_state,
            "staff_slot_idx": self.staff_slot_idx,
            "last_staff_state": self.last_staff_state,
            "last_staff_action": self.last_staff_action,
            "last_staff_slot": self.last_staff_slot,
            "dialog_history": self.dialog_history,
            "goal_value": ''.join(self.goal),
            "confirm_id": self.confirm_id,
            "reg_exp": self.reg_exp
        }

    @staticmethod
    def get_sys_index(label, staff_state):
        slot_index = [0] * len(staff_state)
        if '(' not in label:
            return slot_index
        action = re.findall(r'(.*)[(]', label)[0]
        if action != 'compare':
            slot = re.findall(r'[(](.*?)[)]', label)[0].replace(' ', '').split(',')
            if len(slot) == 1:
                try:
                    slot_index[staff_state.index(slot[0])] = 1
                except:
                    raise Exception('cant find sub slot in sys state')
            else:
                for idx, s in enumerate(staff_state):
                    if s in slot:
                        slot_index[idx] = 1
        else:
            slot = re.findall(r'[(](.*?)[)]', label)[0].replace(' ', '')
            slot = slot.split(',')
            max_len = 0
            for sl in slot:
                idx = ''.join(staff_state).find(sl)
                if idx != -1:
                    if idx == 0:
                        slot_index[0] = 1
                        slot_index[1] = 1
                    else:
                        slot_index[1] = 1
                        slot_index[2] = 1
        return slot_index

    def get_re(self, state):
        regs = cfg.RegularExpressions
        state_re = []
        for regular in regs:
            regular = re.compile(regular)
            if re.match(regular, state):
                state_re.append(1)
            else:
                state_re.append(0)
        return state_re

    def get_confirm_id(self):
        confirm_id = [0] * len(''.join(self.last_staff_state))
        slot_length = len(''.join(self.last_staff_slot))
        if len(''.join(self.last_staff_slot)) > 0:
            index = ''.join(self.last_staff_state).rfind(''.join(self.last_staff_slot))
            if index != -1:
                confirm_id[index: index + slot_length] = [1] * slot_length
        return confirm_id

    def convert_label_to_action_and_slot(self, label, state):
        if '(' not in label:
            label += '()'
        action = re.findall(r'(.*)[(]', label)[0]
        if action != 'compare':
            slot = re.findall(r'[(](.*?)[)]', label)[0].replace(' ', '')
            slot = slot.replace(',', '').replace('，', '')
            slot = slot.split('->')[-1]
        else:
            slot = re.findall(r'[(](.*?)[)]', label)[0].replace(' ', '')
            slot = slot.split(',')
            max_len = 0
            for sl in slot:
                if ''.join(state).find(sl) != -1:
                    if max_len < len(sl):
                        max_len = len(sl)
                        slot = sl

        return action, slot

    # @staticmethod
    # def _get_user_act(user_label, tag):
    #     act_map = {
    #         "inform-multi_inform": "inform_multi",
    #         "how_signal-signal": "how_signal",
    #         "good_signal-signal": "good_signal",
    #         "restart-restart": "restart",
    #         "affirm-normal": "affirm",
    #         "inform-2X": "inform_2x",
    #         "inform-compare": "update_normal",
    #         "update-negate_n": "update_negate_n",
    #         "update-update_jige": "update_normal",
    #         "update-negate": "update_negate",
    #         "deny-normal": "deny",
    #         "inform-asr_num_to_han": "inform_normal",
    #         "inform-inform_jige": "inform_normal",
    #         "update-update_sure": "update_sure",
    #         "bad_signal-signal": "bad_signal",
    #         "update-sub_update": "update_sub",
    #         "update-update_two_part": "update_two_part",
    #         "ask_repeat-normal": "ask_repeat",
    #         "update-update_special": "update_special",
    #         "wait-normal": "wait",
    #         "ack-normal": "ack",
    #         "update-before_miss": "update_before_miss",
    #         "update-after_miss": "update_after_miss",
    #         "update-compare": "update_normal",
    #         "doubt_identity-robot": "doubt_identity",
    #         "inform-tone_inform": "inform_tone",
    #         "inform-inform_update": "inform_update",
    #         "update-last_miss": "update_last_miss",
    #         "inform-inform": "inform_normal",
    #         "ask_state-normal": "ask_state",
    #         "other-jushi": "other",
    #         "finish-normal": "finish",
    #         "update-update_normal": "update_normal",
    #         "offer-normal": "offer"
    #     }
    #     if '(' in user_label: user_label = user_label[:user_label.rfind('(')]
    #     if tag is None:
    #         return "offer"
    #     idx = user_label + '-' + tag
    #
    #     if idx not in act_map.keys():
    #         raise Exception('Error, unknown act and tag map!')
    #     return act_map[idx]

