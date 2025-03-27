import torch
from torch.utils.data.dataloader import Dataset
from src.utilities.config_new import global_config as cfg


class ManagerDataset2(Dataset):
    def __init__(self, voc, dialogues):
        super(ManagerDataset2, self).__init__()
        self.voc = voc
        self.dialogs = dialogues

    def __getitem__(self, item):
        dialog = self.dialogs[item]
        return dialog

    def __len__(self):
        return len(self.dialogs)

    def get_utt_seq(self, utterance):
        utt_seq = []
        for w in utterance:
            utt_seq.append(self.voc[w] if w in self.voc else self.voc['<UNK>'])
        utt_seq.append(self.voc['<EOS>'])
        return utt_seq

    def get_target_state_seq(self, state):
        return self.get_utt_seq(','.join(state))

    def get_state_seq(self, state):
        state_list_seqs = []
        list_chunk_id = []
        for i, st in enumerate(state, start=1):
            utt_seq = [self.voc[w] for w in st]
            chunk_id = [i] * (len(st))
            if i < len(state):
                utt_seq.append(self.voc[','])
                chunk_id.append(i)
            state_list_seqs.extend(utt_seq)
            list_chunk_id.extend(chunk_id)
        state_list_seqs.append(self.voc['<EOS>'])
        list_chunk_id.append(0)
        return state_list_seqs, list_chunk_id

    def pad_seq(self, seqs, max_len=None):
        seq_len = [len(seq) for seq in seqs]
        if not max_len:
            max_len = max(seq_len)
        for seq in seqs:
            seq.extend([self.voc['<PAD>']] * (max_len - len(seq)))
        return seq_len

    def pad_vector(self, vector, id=0):
        max_len = cfg.max_chunk
        # for vector in vectors:
        #     vector.extend([id] * (max_len - len(vectors)))
        # return vectors
        return vector + [id] * (max_len - len(vector))


    def collate_fn(self, data_batch):
        """
        :return: [turn, batch]
        """
        batch_size = len(data_batch)

        # check max turns within considered dialogues
        dial_len = [dial['dial_len'] for dial in data_batch]
        max_dial_len = max(dial_len)

        batch_list = []
        for turn_idx in range(max_dial_len):
            # prepare a batch
            user_utter, user_utter_len, user_act, dlg_his = [], [], [], []
            lst_sys_state, lst_sys_state_chunk = [], []
            sys_state, belief_state, sys_act, sys_slot_idx = [], [], [], []
            reg_exp = []
            valid_turn = []
            for dial_idx in range(batch_size):
                dial = data_batch[dial_idx]
                valid = True if turn_idx < dial['dial_len'] else False

                # input utterance
                user_utter.append(self.get_utt_seq(dial['user_utterance'][turn_idx]) if valid else [self.voc['<EOS>']])
                dlg_his.append(self.get_utt_seq(dial['dialog_history'][turn_idx]) if valid else [self.voc['<EOS>']])

                # act
                user_act.append([cfg.user_act_name2id[dial['user_action'][turn_idx]]] if valid else [0])
                sys_act.append([cfg.sys_act_name2id[dial['staff_action'][turn_idx]]] if valid else [0])
                sys_slot_idx.append(self.pad_vector(dial['staff_slot_idx'][turn_idx]) if valid else [0]*cfg.max_chunk)

                # state
                lst_sys_state_seq, lst_sys_state_chunk_id = [self.voc['<EOS>']], [0]
                if valid:
                    lst_sys_state_seq, lst_sys_state_chunk_id = self.get_state_seq(dial['last_staff_state'][turn_idx])
                lst_sys_state.append(lst_sys_state_seq)
                lst_sys_state_chunk.append(lst_sys_state_chunk_id)

                sys_state.append(self.get_target_state_seq(dial['staff_state'][turn_idx]) if valid else [self.voc['<EOS>']])

                belief_state.append(("号-号-" + ','.join(dial['staff_state'][turn_idx])) if valid else "")
                reg_exp.append(dial['reg_exp'][turn_idx] if valid else [0, 0])
                valid_turn.append(valid)

            user_utter_len = self.pad_seq(user_utter)
            dlg_his_len = self.pad_seq(dlg_his)
            user_act_len = self.pad_seq(user_act)
            sys_act_len = self.pad_seq(sys_act)

            lst_sys_state_len = self.pad_seq(lst_sys_state)
            lst_sys_state_chunk_len = self.pad_seq(lst_sys_state_chunk)
            assert lst_sys_state_len == lst_sys_state_chunk_len
            sys_state_len = self.pad_seq(sys_state, max_len=cfg.MAX_STATE_LEN)

            # add a tensor of batch into output list
            batch = {'word_idx': {}, 'sent_len': {}, 'act_idx': {}, 'dst_idx': {}}
            batch['word_idx']['usr_utter'] = torch.LongTensor(user_utter)
            batch['word_idx']['dialog_history'] = torch.LongTensor(dlg_his)
            batch['act_idx']['user_act'] = torch.LongTensor(user_act)
            batch['act_idx']['sys_act'] = torch.LongTensor(sys_act)
            batch['act_idx']['sys_slot_idx'] = torch.LongTensor(sys_slot_idx)
            batch['sent_len']['user_utter_len'] = user_utter_len
            batch['sent_len']['dialog_history_len'] = dlg_his_len
            batch['sent_len']['user_act_len'] = user_act_len
            batch['sent_len']['sys_act_len'] = sys_act_len
            batch['sent_len']['lst_sys_state_len'] = lst_sys_state_len
            batch['sent_len']['sys_state_len'] = sys_state_len
            batch['dst_idx']['lst_sys_state'] = torch.LongTensor(lst_sys_state)
            batch['dst_idx']['lst_sys_state_chunk'] = torch.LongTensor(lst_sys_state_chunk)
            batch['dst_idx']['sys_state'] = torch.LongTensor(sys_state)
            batch['dst_idx']['reg_exp'] = torch.LongTensor(reg_exp)
            batch['belief_state'] = belief_state
            batch['dial_len'] = dial_len
            batch['valid'] = torch.Tensor(valid_turn)

            batch_list.append(batch)

        # add prev act idx
        for turn_idx , batch in enumerate(batch_list):
            batch['prev_act_idx'] = {}
            if turn_idx == 0:  # empty prev act for first turn
                batch['prev_act_idx']['usr'] = cfg.user_act_name2id['offer'] * torch.ones(batch_size, 1).long()
                batch['prev_act_idx']['sys'] = cfg.sys_act_name2id['other'] * torch.ones(batch_size, 1).long()
                batch['sent_len']['prev_act_usr'] = torch.ones(batch_size).long()
                batch['sent_len']['prev_act_sys'] = torch.ones(batch_size).long()
            else:
                batch['prev_act_idx']['usr'] = batch_list[turn_idx - 1]['act_idx']['user_act']
                batch['prev_act_idx']['sys'] = batch_list[turn_idx - 1]['act_idx']['sys_act']
                batch['sent_len']['prev_act_usr'] = batch_list[turn_idx - 1]['sent_len']['user_act_len']
                batch['sent_len']['prev_act_sys'] = batch_list[turn_idx - 1]['sent_len']['sys_act_len']
        return batch_list