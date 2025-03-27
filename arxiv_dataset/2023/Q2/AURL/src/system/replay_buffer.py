import random
from collections import namedtuple, deque
from src.utilities.config_new import global_config as cfg

Dst_state = namedtuple('Dst_state',  ('input_idx', 'last_state_idx', 'last_state_chunk', 'target_user_act', 'target_mem'))


class Replay_buffer():
    def __init__(self, mode, buffer_size=3000):
        self.mode = mode
        self.clear_buffer()

    def clear_buffer(self):
        if self.mode == 'ross_dst':
            self.buffer = {'easy': [], 'middle': [], 'hard': []}
        elif self.mode == 'sl_dst':
            self.buffer = []
        else:
            self.buffer = deque(maxlen=6)

    def get_len(self):
        if self.mode == 'ross_dst':
            e_count = len(self.buffer['easy'])
            m_count = len(self.buffer['middle'])
            h_count = len(self.buffer['hard'])
            total_count = e_count + m_count + h_count
            if total_count == 0:
                return 0, 0, 0, 0
            else:
                return e_count / total_count, m_count/total_count, h_count/total_count, total_count
        else:
            return 0, 0, 0, 0

    def _check_level(self, eposide):
        """

        Args:
            eposide:

        Returns:
            'easy', 'middle', 'hard'
        """
        user_act = cfg.user_act_id2name[eposide.target_user_act]
        if user_act in ['update_sub', 'update_special', 'update_one']:
            return 'hard'
        elif user_act in ['inform_2x', 'update_sure', 'inform_multi', 'update_normal']:
            return 'middle'
        else:
            return 'easy'

    def add_to_buffer(self, eposide):
        if self.mode == 'ross_dst':
            level = self._check_level(eposide)
            self.buffer[level].append(eposide)
        else:
            self.buffer.append(eposide)

    def get_dst_batchs(self, batch_size, level='all'):
        if self.mode == 'ross_dst':
            if level != 'all':
                all_batches = self._construct_mini_batch(self.buffer[level], batch_size)
                for i, batch in enumerate(all_batches):
                    yield batch
            else:
                all_batches = self._construct_mini_batch(self.buffer['easy'], batch_size)
                all_batches.extend(self._construct_mini_batch(self.buffer['middle'], batch_size))
                all_batches.extend(self._construct_mini_batch(self.buffer['hard'], batch_size))
                for i, batch in enumerate(all_batches):
                    yield batch
        else:
            all_batches = self._construct_mini_batch(self.buffer, batch_size)
            for i, batch in enumerate(all_batches):
                yield batch

    def _construct_mini_batch(self, data, batch_size):
        all_batches = []
        idx, n = 0, len(data)
        while idx < n:
            all_batches.append(data[idx:idx+batch_size])
            idx += batch_size
        random.shuffle(all_batches)
        return all_batches

