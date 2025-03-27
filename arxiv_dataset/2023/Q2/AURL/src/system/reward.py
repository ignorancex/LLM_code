

class Reward(object):
    @staticmethod
    def manager_r(terminal, success, legal):
        if terminal:
            if success:
                return 2.0
            else:
                return -1.0
        else:
            turn_r = -0.05
            n_legal_r = -0.05
            return turn_r - legal * n_legal_r

    @staticmethod
    def simulator_r(terminal, success, legal, total_turn):
        r_succ = 2.0
        r_nlegel = -0.02
        r_adj = 0.01

        if terminal:
            if success:
                return 2.0
            else:
                return -1.0
        else:
            return -0.02 -legal * r_nlegel


reward = Reward()
