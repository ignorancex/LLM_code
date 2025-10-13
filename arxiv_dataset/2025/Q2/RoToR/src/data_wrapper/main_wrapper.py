from src.data_wrapper.mmlu_wrapper import MmluWrapper
from src.data_wrapper.litm_wrapper import LitmWrapper
from src.data_wrapper.kaping_wrapper import KapingWrapper

class MainDataWrapper():
    def __init__(self, args):
        self.args = args
        self.wrapper = self.route_data()
        self.total_length = len(self.wrapper.dataset)

    def route_data(self):
        if self.args.data == 'mmlu':
            return MmluWrapper(self.args)
        elif self.args.data == 'lostinthemiddle':
            return LitmWrapper(self.args)
        elif self.args.data == 'mintaka':
            return KapingWrapper(self.args)
        else:
            raise NotImplementedError

    def get_data_by_index(self, i):
        return self.wrapper.get_data_by_index_raw(i)
        # return: {'prefix': xx, 'rows': [], 'suffix': xx, 'answer': xx}

    def parse_output(self, pred):
        return self.wrapper.parse_output(pred)

    def get_accuracy_instance(self, pred, answers):
        return self.wrapper.get_accuracy_instance(pred, answers)

    def get_accuracy_all(self, output_data):
        return self.wrapper.get_accuracy_all(output_data)
