from RST.src.ELib import ELib
from RST.src.ELbl import ELbl
from RST.src.EVar import EVar


class ELblConf:

    def __init__(self, negative_new_label=None, positive_new_label=None, binary_labels=None, multiclass_labels=None):
        """the args should be either setup for binary or multi-class tasks"""
        if multiclass_labels is None:
            if negative_new_label is None or positive_new_label is None or binary_labels is None:
                print('Incorrect ELblConf Args')
                exit(1)
            self.negative_new_label = None
            self.positive_new_label = None
            binary_labels.sort(key=lambda item: item.new_label)
            self.labels = binary_labels
            for cur_lbl in binary_labels:
                if negative_new_label == cur_lbl.new_label:
                    self.negative_new_label = cur_lbl
                    break
            for cur_lbl in binary_labels:
                if positive_new_label == cur_lbl.new_label:
                    self.positive_new_label = cur_lbl
                    break
        else:
            if negative_new_label is not None or positive_new_label is not None or binary_labels is not None:
                print('Incorrect ELblConf Args')
                exit(1)
            multiclass_labels.sort(key=lambda item: item.new_label)
            self.labels = multiclass_labels

    def get_correct_new_label(self, lbl):
        for cur_lbl in self.labels:
            if cur_lbl.is_source(lbl):
                return cur_lbl.new_label
        ELib.out("Unknown label to map!")
        return -10

    def get_sample_label_from_new_label(self, new_label):
        for cur_lbl in self.labels:
            if new_label == cur_lbl.new_label:
                return cur_lbl.source_lbls[-1]
        ELib.out("Unknown NewLabel to map!")
        return -10

    @staticmethod
    def get_regular_lblconfig():
        return ELblConf(multiclass_labels=[ELbl(0, EVar.LblNonEventHealth),
                                           ELbl(1, EVar.LblEventHealth),
                                           ELbl(2, EVar.Lbl3rdClass),
                                           ELbl(3, EVar.Lbl4thClass),
                                           ELbl(4, EVar.Lbl5thClass),
                                           ELbl(5, EVar.Lbl6thClass),
                                           ELbl(6, EVar.Lbl7thClass)])


