
class EVar:
    BertBatchSize = 32
    MaxSequence = 160
    Dropout = 0.1
    DefaultTask = 't'

    LblAll = [-1, 0, 1, 2, 3]
    LblNonHealth = [-1, 0]
    LblHealth = [1, 2, 3]
    LblGeneralHealth = [1]
    LblNonGeneralHealth = [-1, 0, 2, 3]
    LblEventHealth = [2, 3]
    LblNonEventHealth = [-1, 0, 1]

    Lbl3rdClass = [4]
    Lbl4thClass = [5]
    Lbl5thClass = [6]
    Lbl6thClass = [7]
    Lbl7thClass = [8]
    Lbl8thClass = [9]
    Lbl9thClass = [10]

