from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC, MultilabelRecall
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision, BinaryF1Score
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score


def classification_binary_metrics(predictions, labels):
    accuracy = BinaryAccuracy()
    f1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    auc = BinaryAUROC()

    acc = accuracy(predictions, labels)
    f1_score = f1(predictions, labels)
    prec = precision(predictions, labels)
    rec = recall(predictions, labels)
    auc_score = auc(predictions, labels)

    return acc, f1_score, prec, rec, auc_score


def classification_multiclass_metrics(predictions, labels, num_classes):
    accuracy = MulticlassAccuracy(num_classes=num_classes)(predictions, labels)
    auroc = MulticlassAUROC(num_classes=num_classes)(predictions, labels)
    f1 = MulticlassF1Score(num_classes=num_classes,
                           average='micro')(predictions, labels)

    return accuracy, auroc, f1


# def classification_multilabel_metrics(predictions, labels):
#     acc = MultilabelAccuracy(num_labels=12, average='weighted')
#     f1 = MultilabelF1Score(num_labels=12, average='micro')
#     precision = MultilabelPrecision(num_labels=12, average='micro')
#     recall = MultilabelRecall(num_labels=12, average='micro')
#     auc = MultilabelAUROC(num_labels=12, average='micro')

#     label_accuracy = acc(predictions, labels)
#     f1_micro = f1(predictions, labels)
#     prec = precision(predictions, labels)
#     rec = recall(predictions, labels)
#     auc_score = auc(predictions, labels)

#     return label_accuracy, f1_micro, prec, rec, auc_score
