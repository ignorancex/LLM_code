from error_detection_models import CladaErrorDetectionModel, FastTextErrorDetectionModel
from data_utils import PUNCTUATION_ALL, clean_text
from config import ConfigManager
from dataset_parser import DatasetParser

def evaluate_clada(config, parser):
    clada = CladaErrorDetectionModel(config)
    train_words, train_aligned_words, train_gs_words, train_labels = parser.read_train_dataset()
    
    new_words_from_icdar = extend_dictionary(train_gs_words, clada)
    clada.clada_dictionary.update(new_words_from_icdar)

    test_words, test_aligned_words, test_gs_words, test_labels = parser.read_test_dataset()

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    #aligned_words_manual, words_manual, aligned_gs_words_manual, labels_manual 
    for sentence, sgs, labels in zip(test_words, test_gs_words, test_labels):
        for word, gs1, label in zip(sentence, sgs, labels):
            word = clean_text(word)

            if not clada.forward(word):
                if label == 1:
                    fn = fn + 1
                else:
                    tn = tn + 1
            else:
                if label == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1

    print(tp)
    print(fp)
    print(tn)
    print(fn)
    
    print("ICDAR Evaluation: ")
    print("Accuracy: %f" % ((tp + tn) / (tp + fp + tn + fn)))
    print("Precision: %f" % ((tp) / (tp + fp)))
    print("Recall: %f" % ((tp) / (tp + fn)))
    print("F1 score: %f" % ((tp) / (tp + 0.5 * (fp + fn))))

def evaluate_fasttext(config, parser):
    train_words, train_aligned_words, train_gs_words, train_labels = parser.read_train_dataset()
    test_words, test_aligned_words, test_gs_words, test_labels = parser.read_test_dataset()

    dataset = config.get_main_path() + '/fasttext/train.txt'

    model = FastTextErrorDetectionModel(config)

    create_dataset_for_fastText(train_words, train_labels, dataset)
    model.train(dataset, epoch=1500)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for word, label in zip(test_words, test_labels):
        for w, l in zip(word, label):
            w = clean_text(w)
            lbl = model.forward(w)
            if lbl == l:
                if lbl == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if lbl == 1:
                    fp = fp + 1
                else:
                    fn = fn + 1
    print(tp)
    print(tn)
    print(fp)
    print(fn)

    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))
    print("Recall:", tp / (tp + fn))
    print("Precision:", tp / (tp + fp))
    print("F1:", (tp / (tp + 0.5 * (fn + fp))))



def extend_dictionary(gs_words, clada):
    all_words = []
    for word in gs_words:
      word = [clean_text(w) for w in word]
      word = list(filter(lambda w: clada.forward(w) and len(w) > 2, word))

      all_words.extend(word)
          
    return all_words

def create_dataset_for_fastText(words, labels, file):
  fast_text_input = []

  for i, (row, labels) in enumerate(zip(words, labels)):
    for j, (word, label) in enumerate(zip(row, labels)):
      fast_text_i = "__label__{} {}".format(label, word)
      fast_text_input.append(fast_text_i)

  f = open(file, "w", encoding="utf-8")

  for line in fast_text_input:
    f.write(line + '\n')

  f.close()

def main():
    config = ConfigManager()
    datasetparser = DatasetParser(config)
    # evaluate_clada(config, datasetparser)
    evaluate_fasttext(config, datasetparser)
    
if __name__ == '__main__':
    main()