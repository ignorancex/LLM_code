from error_detection_models import CladaErrorDetectionModel, FastTextErrorDetectionModel
from data_utils import PUNCTUATION_ALL, clean_text
from config import ConfigManager
from dataset_parser import DatasetParser
from error_corrector_knn import ErrorCorrectorKnn

from Levenshtein import distance as levenshtein_distance

def evaluate_knn(config, parser):
    clada = CladaErrorDetectionModel(config)
    knn = ErrorCorrectorKnn(config)

    test_words, test_aligned_words, test_gs_words, test_labels = parser.read_test_dataset()

    distances0 = []
    distances1 = []

    len_cor = []
    cnt = 0
    n = len(test_aligned_words)

    for sentence, ocr_sentence, gs_sentence, labels in zip(test_words, test_aligned_words, test_gs_words, test_labels):
        gs_sentence_string = ' '.join(gs_sentence)
        ocr_sentence_string = ' '.join(sentence)
        corrected = []
        raw = []
        cnt = cnt + 1

        print("{}/{}".format(cnt, n))

        for word, label, gs in zip(sentence, labels, gs_sentence):
            corrected_word = word
            x = None
            if label == 1:
                x = knn.call(word)
            # print(x)
            if x is not None:
                corrected_word = str(x)
                len_cor.append(len(gs))
            else: 
                corrected_word = word

            raw.append(word)
            corrected.append(corrected_word)

        ocr_sentence_string = ' '.join(raw)
        corrected_sentence_string = ' '.join(corrected)

        levdist0 = levenshtein_distance(ocr_sentence_string, gs_sentence_string) # distance between aligned and GS
        levdist1 = levenshtein_distance(corrected_sentence_string, gs_sentence_string) # distance between model and GS

        distances0.append(levdist0)
        distances1.append(levdist1)

    improvement = (sum(distances0) - sum(distances1)) / (sum(len_cor) + 1)
    print("The improvement is {:.3f}%".format(improvement * 100))

def extend_dictionary(gs_words, clada):
    all_words = []
    for word in gs_words:
      word = [clean_text(w) for w in word]
      word = list(filter(lambda w: clada.forward(w) and len(w) > 2, word))

      all_words.extend(word)
          
    return all_words

def main():
    config = ConfigManager()
    datasetparser = DatasetParser(config)
    evaluate_knn(config, datasetparser)
    
if __name__ == '__main__':
    main()