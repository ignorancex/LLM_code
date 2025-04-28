import sys
sys.path.append("/home/user/asr-fuzzing-testing/CrossASRplus")

from crossasr.mutator import Mutator
from crossasr.text import Text
import nlpaug.augmenter.word as naw
from crossasr.utils import preprocess_text

class Augmenter(Mutator):

    def __init__(self, name="augmenter"):
        super().__init__(name)

    def generate_mutated_sentences(self):
        mutated_sentences = []
        failed_transcription_results = self.get_not_mutated_failed_transcription_results()

        number_of_sentences_to_generate = 1

        for i in range(len(failed_transcription_results)):
            text_obj = failed_transcription_results[i].get_text()
            text = text_obj.getText()
            filename = text_obj.getFilename()
            transcriptions = failed_transcription_results[i].get_transcriptions()
            cases = failed_transcription_results[i].get_cases()

            error_words = self.get_all_error_words(text, transcriptions, cases)

            new_sentences = self.get_sentences_with_word_inserted(text, number_of_sentences_to_generate)
            for i in range(len(new_sentences)):
                processed_new_sentence = preprocess_text(new_sentences[i])
                if processed_new_sentence != '':
                    mutated_sentences.append(Text(id=f"{filename}_{i+1}", text=processed_new_sentence, original_sentence=text, error_words = error_words))

        return mutated_sentences

    def get_sentences_with_word_inserted(self, text, n):
        aug = naw.ContextualWordEmbsAug(
        model_path='bert-base-uncased', action="insert")
        augmented_text = aug.augment(text, n)
        print("Original:", text)
        print("Augmented Text:", augmented_text)

        return augmented_text


if __name__ == "__main__":
    text = 'The quick brown fox jumps over the lazy dog .'
    text2 = 'what are the greens doing calling for taxes on foods with a particular type of nutritional content'

    # Insert word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
    aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text2, n=1)

    for text in augmented_text:
        print(preprocess_text(text))

    # print("Original:")
    # print(text)
    # print("Augmented Text:")
    # print(augmented_text)

    # Substitute word by contextual word embeddings (BERT, DistilBERT, RoBERTA or XLNet)
    # aug = naw.ContextualWordEmbsAug(
    # model_path='bert-base-uncased', action="substitute")
    # augmented_text = aug.augment(text)
    # print("Original:")
    # print(text)
    # print("Augmented Text:")
    # print(augmented_text)