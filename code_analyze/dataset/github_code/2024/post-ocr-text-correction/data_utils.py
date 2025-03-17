import editdistance

import pandas as pd
import regex as re
import numpy as np

PUNCTUATION_WITHOUT_AT = '\ufeff!"#$%&\'()*+, -./:;<=>?[\]^_`{|}~“'
PUNCTUATION_ALL = '!"#$„%&\'()*+, -.@/:;<=>?[\]^_`{|}~“'

pd.options.display.max_rows = 5

class TextProcessor:
  def __init__(self, max_sequence_length, number_of_unique_tokens, char_to_id):
    self.max_sequence_length = max_sequence_length
    self.number_of_unique_tokens = number_of_unique_tokens
    self.char_to_id = char_to_id

  def to_ids(self, words):
    vector = np.zeros((len(words), self.max_sequence_length, self.number_of_unique_tokens), dtype="float32")

    for i, word in enumerate(words):
      for t, char in enumerate(word):
        vector[i, t, self.char_to_id[char]] = 1.0
      vector[i, t + 1:, self.char_to_id[' ']] = 1.0

    return vector

def clean_text_all(word):
  word = re.sub(r'\n', '', word)

  return word.lower().strip(PUNCTUATION_ALL).replace('@', '')

def clean_text(word):
  word = re.sub(r'\n', '', word)

  return word.lower().strip(PUNCTUATION_WITHOUT_AT)

def clean_up(word):
  word = re.sub(r'\n', '', word)
  word = word.strip(PUNCTUATION_ALL).lower().replace('@', '')

  return word.strip()

def get_space_positions(sentence, gold_standard_sentence):

    gold_standard_sentence = gold_standard_sentence.strip('')

    raw_ocr_space_positions = [match.span()[0] for match in re.finditer(" ", sentence)]
    gold_standard_ocr_space_positions = [match.span()[0] for match in re.finditer(" ", gold_standard_sentence)]

    gold_standard_ocr_space_len = len(gold_standard_ocr_space_positions)

    cursor = 0
    usable = []

    for space_position in raw_ocr_space_positions:
        while cursor < gold_standard_ocr_space_len and gold_standard_ocr_space_positions[cursor] < space_position:
            cursor += 1

        if cursor < gold_standard_ocr_space_len and gold_standard_ocr_space_positions[cursor] == space_position:
            usable.append(space_position)
    
    usable.append(len(sentence))
    
    return usable

def read_data(files, sentence_tokenizer):
    return read_dataset(files, sentence_tokenizer)

def read_dataset(files, sentence_tokenizer, clean = True):
    words = []
    aligned_words = []
    aligned_gs_words = []

    labels = []

    total_files = len(files)
    print("Total files {}".format(total_files))
    
    for file in files:
        print("Processing {}".format(file))
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.readlines()

        aligned_ocr = raw_text[1][14:]
        aligned_gs = raw_text[2][14:]

        file_aligned_words = []
        file_words = []
        file_aligned_gs_words = []
        file_labels = []
        
        sentence_spans = sentence_tokenizer.span_tokenize(aligned_ocr)
        
        for sentence_start, sentence_end in sentence_spans:
            aligned_sentence = aligned_ocr[sentence_start:sentence_end]
            gs_sentence = aligned_gs[sentence_start:sentence_end]
            sentence_aligned_words = []
            sentence_aligned_gs_words = []
            sentence_words = []
            sentence_labels = []

            ocr_space_positions = get_space_positions(aligned_sentence, gs_sentence)
            
            word_start = 0
            for space_position in ocr_space_positions:
                word = aligned_sentence[word_start:space_position]

                if len(word) == 0:
                    word_start = space_position + 1 
                    continue
                
                cleared_word = clean_text_all(word) if clean else word
                gs_word = gs_sentence[word_start:space_position]
                gs_word = clean_text_all(gs_word) if clean else gs_word

                label = 0
                if cleared_word != gs_word:
                    label = 1
                
                word_start = space_position + 1

                if len(cleared_word) == 0 or len(gs_word) == 0:
                  continue
            
                sentence_labels.append(label)
                sentence_words.append(cleared_word)
                sentence_aligned_words.append(word)
                sentence_aligned_gs_words.append(gs_word) 
            
            if len(sentence_words) > 0 and len(sentence_aligned_gs_words) > 0:
                file_words.append(sentence_words)
                file_aligned_words.append(sentence_aligned_words)
                file_aligned_gs_words.append(sentence_aligned_gs_words)
                file_labels.append(sentence_labels)
        
        words.extend(file_words)
        aligned_words.extend(file_aligned_words)
        aligned_gs_words.extend(file_aligned_gs_words)
        labels.extend(file_labels)

    return words, aligned_words, aligned_gs_words, labels

def read_synthetic_data(files):
    words = []
    aligned_words = []
    aligned_gs_words = []
    labels = []

    total_files = len(files)
    print("Total files {}".format(total_files))

    for file in files:
        print("Processing {}".format(file))
        with open(file, "r", encoding="utf-8") as f:
            raw_text = f.readlines()

        file_words = []
        file_aligned_words = []
        file_aligned_gs_words = []
        file_labels = []

        for i in range(0, len(raw_text), 2):
            gs_sentence = clean_text(raw_text[i])
            sentence = clean_text(raw_text[i + 1])
              
            sentence_words = []
            sentence_aligned_words = []
            sentence_aligned_gs_words = []
            sentence_labels = []

            new_ocr_space_ids = get_space_positions(sentence, gs_sentence)
            
            word_start = 0
            for space_id in new_ocr_space_ids:
                word = sentence[word_start:space_id]

                if len(word) == 0:
                    word_start = space_id + 1 
                    continue
                
                trimmed_word = clean_up(word)
                gs_word = gs_sentence[word_start:space_id]
                gs_word = clean_up(gs_word)

                label = 0
                if trimmed_word != gs_word:
                    label = 1
                
                sentence_labels.append(label)
                sentence_aligned_words.append(word)
                sentence_words.append(trimmed_word)
                sentence_aligned_gs_words.append(gs_word)
                
                word_start = space_id + 1

            if len(sentence_aligned_words) > 0 and len(sentence_aligned_gs_words) > 0:
                file_aligned_words.append(sentence_aligned_words)
                file_words.append(sentence_words)
                file_aligned_gs_words.append(sentence_aligned_gs_words)
                file_labels.append(sentence_labels)
          
        aligned_words.extend(file_aligned_words)
        words.extend(file_words)
        aligned_gs_words.extend(file_aligned_gs_words)
        labels.extend(file_labels)
    
    return aligned_words, words, aligned_gs_words, labels

def levenhstein_distance(sentences):
    raw_ocr_sentence = ''.join(sentences['ocr_sentence'])
    gold_standard_sentence = ''.join(sentences['gs_sentence'])

    return editdistance.distance(raw_ocr_sentence, gold_standard_sentence) / max(len(raw_ocr_sentence), len(gold_standard_sentence))

def filter_data(words_raw, aligned_words, gs_words, labels, max_norm_lev_distance=0.5):
    sent_stat = pd.DataFrame({
        "ocr_sentence": aligned_words, 
        "gs_sentence": gs_words
    })

    sent_stat["sent_levenshtein_distance"] = sent_stat.apply(levenhstein_distance, axis=1)
    print("good sentences: %s\ntotal sentences: %s" % ((sent_stat["sent_levenshtein_distance"] <= max_norm_lev_distance).sum(), sent_stat.shape[0]))
    
    good_sentences_stat = sent_stat[sent_stat["sent_levenshtein_distance"] <= max_norm_lev_distance]

    words_filtered = np.array(words_raw, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    aligned_words_filtered = np.array(aligned_words, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    gs_words_filtered = np.array(gs_words, dtype=object)[good_sentences_stat.index.tolist()].tolist()
    labels_filtered = np.array(labels, dtype=object)[good_sentences_stat.index.tolist()].tolist()

    return words_filtered, aligned_words_filtered, gs_words_filtered, labels_filtered

def truncate_and_pad(arr, max_sequence_length, tokenizer):   
    return arr[:max_sequence_length] + [tokenizer.pad_token_id] * (max_sequence_length - len(arr))

def tokenize(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        number_of_subtokens = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * number_of_subtokens)

    tokenized_sentence = [tokenizer.cls_token] + tokenized_sentence + [tokenizer.sep_token]
    labels = [0] + labels + [0]
    
    return tokenized_sentence, labels

def load_clada(file):
    clada_set = set()
    with open(file, "r", encoding="utf-16") as f:
        raw_text = f.readlines()

    for word in raw_text:
        temp = clean_text(word)

        if temp not in clada_set:
            clada_set.add(temp)

    return clada_set