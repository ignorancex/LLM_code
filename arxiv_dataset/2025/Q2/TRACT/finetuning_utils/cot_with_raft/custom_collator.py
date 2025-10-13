from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask

from colorama import Fore, Style


class SFTDataCollatorWith4DAttentionMaskForCoTRaft(SFTDataCollatorWith4DAttentionMask):
    def __call__(self, features):
        features = super().__call__(features)
        features['labels'] = {
            'lm_labels': features['labels'],
            'score_labels': features['score_labels'],
        }
        features.pop('score_labels')
        return features