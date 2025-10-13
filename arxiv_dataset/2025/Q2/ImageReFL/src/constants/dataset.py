import enum


class DatasetColumns(enum.Enum):
    tokenized_text = "tokenized_text"
    text_attention_mask = "text_attention_mask"
    original_image = "original_image"
    original_sizes = "original_sizes"
