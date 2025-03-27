from data import VarDialDataset
from dialect_id import DialectID
from mappings import (
    label_to_int_mappings_3class,
    label_to_int_mappings_2class,
    int_to_label_mappings__2class,
    int_to_label_mappings__3class,
    label_mappings_en_3class,
    label_mappings_en_2class,
    label_mappings_es_3class,
    label_mappings_es_2class,
    label_mappings_pt_3class,
    label_mappings_pt_2class,
)


train_df = VarDialDataset(
    "SharedTask/EN_train.tsv",
    label_mappings=label_mappings_en_3class,
    num_classes=3,
    mode="train",
)
val_df = VarDialDataset(
    "SharedTask/EN_dev.tsv",
    label_mappings=label_mappings_en_3class,
    num_classes=3,
    mode="validation",
)
train = train_df.create_loader(8)
val = val_df.create_loader(8)

model = DialectID(
    "roberta-base",
    label_mappings=label_mappings_en_3class,
    language="en",
    num_classes=3,
)
model.fit(train, val)
