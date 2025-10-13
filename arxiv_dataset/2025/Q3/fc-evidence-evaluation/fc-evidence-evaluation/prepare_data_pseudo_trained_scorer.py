import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import utils
import properties

SPLIT = "dev"
dev_merge = True
path_averitec = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_{}.json".format(SPLIT)
path_hover = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/hover/hover_{}_release_v1.1.json".format(SPLIT)
path_vitaminc = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/vitaminc_factchecking/{}.jsonl".format(SPLIT)
data_dir = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data"

finetuning_data_path = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/train.jsonl"
finetuning_data_path_dev = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/dev.jsonl"

output_averitec = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/averitec_{}.jsonl".format(SPLIT)
output_hover = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/hover_{}.jsonl".format(SPLIT)
output_vitaminc = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/vitaminc_{}.jsonl".format(SPLIT)

SAMPLE_DICT = {
    'claim': None,
    'evidence': None,
    'label': None,
}

# wiki db for hover
wiki_db = utils.connect_to_db(os.path.join(data_dir, "hover", 'wiki_wo_links.db'))

# prepare training data
claims_averitec, evidence_averitec, labels_averitec = utils.read_averitec_dataset(path_averitec)
claims_vitaminc, evidence_vitaminc, labels_vitaminc = utils.read_vitaminc_dataset(path_vitaminc)
if SPLIT != "dev":
    # hover does not have dev set
    claims_hover, evidence_hover, labels_hover = utils.read_hover_dataset(path_hover, wiki_db)


def _prepare_df_sample(claims, evidences, labels):
    ds_entries = []
    for claim, evid, label in zip(claims, evidences, labels):
        ds_entry = SAMPLE_DICT.copy()
        ds_entry['claim'] = claim
        ds_entry['evidence'] = evid
        ds_entry['label'] = properties.LABEL_DICT_REVERSE[label]
        ds_entries.append(ds_entry)
    return ds_entries


def _prepare_and_save(claims, evidences, labels, path):
    data_formatted = _prepare_df_sample(claims, evidences, labels)
    utils.save_jsonl_file(data_formatted, path)


if SPLIT == "train":
    # merge all datasets to create a joined dataset for scorer finetuning
    claims_finetuning = claims_averitec + claims_hover + claims_vitaminc
    evidences_finetuning = evidence_averitec + evidence_hover + evidence_vitaminc
    labels_finetuning = labels_averitec + labels_hover + labels_vitaminc
    _prepare_and_save(claims_finetuning, evidences_finetuning, labels_finetuning, finetuning_data_path)
elif SPLIT == "dev" and dev_merge:
    # merge all datasets to create a joined dataset for scorer eval
    # if dev_merge is false => separate dev files created
    claims_finetuning = claims_averitec + claims_vitaminc
    evidences_finetuning = evidence_averitec + evidence_vitaminc
    labels_finetuning = labels_averitec + labels_vitaminc
    _prepare_and_save(claims_finetuning, evidences_finetuning, labels_finetuning, finetuning_data_path_dev)
else:
    _prepare_and_save(claims_averitec, evidence_averitec, labels_averitec, output_averitec)
    _prepare_and_save(claims_vitaminc, evidence_vitaminc, labels_vitaminc, output_vitaminc)
    if SPLIT != "dev":
        # hover does not have dev set
        _prepare_and_save(claims_hover, evidence_hover, labels_hover, output_hover)
