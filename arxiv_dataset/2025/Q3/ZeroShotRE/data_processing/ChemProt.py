import pandas as pd
from tqdm import tqdm
from universal_classes import Entity, Relation, Example, Dataset
from utils import pickle_save

path_dataset = 'data/raw/ChemProt'  # Path to the dataset folder
output_data_dir = 'data/processed/ChemProt'


def format_relations(relations):
    # Convert to dictionaries
    res = {}
    for _, row in relations.iterrows():
        ent1 = row["arg1"].replace("Arg1:", "")
        ent2 = row["arg2"].replace("Arg2:", "")
        key = (ent1, ent2)
        res[key] = row["label"]
    return res


def ChemProt_dataset(df_abstracts, df_entities, df_relations):

    examples = []

    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):

        example = {}
        example["title"] = abstract["title"]
        example["text"] = abstract["abstract"]
        doc_key = abstract["doc_key"]
        entities = df_entities[df_entities['doc_key'] == doc_key]
        relations = format_relations(df_relations[df_relations['doc_key'] == doc_key])

        rel = []
        ent = set()

        for ents, label in relations.items():
            ents0 = entities.loc[entities['entity_id'] == ents[0], 'text'].to_string(index=False)
            ents0 = {ents0.lower()}
            ents1 = entities.loc[entities['entity_id'] == ents[1], 'text'].to_string(index=False)
            ents1 = {ents1.lower()}
            label = label.lower()

            gold_ent0 = Entity(ents0, 'chemical')
            gold_ent1 = Entity(ents1, 'gene')
            gold_rel = Relation({gold_ent0, gold_ent1}, label)

            ent.add(gold_ent0)
            ent.add(gold_ent1)
            rel.append(gold_rel)

        example["entities"] = ent
        example["relations"] = set(rel)

        example = Example(**example)
        examples.append(example)

        dataset = Dataset(**{"examples": examples})

    return dataset


df_abstracts = pd.read_table(path_dataset + "/chemprot_training/chemprot_training_abstracts.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table(path_dataset + "/chemprot_training/chemprot_training_entities.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table(path_dataset + "/chemprot_training/chemprot_training_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

train = ChemProt_dataset(df_abstracts, df_entities, df_relations)
pickle_save(train, output_data_dir + '/' + 'train_data.save')

df_abstracts = pd.read_table(path_dataset + "/chemprot_development/chemprot_development_abstracts.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table(path_dataset + "/chemprot_development/chemprot_development_entities.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table(path_dataset + "/chemprot_development/chemprot_development_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

valid = ChemProt_dataset(df_abstracts, df_entities, df_relations)
pickle_save(valid, output_data_dir + '/' + 'valid_data.save')

df_abstracts = pd.read_table(path_dataset + "/chemprot_test_gs/chemprot_test_abstracts_gs.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table(path_dataset + "/chemprot_test_gs/chemprot_test_entities_gs.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table(path_dataset + "/chemprot_test_gs/chemprot_test_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

test = ChemProt_dataset(df_abstracts, df_entities, df_relations)
pickle_save(test, output_data_dir + '/' + 'test_data.save')
