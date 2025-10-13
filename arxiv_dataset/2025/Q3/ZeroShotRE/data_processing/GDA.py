import pandas as pd
from tqdm import tqdm
from universal_classes import Entity, Relation, Example, Dataset
from utils import pickle_save


path_dataset = 'data/raw/GDA'  # Path to the dataset folder
output_data_dir = 'data/processed/GDA'


def read_text_to_df(file_path):
    # Initialize variables
    data = []
    pmid = None
    title = None
    abstract = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove leading and trailing whitespaces

            # Check if the line is a document key
            if line.isdigit():
                # If there is an existing document, save it
                if pmid is not None:
                    data.append({'pmid': pmid, 'title': title, 'abstract': abstract.strip()})
                # Start a new document
                pmid = line
                title = None
                abstract = ""
            # Check if the line is a title (non-empty and document_key is set)
            elif pmid is not None and title is None:
                title = line
            # Otherwise, it's part of the abstract
            elif pmid is not None and title is not None:
                abstract += line + " "

        # Save the last document
        if pmid is not None:
            data.append({'pmid': pmid, 'title': title, 'abstract': abstract.strip()})

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    return df


def GDA_dataset(df_abstracts, df_entities, df_relations):

    examples = []

    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):

        example = {}
        example["title"] = abstract["title"]
        example["text"] = abstract["abstract"]

        pmid = abstract["pmid"]

        entities = df_entities[df_entities['pmid'] == pmid]
        relations = df_relations[df_relations['pmid'] == pmid]

        rel = []
        ent = set()

        for _, row in relations.iterrows():
            geneId = str(row["geneId"])
            diseaseId = str(row["diseaseId"])
            assert row["label"] == 1

            ents0 = set(entities.loc[entities['id'] == geneId, 'text'].tolist())
            ents1 = set(entities.loc[entities['id'] == diseaseId, 'text'].tolist())
            gold_ent0 = Entity(ents0, 'gene', geneId)
            gold_ent1 = Entity(ents1, 'disease', diseaseId)
            gold_rel = Relation({gold_ent0, gold_ent1}, 'association')

            ent.add(gold_ent0)
            ent.add(gold_ent1)
            rel.append(gold_rel)

        example["entities"] = ent
        example["relations"] = set(rel)

        example = Example(**example)
        examples.append(example)
        dataset = Dataset(**{"examples": examples})

    return dataset


df_abstracts = read_text_to_df(path_dataset + '/training_data/abstracts.txt')
df_entities = pd.read_csv(path_dataset + '/training_data/anns.txt', delimiter='\t',
                          names=['pmid', 'char_start', 'char_end', 'text', 'type', 'id'], skipinitialspace=True,
                          encoding='utf-8')
df_relations = pd.read_csv(path_dataset + "/training_data/labels.csv", keep_default_na=False, encoding='utf-8')
df_abstracts['pmid'] = df_abstracts['pmid'].astype(str)
df_entities['pmid'] = df_entities['pmid'].astype(str)
df_relations['pmid'] = df_relations['pmid'].astype(str)

train = GDA_dataset(df_abstracts, df_entities, df_relations)
pickle_save(train, output_data_dir + '/' + 'train_data.save')


df_abstracts = read_text_to_df(path_dataset + '/testing_data/abstracts.txt')
df_entities = pd.read_csv(path_dataset + '/testing_data/anns.txt', delimiter='\t',
                          names=['pmid', 'char_start', 'char_end', 'text', 'type', 'id'], skipinitialspace=True,
                          encoding='utf-8')
df_relations = pd.read_csv(path_dataset + "/testing_data/labels.csv", keep_default_na=False, encoding='utf-8')
df_abstracts['pmid'] = df_abstracts['pmid'].astype(str)
df_entities['pmid'] = df_entities['pmid'].astype(str)
df_relations['pmid'] = df_relations['pmid'].astype(str)

test = GDA_dataset(df_abstracts, df_entities, df_relations)
pickle_save(test, output_data_dir + '/' + 'test_data.save')
