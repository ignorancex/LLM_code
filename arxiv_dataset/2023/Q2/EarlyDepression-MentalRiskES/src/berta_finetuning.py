from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset#, Features, Value, ClassLabe

ds = load_dataset('nlpUc3mStudents/mental-risk-c')
# to pandas
train_df = ds['train'].to_pandas()
test_df = ds['test'].to_pandas()
label_names = train_df.iloc[:,4:].columns.tolist()
# concat messages by subject id
train_by_subjectid = (
    train_df.groupby('subject_id')
    .agg({'message': lambda x: ' | '.join(x), **{col: 'first' for col in label_names}})
    .reset_index()
    # .assign(
    #     num_messages=lambda x: x.message.str.count('\|') + 1
    # )
)
# back to datasets
train_df = Dataset.from_pandas(train_by_subjectid)

model_name= 'hackathon-somos-nlp-2023/roberta-base-bne-finetuned-suicide-es'
    
tokenizer = AutoTokenizer.from_pretrained(model_name)
# this model is trained with 2 labels, yet we need 4, so we need to change the head
model = None


