from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs,
    MultiLabelClassificationModel,
)
import pandas as pd
from sklearn.metrics import matthews_corrcoef,accuracy_score
import numpy as np
import time

t1=time.time()
df = pd.read_csv(
    "/wrk/knc6/AtomNLP/chemnlp/chemnlp/sample_data/cond_mat_small.csv"
)
df=pd.read_csv('/wrk/knc6/AtomNLP/Summarize/cond_mat.csv') #[0:10000]
ratio = 0.8
n_train = int(len(df) * ratio)
n_test = len(df) - n_train
key = "categories"
df["category_id"] = df[key].factorize()[0]
category_id_df = df[[key, "category_id"]].sort_values("category_id")
category_to_id = dict(category_id_df.values)
print("category_to_id", category_to_id)
df["text"] = df["title"]
df["labels"] = df["category_id"]

train_df = df[0:n_train]
eval_df = df[n_train:-1]
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=9,
    args={
        "stride": 0.8,
        "num_train_epochs": 2,
        "no_save": True,
        "train_batch_size": 8,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 8,
    },
)
# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
pred_labels=np.argmax(model_outputs,axis=1)
test_labels = eval_df['labels']
print("result", result)
print ('acc',accuracy_score(test_labels,pred_labels))
# print ('model_outputs',model_outputs)
# print('wrong_predictions',wrong_predictions)
# Make predictions with the model
#predictions, raw_outputs = model.predict(["Sam was a Wizard"])
t2=time.time()
print ('Time',t2-t1)
