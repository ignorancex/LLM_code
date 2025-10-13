import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

_MANUAL_EVAL_ANNOTATED_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_annotated.xlsx"
_MANUAL_EVAL_INPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_with_label.csv"
_MAJORITY_VOTING_OUTPUT_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_majority.csv"

_NO_ERROR = "AT LEAST SOME PART of the evidence is non-empty, understandable, and related to the claim."
scaler = MinMaxScaler()


def main():
    # load both files
    annotated_data = pd.read_excel(_MANUAL_EVAL_ANNOTATED_PATH)
    input_data = pd.read_csv(_MANUAL_EVAL_INPUT_PATH)

    # Remove entries which say there is an error in the claim/evidence pair
    annotated_data = annotated_data[annotated_data['predicted_evidence_errors'] == _NO_ERROR]

    count_df = annotated_data.groupby('id').size()
    count_df.to_excel("/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/counts.xlsx")

    # majority voting => average of the scores provided by annotators
    annotated_data_summary = annotated_data.groupby('id')['semantic_coverage'].mean().reset_index()
    annotated_data_summary['coherence'] = annotated_data.groupby('id')['coherence'].mean().values
    annotated_data_summary['redundancy'] = annotated_data.groupby('id')['redundancy'].mean().values
    annotated_data_summary['consistency'] = annotated_data.groupby('id')['consistency'].mean().values
    annotated_data_summary['count'] = annotated_data.groupby('id').size().values

    # remove all entries with only one annotation available (no majority voting possible)
    annotated_data_summary = annotated_data_summary[annotated_data_summary['count'] > 1]

    # scaling values on scale [0, 1] after majority voting
    annotated_data_summary['semantic_coverage_scaled'] = scaler.fit_transform(annotated_data_summary[['semantic_coverage']])
    annotated_data_summary['coherence_scaled'] = scaler.fit_transform(annotated_data_summary[['coherence']])
    annotated_data_summary['redundancy_scaled'] = scaler.fit_transform(annotated_data_summary[['redundancy']])
    annotated_data_summary['consistency_scaled'] = scaler.fit_transform(annotated_data_summary[['consistency']])

    # merge with input_data df
    merged_df = pd.merge(input_data, annotated_data_summary, on='id', how='inner')

    # add label after majority voting to annotated_data_summary['label_majority']
    merged_df['label_majority'] = ""
    merged_df['verdict_agreement'] = ""
    merged_df['nei_disagreement'] = ""

    for i, row in merged_df.iterrows():
        labels = list(annotated_data[annotated_data['id'] == row['id']]['label'])
        # get most common label
        c = Counter(labels).most_common()
        merged_df.at[i, 'label_majority'] = c[0][0]

        # add field by comparing to gold, 1 if gold and majority labels same otherwise 0
        merged_df.at[i, 'verdict_agreement'] = 1 if merged_df.at[i, 'label_majority'] == merged_df.at[i, 'gold label'] else 0
        merged_df.at[i, 'nei_disagreement'] = 0 if merged_df.at[i, 'label_majority'] == "not enough information" and merged_df.at[i, 'gold label'] != "not enough information" else 1

    # save output
    merged_df.to_csv(_MAJORITY_VOTING_OUTPUT_PATH)


if __name__ == '__main__':
    main()
