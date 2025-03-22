import pandas as pd
def save_predictions(folder, sample_idx_list, predictions_list, true_list, correct_list, class_probs, name):
    df_dict = {
        "sample_index": sample_idx_list,
        "prediction": predictions_list,
        "true": true_list,
        "correct": correct_list,
    }
    df_dict.update({f"class_{i}_prob": class_i_prob for i, class_i_prob in enumerate(class_probs)})
    df = pd.DataFrame.from_dict(df_dict)
    df = df.set_index("sample_index").sort_index()
    df.to_csv(f"{folder}/{name}-predictions.csv")