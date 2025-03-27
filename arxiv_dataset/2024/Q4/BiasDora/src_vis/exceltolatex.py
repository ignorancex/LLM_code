import pandas as pd
import os

# Define file paths
file_paths = {
    "t2t": {
        "1": "../analysis/closed_source/t2t_gpt4/setting1_pvalue_significant_tfidf_sentiments.csv",
        "2": "../analysis/closed_source/t2t_gpt4/setting2_pvalue_significant_tfidf_sentiments.csv",
        "3": "../analysis/closed_source/t2t_gpt4/setting3_pvalue_significant_tfidf_sentiments.csv",
        "4": "../analysis/closed_source/t2t_gpt4/setting4_pvalue_significant_tfidf_sentiments.csv",
        "5": "../analysis/closed_source/t2t_gpt4/setting5_pvalue_significant_tfidf_sentiments.csv"
    },
    "i2t": {
        "1": "../analysis/closed_source/i2t_gpt4o/setting1_pvalue_significant_tfidf_sentiments.csv",
        "2": "../analysis/closed_source/i2t_gpt4o/setting2_pvalue_significant_tfidf_sentiments.csv",
        "3": "../analysis/closed_source/i2t_gpt4o/setting3_pvalue_significant_tfidf_sentiments.csv",
        "4": "../analysis/closed_source/i2t_gpt4o/setting4_pvalue_significant_tfidf_sentiments.csv",
        "5": "../analysis/closed_source/i2t_gpt4o/setting5_pvalue_significant_tfidf_sentiments.csv"
    }
}

# Output directory
output_dir = "../src_vis/appendix/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process files and extract negative sentiment pairs
for mode, settings in file_paths.items():
    sentiment_column = 'concatenated_sentence_sentiment' if mode == 't2t' else 'word_sentiment'
    for setting, path in settings.items():
        df = pd.read_csv(path)
        # Filter for negative sentiments using the appropriate column
        negative_df = df[df[sentiment_column] == 'NEGATIVE']
        # Format output filename based on input filename and include the dataset type
        base_filename = os.path.basename(path)
        output_filename = f"{output_dir}{base_filename.split('.')[0]}_{mode}_negative_associations.txt"
        latex_pairs = [f"{row['target']}$\\leftrightarrow${row['word']}" for _, row in negative_df.iterrows()]
        
        # Save to a file
        with open(output_filename, 'w') as f:
            f.write(", ".join(latex_pairs))
        
        print(f"Output written to {output_filename}")
