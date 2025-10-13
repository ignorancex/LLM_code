from pathlib import Path
import toml
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
import seaborn as sns

BASELINE_FIRST_RESULT_SELECTION = 0.8043775649794802
BASELINE_GPT4O = 0.4269

# Models to evaluate
models = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.5-preview',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4.1',
    'gpt-4.5-preview',
    'o1',
    'o1-mini',
    'o3',
    'o3-mini',
    'o4-mini',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-pro-exp-02-05',
    'gemini-2.5-pro-exp-03-25',
    'gemma-3-4b-it',
    'gemma-3-27b-it',
    'gemma-2-27b-it',
    'DeepSeek-V3',
    'DeepSeek-R1',
    'DeepSeek-R1-Distill-Qwen-1.5B',
    'DeepSeek-R1-Distill-Qwen-7B',
    'Llama-3.3-70B-Instruct',
    'Llama-3.2-3B-Instruct',
    'Llama-3.2-1B-Instruct',
    'Llama-3.1-405B-Instruct',
    'Llama-3.1-70B-Instruct',
    'Llama-3-70B-Instruct',
    'Llama-4-Maverick-Instruct-Basic',
    'Llama-4-Scout-Instruct-Basic',
    'sabia-3',
    'sabiazinho-3',
    'QwQ-32B',
]

# Define relevant directories
plots_path = Path('plots')
plots_path.mkdir(exist_ok=True)

# Load relevant data
with open('llms_metadata.toml', 'r') as f:
    config = toml.load(f)

model_prices = {}

for provider in config['models'].keys():
    for model in config['models'][provider].keys():
        if model == 'provider_name': continue
        model_prices[model] = config['models'][provider][model]

# Helper functions
def load_data(metrics_path: str, results_path: str):
    """
    Load the computed metrics and LLM token usage results.
    
    Args:
        metrics_path: Path to computed_metrics.csv
        results_path: Path to llms_results.csv
        
    Returns:
        cm: DataFrame with per-model, per-top-k F1 scores
        lr: DataFrame with per-model, per-top-k token usage
    """
    cm = pd.read_csv(metrics_path)
    lr = pd.read_csv(results_path)
    return cm, lr

# Paths to your CSV files
metrics_csv = 'metrics/computed_metrics.csv'
results_csv = 'data/llms_results.csv'

# Load data
cm, lr = load_data(metrics_csv, results_csv)




# PLOT F1-SCORE VS AVERAGE TOKEN USAGE PER RESPONSE #################################

def summarize_models(cm: pd.DataFrame, lr: pd.DataFrame):
    """
    For each model, find the top_k with the maximum F1 score, then compute
    the average total token usage at that top_k across all queries.
    
    Args:
        cm: DataFrame containing columns ['model', 'top_k', 'f1_score']
        lr: DataFrame containing columns like '{model}-top-{top_k}-total_tokens'
        
    Returns:
        summary: DataFrame with columns ['model', 'top_k_max', 'f1_score_max', 'avg_total_tokens']
    """
    # 1) Find the row index of max F1 score per model
    idx = cm.groupby('model')['f1_score'].idxmax()
    # 2) Extract model, optimal top_k, and max F1
    summary = cm.loc[idx, ['model', 'top_k', 'f1_score']].rename(
        columns={'top_k': 'top_k_max', 'f1_score': 'f1_score_max'}
    ).reset_index(drop=True)

    # filter models of interest
    summary = summary[summary['model'].isin(models)]
    
    # 3) Compute average token usage for each model at its optimal top_k
    avg_tokens = []
    for _, row in summary.iterrows():
        model = row['model']
        top_k = row['top_k_max']
        col_name = f"{model}-top-{top_k}-total_tokens"
        if col_name in lr.columns:
            avg = lr[col_name].mean()
        else:
            avg = float('nan')
        avg_tokens.append(avg)
    summary['avg_total_tokens'] = avg_tokens
    
    # 4) Drop any models lacking token usage data
    summary = summary.dropna(subset=['avg_total_tokens']).reset_index(drop=True)

    # 5) Filter models of interest
    summary = summary[summary['model'].isin(models)]
    
    return summary

def plot_summary(summary: pd.DataFrame, figsize=(12, 8)):
    """
    Create a scatter plot of average token usage vs. max F1 score, annotating each point
    with the model name and using adjustText to minimize label overlap. Y-axis spans 0 to 1.
    
    Args:
        summary: DataFrame from summarize_models()
        figsize: Tuple specifying figure size
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    x = summary['avg_total_tokens']
    y = summary['f1_score_max']
    
    # Scatter points
    ax.scatter(x, y, marker='.')
    
    # Prepare annotations
    texts = []
    for _, row in summary.iterrows():
        texts.append(
            ax.annotate(
                text=row['model'], 
                xy=(row['avg_total_tokens'], row['f1_score_max']),              
                fontsize=8
            )
        )
    
    # Adjust text to avoid overlaps
    adjust_text(
        texts,
        x=x,
        y=y,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        force_text=(1,6),
        max_move=200,
    )
    
    ax.set_xlabel('Mean token usage')
    ax.set_ylabel('Max F1-score')
    ax.set_ylim(0, 1)  # Force y-axis to span from 0 to 1
    # ax.set_title('Uso médio de tokens vs. F1-score máximo')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(BASELINE_FIRST_RESULT_SELECTION, color='orange', linestyle='--', alpha=0.5,label="first result selection")
    ax.axhline(BASELINE_GPT4O, color='gray', linestyle='--', alpha=0.5,label="gpt-4o without search results")
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(plots_path /'token_usage_vs_f1.png')

# Paths to your CSV files
metrics_csv = 'metrics/computed_metrics.csv'
results_csv = 'data/llms_results.csv'

# Load data
cm, lr = load_data(metrics_csv, results_csv)

# Summarize per-model performance
summary = summarize_models(cm, lr)

# Plot the results
plot_summary(summary)



# PLOT F1-SCORE VS AVERAGE PRICE PER RESPONSE #################################

def summarize_models(cm: pd.DataFrame, lr: pd.DataFrame):
    """
    For each model, find the top_k with the maximum F1 score, then compute
    the average price per response considering the model price and token usage.
    
    Args:
        cm: DataFrame containing columns ['model', 'top_k', 'f1_score']
        lr: DataFrame containing columns like '{model}-top-{top_k}-total_tokens'
        
    Returns:
        summary: DataFrame with columns ['model', 'top_k_max', 'f1_score_max', 'avg_total_tokens']
    """
    # 1) Find the row index of max F1 score per model
    idx = cm.groupby('model')['f1_score'].idxmax()
    
    # 2) Extract model, optimal top_k, and max F1
    summary = cm.loc[idx, ['model', 'top_k', 'f1_score']].rename(
        columns={'top_k': 'top_k_max', 'f1_score': 'f1_score_max'}
    ).reset_index(drop=True)

    # filter models of interest
    summary = summary[summary['model'].isin(models)]
    
    # 3) Compute average input token usage for each model at its optimal top_k
    avg_input_tokens = []
    for _, row in summary.iterrows():
        model = row['model']
        top_k = row['top_k_max']
        col_name = f"{model}-top-{top_k}-input_tokens"
        if col_name in lr.columns:
            avg = lr[col_name].mean()
        else:
            avg = float('nan')
        avg_input_tokens.append(avg)
    summary['avg_input_tokens'] = avg_input_tokens
    
    # 4) Compute average input token usage for each model at its optimal top_k
    avg_output_tokens = []
    for _, row in summary.iterrows():
        model = row['model']
        top_k = row['top_k_max']
        col_name = f"{model}-top-{top_k}-output_tokens"
        if col_name in lr.columns:
            avg = lr[col_name].mean()
        else:
            avg = float('nan')
        avg_output_tokens.append(avg)
    summary['avg_output_tokens'] = avg_output_tokens

    # 5) Compute average price per response for each model at its optimal top_k
    avg_price_per_response = []
    for _, row in summary.iterrows():
        model = row['model']
        try:
            avg_input_tks = row['avg_input_tokens']
            avg_output_tks = row['avg_output_tokens']

            if not model_prices[model]['input_price'] or not model_prices[model]['output_price']:
                avg_price_per_response.append(float('nan'))
                continue

            avg_input_price = model_prices[model]['input_price']*10**-6
            avg_output_price = model_prices[model]['output_price']*10**-6
            avg_price_per_response.append((avg_input_price*avg_input_tks + avg_output_price*avg_output_tks)*10**3)
        except:
            avg_price_per_response.append(float('nan'))
    summary['avg_price_per_response'] = avg_price_per_response
    
    # 4) Drop any models lacking token usage data
    summary = summary.dropna(subset=['avg_price_per_response']).reset_index(drop=True)
    return summary

def plot_summary(summary: pd.DataFrame, figsize=(12, 8)):
    """
    Create a scatter plot of average token usage vs. max F1 score, annotating each point
    with the model name and using adjustText to minimize label overlap. Y-axis spans 0 to 1.
    
    Args:
        summary: DataFrame from summarize_models()
        figsize: Tuple specifying figure size
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    x = summary['avg_price_per_response']
    y = summary['f1_score_max']
    
    # Scatter points
    ax.scatter(x, y, marker='x')
    ax.set_xscale('log')
    
    # Prepare annotations
    texts = []
    for _, row in summary.iterrows():
        texts.append(
            ax.annotate(
                text=row['model'],
                xy=(row['avg_price_per_response'], row['f1_score_max']),
                fontsize=8
            )
        )
    
    # Adjust text to avoid overlaps
    adjust_text(
        texts,
        x=x,
        y=y,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        expand_text=(1.5, 2.0),
        expand_points=(1.5, 2.0),
        force_text=(1.0,1.0),
        force_points=(0.3, 0.3),
        max_iter=200,
        expand=(1.5,1.5)
    )
    
    ax.set_xlabel('Mean price (USD) per 1000 responses')
    ax.set_ylabel('Max F1-score')
    ax.set_ylim(0.65, 0.9)  # Force y-axis to span from 0 to 1
    # ax.set_title('Preço médio de 1000 respostas vs. F1-score máximo')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(BASELINE_FIRST_RESULT_SELECTION, color='orange', linestyle='--', alpha=0.5,label="first result selection")
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(plots_path / 'price_per_response_vs_f1.png')

# Paths to your CSV files
metrics_csv = 'metrics/computed_metrics.csv'
results_csv = 'data/llms_results.csv'

# Summarize per-model performance
summary = summarize_models(cm, lr)

# Plot the results
plot_summary(summary)



# PLOT F1-SCORE VS MODEL SIZE #########################################################

def summarize_models(cm: pd.DataFrame, lr: pd.DataFrame):
    """
    For each model, find the top_k with the maximum F1 score, then compute
    the average price per response considering the model price and token usage.
    
    Args:
        cm: DataFrame containing columns ['model', 'top_k', 'f1_score']
        lr: DataFrame containing columns like '{model}-top-{top_k}-total_tokens'
        
    Returns:
        summary: DataFrame with columns ['model', 'top_k_max', 'f1_score_max', 'avg_total_tokens']
    """
    # 1) Find the row index of max F1 score per model
    idx = cm.groupby('model')['f1_score'].idxmax()
    
    # 2) Extract model, optimal top_k, and max F1
    summary = cm.loc[idx, ['model', 'top_k', 'f1_score']].rename(
        columns={'top_k': 'top_k_max', 'f1_score': 'f1_score_max'}
    ).reset_index(drop=True)

    # filter models of interest
    summary = summary[summary['model'].isin(models)]

    # 3) Extract model's number of parameters
    n_params = []
    for _, row in summary.iterrows():
        model = row['model']
        model_n_params = model_prices[model]['num_params']
        if model_n_params:
            n_params.append(model_n_params)
        else: 
            n_params.append(None)
    summary['n_params'] = n_params
    
    # 4) Drop any models lacking token usage data
    summary = summary.dropna(subset=['n_params']).reset_index(drop=True)
    return summary

def plot_summary(summary: pd.DataFrame, figsize=(12, 8)):
    """
    Create a scatter plot of average token usage vs. max F1 score, annotating each point
    with the model name and using adjustText to minimize label overlap. Y-axis spans 0 to 1.
    
    Args:
        summary: DataFrame from summarize_models()
        figsize: Tuple specifying figure size
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    x = summary['n_params']
    y = summary['f1_score_max']
    
    # Scatter points
    ax.scatter(x, y, marker='o')
    ax.set_xscale('log')
    
    # Prepare annotations
    texts = []
    for _, row in summary.iterrows():
        texts.append(
            ax.text(
                row['n_params'], 
                row['f1_score_max'], 
                row['model'], 
                fontsize=9
            )
        )
    
    # Adjust text to avoid overlaps
    adjust_text(
        texts,
        x=x,
        y=y,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        # expand_text=(3,3),
        # expand_points=(2.0, 2.0),
        force_text=(1,6),
        # force_points=(0.3, 0.3),
        # max_iter=200,
        # expand=(2,2),
        max_move=200,
    )
    
    ax.set_xlabel('Number of parameters (billions)')
    ax.set_ylabel('Max F1-score')
    ax.set_ylim(0.0, 1.0)  # Force y-axis to span from 0 to 1
    # ax.set_title('Preço médio de 1000 respostas vs. F1-score máximo')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(BASELINE_FIRST_RESULT_SELECTION, color='orange', linestyle='--', alpha=0.5,label="first result selection")
    ax.axhline(BASELINE_GPT4O, color='gray', linestyle='--', alpha=0.5,label="gpt-4o without search results")
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(plots_path /    'n_params_vs_f1.png')

# Summarize per-model performance
summary = summarize_models(cm, lr)

# Plot the results
plot_summary(summary)



# PLOT F1-SCORE VS AVERAGE TIME PER RESPONSE #########################################################

def summarize_models(cm: pd.DataFrame, lr: pd.DataFrame):
    """
    For each model, find the top_k with the maximum F1 score, then compute
    the average price per response considering the model price and token usage.
    
    Args:
        cm: DataFrame containing columns ['model', 'top_k', 'f1_score']
        lr: DataFrame containing columns like '{model}-top-{top_k}-total_tokens'
        
    Returns:
        summary: DataFrame with columns ['model', 'top_k_max', 'f1_score_max', 'avg_total_tokens']
    """
    # 1) Find the row index of max F1 score per model
    idx = cm.groupby('model')['f1_score'].idxmax()
    
    # 2) Extract model, optimal top_k, and max F1
    summary = cm.loc[idx, ['model', 'top_k', 'f1_score']].rename(
        columns={'top_k': 'top_k_max', 'f1_score': 'f1_score_max'}
    ).reset_index(drop=True)

    # 3) Compute average time to response for each model at its optimal top_k
    avg_time_per_response = []
    for _, row in summary.iterrows():
        model = row['model']
        top_k = row['top_k_max']
        col_name = f"{model}-top-{top_k}-timedelta"
        if col_name in lr.columns:
            try:
                if type(lr[col_name].iloc[0])!=str:
                    avg = lr[col_name].mean()
                else:
                    avg = pd.to_timedelta(lr[col_name]).dropna().mean().total_seconds()
            except:
                avg = float('nan')
        else:
            avg = float('nan')
        avg_time_per_response.append(avg)
    summary['avg_time_per_response'] = avg_time_per_response
    
    # 4) Drop any models lacking token usage data
    summary = summary.dropna(subset=['avg_time_per_response']).reset_index(drop=True)

    summary = summary[summary['model'].isin(models)]
    return summary

def plot_summary(summary: pd.DataFrame, figsize=(12, 8)):
    """
    Create a scatter plot of average token usage vs. max F1 score, annotating each point
    with the model name and using adjustText to minimize label overlap. Y-axis spans 0 to 1.
    
    Args:
        summary: DataFrame from summarize_models()
        figsize: Tuple specifying figure size
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    x = summary['avg_time_per_response']
    y = summary['f1_score_max']
    
    # Scatter points
    ax.scatter(x, y, marker='.')
    ax.set_xscale('log')
    
    # Prepare annotations
    texts = []
    for _, row in summary.iterrows():
        texts.append(
            ax.text(
                row['avg_time_per_response'], 
                row['f1_score_max'], 
                row['model'], 
                fontsize=8
            )
        )
    
    # Adjust text to avoid overlaps
    adjust_text(
        texts,
        x=x,
        y=y,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        # expand_text=(3,3),
        # expand_points=(2.0, 2.0),
        force_text=(1,6),
        # force_points=(0.3, 0.3),
        # max_iter=200,
        # expand=(2,2),
        max_move=200,
    )
    
    ax.set_xlabel('Mean time (seconds) per response')
    ax.set_ylabel('Max F1-score')
    ax.set_ylim(0.0, 1.0)  # Force y-axis to span from 0 to 1
    # ax.set_title('Tempo médio por resposta vs. F1-score máximo')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(BASELINE_FIRST_RESULT_SELECTION, color='orange', linestyle='--', alpha=0.5,label="first result selection")
    ax.axhline(BASELINE_GPT4O, color='gray', linestyle='--', alpha=0.5,label="gpt-4o without search results")
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(plots_path / 'timedelta_per_response_vs_f1.png')

# Summarize per-model performance
summary = summarize_models(cm, lr)

# Plot the results
plot_summary(summary)



# PLOT OTHER RELEVANT METRICS ########################################################

cm = cm[cm['model'].isin(models)]

# Define metrics to plot
metrics = [
    ('right_answer_format', 'Proporção de respostas dentro do formato solicitado por modelo (%)'),
    ('valid_icpc_2_code_rate', 'Proporção de respostas com código CIAP-2 válido por modelo (%)'),
    ('selected_code_is_in_search_results', 'Proporção de respostas com código selecionado entre os resultados por modelo (%)')
]

# Generate wide, short bar charts with percentage annotations and no top/right spines
for metric, title in metrics:
    means = cm.groupby('model')[metric].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    bars = ax.bar(means.index, means.values * 100)
    # Annotate each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            xy = (bar.get_x() + bar.get_width() / 2, height + 1),
            text=f"{height:.1f}%",
            ha='center',
            va='bottom',
            fontsize=6
        )
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set labels and title
    # ax.set_title(title, pad=15)
    ax.set_xlabel('Model')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 110)
    ax.set_xticks(range(len(means.index)))
    ax.set_xticklabels(means.index, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(plots_path / f'{metric}_per_model.png')




# PLOT HEATMAPS OF THE EVALUATION DATASET ######################

# Load datasets
eval_df = pd.read_csv("data/eval_dataset.csv")
valid_codes_df = pd.read_csv("data/icpc-2_partial.csv")

# Extract valid ICPC-2 codes
valid_codes = set(valid_codes_df['code'].dropna().astype(str).str.strip())
valid_regular_codes = [code for code in valid_codes if not code.startswith("-")]
valid_procedure_codes = [code for code in valid_codes if code.startswith("-") and code[1:].isdigit()]

# Count code frequencies from 'relevant_results'
def extract_codes(value):
    if pd.isna(value):
        return []
    return [code.strip() for code in value.split('|') if code.strip()]

eval_df["parsed_codes"] = eval_df["relevant_results"].apply(extract_codes)
all_codes = [code for codes in eval_df["parsed_codes"] for code in codes]
procedure_codes = [code for code in all_codes if code.startswith("-")]
regular_codes = [code for code in all_codes if not code.startswith("-")]
regular_freq = pd.Series(regular_codes).value_counts()
procedure_freq = pd.Series([int(code[1:]) for code in procedure_codes]).value_counts()

# ICPC-2 chapter mapping
chapters = {
    "A": "A - General and Unspecified", "B": "B - Blood, Blood Forming Organs and Immune Mechanism", "D": "D - Digestive", "F": "F - Eye", "H": "H - Ear",
    "K": "K - Cardiovascular", "L": "L - Musculoskeletal", "N": "N - Neurological", "P": "P - Psychological",
    "R": "R - Respiratory", "S": "S - Skin", "T": "T - Endocrine/Metabolic and Nutritional", "U": "U - Urological",
    "W": "W - Pregnancy, Childbearing, Family Planning", "X": "X - Female Genital", "Y": "Y - Male Genital", 
    "Z": "Z - Social problems"
}

# Determine valid numeric codes per chapter
valid_chapter_codes = {k: set() for k in chapters}
for code in valid_regular_codes:
    if len(code) >= 3 and code[0] in chapters and code[1:].isdigit():
        valid_chapter_codes[code[0]].add(int(code[1:]))

# Create chapter-code frequency matrix
chapter_code_matrix = {chapters[k]: [np.nan]*99 for k in chapters}
for letter, nums in valid_chapter_codes.items():
    chapter_name = chapters[letter]
    # chapter_name = letter
    for i in range(99):
        code_num = i + 1
        full_code = f"{letter}{code_num:02d}"
        if code_num in nums:
            chapter_code_matrix[chapter_name][i] = regular_freq.get(full_code, 0)
        else:
            chapter_code_matrix[chapter_name][i] = -1

# Create DataFrame for heatmap
heatmap_df = pd.DataFrame(chapter_code_matrix, index=range(1, 100)).transpose()

# Special procedures heatmap
proc_range = range(30, 70)
proc_row = []
for code_num in proc_range:
    full_code = f"-{code_num}"
    if full_code in valid_codes:
        proc_row.append(procedure_freq.get(code_num, 0))
    else:
        proc_row.append(-1)
special_proc_df = pd.DataFrame([proc_row], index=[""], columns=proc_range)

# Plot regular ICPC-2 heatmap
heatmap_values = heatmap_df.loc[:, [i for i in range(1, 100) if i < 30 or i > 69]]
invalid_mask = heatmap_values == -1
valid_mask = heatmap_values.isna() | (heatmap_values == 0)

fig = plt.figure(figsize=(20, 6))
ax = sns.heatmap(
    heatmap_values.fillna(0).mask(invalid_mask),
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Frequency'},
    square=True,
    mask=valid_mask & ~invalid_mask,
    linecolor="#ffffff"
)
for i in range(heatmap_values.shape[0]):
    for j in range(heatmap_values.shape[1]):
        if invalid_mask.iloc[i, j]:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='#e8e6e1', lw=0))

plt.ylabel("Chapters", fontsize=12)
plt.xlabel("Numeric code", fontsize=12)
plt.tight_layout()
# save plot
fig.savefig(plots_path / 'heatmap_1.png', dpi=300, bbox_inches='tight')

# Plot special procedures heatmap
special_invalid_mask = special_proc_df == -1
special_valid_mask = special_proc_df.isna() | (special_proc_df == 0)

fig = plt.figure(figsize=(10, 3))
ax2 = sns.heatmap(
    special_proc_df.fillna(0).mask(special_invalid_mask),
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={'label': 'Frequency'},
    square=True,
    mask=special_valid_mask & ~special_invalid_mask,
    linecolor="#ffffff"
)
for j in range(special_proc_df.shape[1]):
    if special_invalid_mask.iloc[0, j]:
        ax2.add_patch(plt.Rectangle((j, 0), 1, 1, fill=True, color='#e8e6e1', lw=0))

plt.ylabel("Process codes", fontsize=12)
plt.xlabel("Numeric code", fontsize=12)
plt.tight_layout()
fig.savefig(plots_path / 'heatmap_2.png', dpi=300, bbox_inches='tight')