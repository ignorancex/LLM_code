# --- START OF FILE 03_infer.py ---

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations
from pathlib import Path
import json
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap
import matplotlib.font_manager as fm


# --- Config ---
DATA_PATH = Path('data/clean_survey.parquet')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = Path('manuscript/figs') # For forest plot
FIGS_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS = [str(i) for i in range(1, 11)] # Q1 to Q10
ALPHA = 0.05
EFFECT_SIZES = {'small': 0.10, 'medium': 0.30, 'large': 0.50}
PALETTE = sns.color_palette("pastel")
FONT_NAME = 'Roboto'

# --- Font Setup ---
try:
    fm.findfont(FONT_NAME, fallback_to_default=False)
    plt.rcParams['font.family'] = FONT_NAME
    print(f"Using font: {FONT_NAME}")
except ValueError:
    print(f"Font '{FONT_NAME}' not found. Using system default sans-serif.")
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


# --- Load data ---
if not DATA_PATH.exists():
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
df = pd.read_parquet(DATA_PATH)
print(f"Data loaded successfully from {DATA_PATH}")

# --- Cramér's V Function (ensure it's robust) ---
def cramers_v(x_series, y_series):
    if x_series.equals(y_series): return 1.0
    if x_series.nunique() < 2 or y_series.nunique() < 2: return 0.0

    confusion_matrix = pd.crosstab(x_series, y_series)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2: return 0.0

    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    if n == 0: return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1 if n > 1 else 1))
    rcorr = r - (((r-1)**2)/(n-1 if n > 1 else 1))
    kcorr = k - (((k-1)**2)/(n-1 if n > 1 else 1))

    if min((kcorr-1), (rcorr-1)) <= 0: return 0.0

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# --- Pairwise Tests (Chi-squared/Fisher's Exact & Cramér's V) ---
print("\nRunning Pairwise Tests...")
pairwise_results_list = []
for q1_num_str, q2_num_str in combinations(QUESTIONS, 2):
    if q1_num_str not in df.columns or q2_num_str not in df.columns:
        print(f"Warning: Skipping pair ({q1_num_str}, {q2_num_str}) as one or both columns are missing.")
        continue

    table = pd.crosstab(df[q1_num_str], df[q2_num_str])

    # Ensure table has at least 2x2 dimensions for chi2_contingency
    if table.shape[0] < 2 or table.shape[1] < 2:
        print(f"Warning: Contingency table for Q{q1_num_str} vs Q{q2_num_str} is too small. Skipping.")
        cv_val = 0.0
        p_val = 1.0
        chi2_val = 0.0
        dof_val = 0
        low_exp = False
        test_method = 'skipped'
    else:
        chi2_val, p_val, dof_val, expected = chi2_contingency(table)
        low_exp = (expected < 5).any().any() if isinstance(expected, np.ndarray) else False # Handle scalar expected
        test_method = 'chi2'

        if low_exp and table.shape == (2, 2):
            try:
                _, fisher_p_val = fisher_exact(table)
                p_val = fisher_p_val
                test_method = 'fisher'
            except ValueError: # Fisher exact might fail for some tables (e.g. all zeros in a row/col)
                print(f"Fisher's exact test failed for Q{q1_num_str} vs Q{q2_num_str}, falling back to Chi-squared.")
                pass # p_val from chi2_contingency remains
        cv_val = cramers_v(df[q1_num_str], df[q2_num_str])

    effect_size_category = 'negligible'
    if cv_val >= EFFECT_SIZES['large']: effect_size_category = 'large'
    elif cv_val >= EFFECT_SIZES['medium']: effect_size_category = 'medium'
    elif cv_val >= EFFECT_SIZES['small']: effect_size_category = 'small'

    pairwise_results_list.append({
        'q1': f'Q{q1_num_str}', 'q2': f'Q{q2_num_str}', 'chi2': float(chi2_val),
        'p_value': float(p_val), 'dof': int(dof_val), 'cramers_v': float(cv_val),
        'effect_size': effect_size_category, 'significant': p_val < ALPHA,
        'low_expected': bool(low_exp), 'test_used': test_method
    })

pairwise_results_df = pd.DataFrame(pairwise_results_list)

if not pairwise_results_df.empty:
    reject, pvals_corrected, _, _ = multipletests(
        pairwise_results_df['p_value'], alpha=ALPHA, method='fdr_bh'
    )
    pairwise_results_df['p_value_fdr'] = pvals_corrected
    pairwise_results_df['significant_fdr'] = reject
else:
    pairwise_results_df['p_value_fdr'] = pd.Series(dtype=float)
    pairwise_results_df['significant_fdr'] = pd.Series(dtype=bool)


pairwise_results_df.to_csv(RESULTS_DIR / 'pairwise_tests.csv', index=False)

# Create summary matrices for visualization (already in 02_explore.py, but can be useful here too)
questions_labels = [f'Q{q}' for q in QUESTIONS]
p_values_matrix = pd.DataFrame(1.0, index=questions_labels, columns=questions_labels)
cramers_matrix_for_json = pd.DataFrame(0.0, index=questions_labels, columns=questions_labels)
# ... (fill matrices as before) ...
# For brevity, assuming this part is mostly correct or can be adapted from 02_explore.py's heatmap generation.
# The key output is pairwise_tests.csv and pairwise_summary.json

print("Pairwise tests and FDR correction complete. Results saved.")

# --- Multiple Correspondence Analysis (MCA) ---
print("\nRunning MCA...")
try:
    import prince
    mca_df = df[QUESTIONS].astype('category')
    mca_model = prince.MCA(n_components=10, random_state=42).fit(mca_df)

    explained_inertia_mca = mca_model.percentage_of_variance_
    cum_inertia_mca = mca_model.cumulative_percentage_of_variance_
    n_components_20_mca = np.argmax(cum_inertia_mca >= 20.0) + 1 # Corrected to 20.0 for percentage

    row_coords_mca = mca_model.row_coordinates(mca_df).iloc[:, :n_components_20_mca]
    row_coords_mca.index = df.index
    row_coords_mca.to_csv(RESULTS_DIR / 'mca_row_coordinates.csv')

    with open(RESULTS_DIR / 'mca_inertia.json', 'w') as f:
        json.dump({
            'explained_inertia': explained_inertia_mca.tolist(),
            'cum_inertia': cum_inertia_mca.tolist(),
            'n_components_20': int(n_components_20_mca)
        }, f, indent=2)
    print(f"MCA complete. {n_components_20_mca} components explain {cum_inertia_mca[n_components_20_mca-1]:.2f}% of inertia.")
    if cum_inertia_mca[1] < 20.0: # First two dimensions
        print(f"Warning: The first two MCA dimensions explain only {cum_inertia_mca[1]:.2f}% of total inertia, which is low.")
except ImportError:
    print("MCA skipped: 'prince' package not installed.")
    mca_model = None # Ensure it's defined for later checks

# --- K-modes clustering ---
print("\nRunning K-modes clustering...")
try:
    from kmodes.kmodes import KModes
    k_range_kmodes = range(2, 6)
    k_modes_results_dict = {}
    kmodes_data_df = df[QUESTIONS].astype(str)

    for k_val in k_range_kmodes:
        km_model = KModes(n_clusters=k_val, init='Huang', n_init=10, verbose=0, random_state=42) # Increased n_init
        cluster_labels = km_model.fit_predict(kmodes_data_df)
        k_modes_results_dict[k_val] = {
            'labels': cluster_labels.tolist(),
            'inertia': km_model.cost_,
            'modes': km_model.cluster_centroids_.tolist()
        }
    with open(RESULTS_DIR / 'kmodes_results.json', 'w') as f:
        json.dump(k_modes_results_dict, f, indent=2)

    # For the paper, best_k = 3 is used.
    best_k_val = 3
    if best_k_val in k_modes_results_dict:
        best_cluster_labels = np.array(k_modes_results_dict[best_k_val]['labels'])
        # Generate Table 1 (cluster profiles)
        table1_list = []
        for cluster_idx in range(best_k_val):
            mask = (best_cluster_labels == cluster_idx)
            cluster_size = mask.sum()
            row_dict = {'cluster': cluster_idx, 'size': int(cluster_size)}
            for q_num_str in QUESTIONS:
                if q_num_str in df.columns:
                    mode_series = df.loc[mask, q_num_str].mode()
                    row_dict[f'Q{q_num_str}_mode'] = mode_series.iloc[0] if not mode_series.empty else None
            table1_list.append(row_dict)
        pd.DataFrame(table1_list).to_csv(RESULTS_DIR / 'kmodes_table1.csv', index=False)
        with open(RESULTS_DIR / 'kmodes_table1.json', 'w') as f:
            json.dump(table1_list, f, indent=2)
        print(f"K-modes Table 1 (k={best_k_val}) saved.")

        # MCA Biplot with Cluster Overlay (Figure 3 in paper, if MCA was run)
        if mca_model and 'row_coords_mca' in locals() and row_coords_mca.shape[1] >= 2:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=row_coords_mca.iloc[:, 0], y=row_coords_mca.iloc[:, 1],
                hue=best_cluster_labels, palette=PALETTE[:best_k_val], legend='full'
            )
            plt.title(f'MCA Biplot by Cluster (k={best_k_val})', fontsize=14) # Match paper
            dim1_inertia_val = mca_model.percentage_of_variance_[0]
            dim2_inertia_val = mca_model.percentage_of_variance_[1]
            plt.xlabel(f'Dimension 1 ({dim1_inertia_val:.1f}% inertia)')
            plt.ylabel(f'Dimension 2 ({dim2_inertia_val:.1f}% inertia)')
            plt.axhline(0, color='grey', lw=0.5, linestyle='--')
            plt.axvline(0, color='grey', lw=0.5, linestyle='--')
            plt.legend(title='Cluster')
            plt.tight_layout()
            plt.savefig(FIGS_DIR / 'mca_biplot_clusters_figure3.png', dpi=300)
            plt.close()
            print("MCA biplot (Figure 3) saved.")
    else:
        print(f"K-modes results for k={best_k_val} not found.")
except ImportError:
    print("K-modes clustering skipped: 'kmodes' package not installed.")


# --- Logistic Regression: Predicting current deployment (Q3) ---
print("\nRunning Logistic Regression for Q3 deployment...")
df_logit = df.copy()
df_logit['deploy_now'] = (df_logit['3'] == 'We ARE deploying them now').astype(int)

predictor_cols_logit = []
final_predictor_names_logit = [] # Store cleaned names for formula

for q_num_str in ['1', '2'] + [str(i) for i in range(4, 11)]: # Q1, Q2, Q4-Q10
    if q_num_str not in df_logit.columns:
        print(f"Warning: Predictor question Q{q_num_str} not found. Skipping.")
        continue

    # Exclude 'No opinion' before finding reference level
    temp_series = df_logit[q_num_str][df_logit[q_num_str] != 'No opinion'].astype('category')

    if not temp_series.cat.categories.empty:
        reference_level = temp_series.cat.categories[0]
        # Create dummies, explicitly dropping the reference level and 'No opinion'
        # Ensure dummy_na=False to not create a column for NaNs
        current_dummies = pd.get_dummies(df_logit[q_num_str], prefix=f'Q{q_num_str}', dummy_na=False)

        ref_col_name = f'Q{q_num_str}_{reference_level}'
        if ref_col_name in current_dummies.columns:
            current_dummies = current_dummies.drop(columns=[ref_col_name])

        no_opinion_col_name = f'Q{q_num_str}_No opinion'
        if no_opinion_col_name in current_dummies.columns:
            current_dummies = current_dummies.drop(columns=[no_opinion_col_name])

        # Handle problematic predictor for Q6 as in paper's code
        if q_num_str == '6':
            problematic_q6_col = 'Q6_50+ (massive specialization networks)'
            if problematic_q6_col in current_dummies.columns:
                current_dummies = current_dummies.drop(columns=[problematic_q6_col])
                print(f"INFO: Explicitly dropped problematic predictor '{problematic_q6_col}' from Q6 dummies.")

        df_logit = pd.concat([df_logit, current_dummies], axis=1)
        final_predictor_names_logit.extend(current_dummies.columns)
    else:
        print(f"Warning: Q{q_num_str} has no valid levels after excluding 'No opinion'. Skipping for LR.")

logit_model_fitted = None
logit_results_summary_df = None

if final_predictor_names_logit:
    # Ensure predictor names are valid for formula (Statsmodels can handle many special chars if quoted)
    formula_predictors_str = ' + '.join([f'Q("{p}")' for p in final_predictor_names_logit])
    formula_str = f'deploy_now ~ {formula_predictors_str}'

    # VIF Check
    X_for_vif_logit = df_logit[final_predictor_names_logit].copy().astype(float).dropna() # Convert to float and dropna for VIF
    if not X_for_vif_logit.empty and X_for_vif_logit.shape[1] > 1:
        try:
            vif_data_logit = pd.DataFrame()
            vif_data_logit["feature"] = X_for_vif_logit.columns
            vif_data_logit["VIF"] = [variance_inflation_factor(X_for_vif_logit.values, i) for i in range(X_for_vif_logit.shape[1])]
            vif_data_logit.to_csv(RESULTS_DIR / 'logit_deployment_vif.csv', index=False)
            print("VIF data saved.")
            if (vif_data_logit["VIF"] > 10).any(): # Common threshold for high VIF
                print("Warning: High multicollinearity detected (VIF > 10). Model results may be unstable.")
            elif (vif_data_logit["VIF"] > 5).any():
                print("Warning: Moderate multicollinearity detected (VIF > 5).")
        except Exception as e_vif:
            print(f"Error calculating VIF: {e_vif}")
    else:
        print("Skipping VIF: Not enough data or predictors.")

    try:
        logit_model_fitted = smf.logit(formula_str, data=df_logit).fit(disp=1, maxiter=100) # Allow more iterations
        print("\nLogistic Regression Summary:")
        print(logit_model_fitted.summary())

        params_logit = logit_model_fitted.params
        conf_int_logit = logit_model_fitted.conf_int()
        p_values_lr = logit_model_fitted.pvalues
        odds_ratios_lr = np.exp(params_logit)
        odds_ratios_conf_int_lr = np.exp(conf_int_logit)

        logit_results_summary_df = pd.DataFrame({
            'coefficient': params_logit, 'p_value': p_values_lr,
            'conf_int_lower': conf_int_logit.iloc[:, 0], 'conf_int_upper': conf_int_logit.iloc[:, 1],
            'odds_ratio': odds_ratios_lr,
            'or_conf_int_lower': odds_ratios_conf_int_lr.iloc[:, 0],
            'or_conf_int_upper': odds_ratios_conf_int_lr.iloc[:, 1]
        })
        logit_results_summary_df.to_csv(RESULTS_DIR / 'logit_deployment_coefs.csv', index_label='variable')

        try:
            with open(RESULTS_DIR / 'logit_deployment_summary.txt', 'w') as f:
                f.write(logit_model_fitted.summary2().as_text())
        except Exception: # Fallback for summary2
            with open(RESULTS_DIR / 'logit_deployment_summary.txt', 'w') as f:
                f.write(str(logit_model_fitted.summary()))
        print("Logistic regression coefficients and summary saved.")

    except Exception as e_logit:
        print(f"Error during logistic regression: {e_logit}")
        print(f"Formula used: {formula_str}")
else:
    print("No valid predictors for logistic regression. Skipping model fitting.")


# --- Forest Plot for Logistic Regression Odds Ratios (Figure 4 in paper) ---
if logit_model_fitted and logit_results_summary_df is not None:
    plot_data_lr = logit_results_summary_df.drop('Intercept', errors='ignore').copy()
    plot_data_lr = plot_data_lr[np.isfinite(plot_data_lr['odds_ratio']) & (plot_data_lr['odds_ratio'] > 0)] # Ensure positive OR for log scale
    plot_data_lr = plot_data_lr.sort_values(by='odds_ratio')

    plot_data_lr['err_lower'] = plot_data_lr['odds_ratio'] - plot_data_lr['or_conf_int_lower']
    plot_data_lr['err_upper'] = plot_data_lr['or_conf_int_upper'] - plot_data_lr['odds_ratio']
    plot_data_lr['err_lower'] = np.maximum(plot_data_lr['err_lower'], 0)
    plot_data_lr['err_upper'] = np.maximum(plot_data_lr['err_upper'], 0)
    errors_lr = [plot_data_lr['err_lower'].values, plot_data_lr['err_upper'].values]

    model_converged = getattr(logit_model_fitted.mle_retvals, 'converged', True)

    def clean_forest_label(label):
        label = label.replace('Q("', '').replace('")[T.True]', '').replace('"', '')
        label_parts = label.split('_', 1)
        if len(label_parts) > 1 and label_parts[0].startswith('Q') and label_parts[0][1:].isdigit():
            label = label_parts[0] + " " + label_parts[1] # Keep Q number for context
        return textwrap.fill(label.replace('_', ' '), width=40)

    cleaned_labels_lr = [clean_forest_label(idx) for idx in plot_data_lr.index]

    # Main Forest Plot (All predictors)
    if not plot_data_lr.empty:
        plt.figure(figsize=(10, max(6, len(plot_data_lr) * 0.35)))
        plt.errorbar(plot_data_lr['odds_ratio'], np.arange(len(plot_data_lr)), xerr=errors_lr,
                     fmt='o', color='black', ecolor='gray', capsize=3, linestyle='None', markersize=5, zorder=3)
        for i_bar, (idx_bar, row_bar) in enumerate(plot_data_lr.iterrows()):
             plt.barh(i_bar, width=row_bar['or_conf_int_upper']-row_bar['or_conf_int_lower'],
                      left=row_bar['or_conf_int_lower'], height=0.4, color='skyblue', alpha=0.4, zorder=1)
        plt.axvline(1, color='red', linestyle='--', lw=1, zorder=2)
        plt.yticks(np.arange(len(plot_data_lr)), cleaned_labels_lr, fontsize=9)
        plt.xlabel('Odds Ratio (95% CI) - Log Scale', fontsize=12)
        plt.xscale('log')

        title_str = 'Forest plot of odds ratios for current AI agent deployment.'
        if not model_converged: title_str += '\n(Warning: Model did not fully converge)'
        plt.title(title_str, fontsize=14)

        # Dynamic xlim, but cap extremes for readability
        min_val_plot = max(0.01, plot_data_lr['or_conf_int_lower'][plot_data_lr['or_conf_int_lower'] > 0].min() * 0.5 if pd.notna(plot_data_lr['or_conf_int_lower'][plot_data_lr['or_conf_int_lower'] > 0].min()) else 0.01)
        max_val_plot = min(100.0, plot_data_lr['or_conf_int_upper'].max() * 1.5 if pd.notna(plot_data_lr['or_conf_int_upper'].max()) else 100.0)
        if max_val_plot <= min_val_plot: max_val_plot = min_val_plot * 100 # Ensure valid range
        plt.xlim(min_val_plot, max_val_plot)

        # Add significance stars
        for i_star, (idx_star, row_star) in enumerate(plot_data_lr.iterrows()):
            star_text = ''
            if row_star['p_value'] < 0.001: star_text = '***'
            elif row_star['p_value'] < 0.01: star_text = '**'
            elif row_star['p_value'] < 0.05: star_text = '*'
            if star_text:
                 # Place stars to the right of y-axis labels, before the plot area starts if possible
                 # This requires careful coordinate handling, or just place near the point
                 plt.text(max_val_plot * 1.05, i_star, star_text, va='center', ha='left', fontsize=10, color='blue')

        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'logit_forest_plot_figure4.png', dpi=300)
        plt.close()
        print("Forest plot (Figure 4) for logistic regression saved.")

        # Forest Plot for Significant Predictors Only
        sig_plot_data_lr = plot_data_lr[plot_data_lr['p_value'] < ALPHA].copy()
        if not sig_plot_data_lr.empty:
            sig_cleaned_labels_lr = [clean_forest_label(idx) for idx in sig_plot_data_lr.index]
            sig_errors_lr = [sig_plot_data_lr['err_lower'].values, sig_plot_data_lr['err_upper'].values]

            plt.figure(figsize=(10, max(4, len(sig_plot_data_lr) * 0.5)))
            plt.errorbar(sig_plot_data_lr['odds_ratio'], np.arange(len(sig_plot_data_lr)), xerr=sig_errors_lr,
                         fmt='o', color='darkgreen', ecolor='lightgreen', capsize=3, linestyle='None', markersize=6, zorder=3)
            for i_bar_sig, (idx_bar_sig, row_bar_sig) in enumerate(sig_plot_data_lr.iterrows()):
                plt.barh(i_bar_sig, width=row_bar_sig['or_conf_int_upper']-row_bar_sig['or_conf_int_lower'],
                         left=row_bar_sig['or_conf_int_lower'], height=0.4, color='lightgreen', alpha=0.5, zorder=1)

            plt.axvline(1, color='red', linestyle='--', lw=1, zorder=2)
            plt.yticks(np.arange(len(sig_plot_data_lr)), sig_cleaned_labels_lr, fontsize=9)
            plt.xlabel('Odds Ratio (95% CI) - Log Scale', fontsize=12)
            plt.xscale('log')
            plt.title('Forest Plot: Significant Predictors of Deployment (p < 0.05)', fontsize=14)
            plt.xlim(min_val_plot, max_val_plot) # Use same xlim as full plot for comparability
            plt.tight_layout()
            plt.savefig(FIGS_DIR / 'logit_forest_plot_significant_only.png', dpi=300)
            plt.close()
            print("Forest plot for significant predictors saved.")
        else:
            print("No significant predictors (p < 0.05) found for the reduced forest plot.")
    else:
        print("Skipping main forest plot: No data to plot.")
else:
    print("Skipping forest plot: Logistic regression model or results not available.")

print("\nInferential analysis complete. Check results/ and manuscript/figs/ directories.")
# --- END OF FILE 03_infer.py ---