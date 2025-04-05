import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def grouped_evaluation(results_df, group_by_columns):
    grouped_results = {}
    for name, group in results_df.groupby(group_by_columns):
        if len(group['Assay Qualitative Measure'].unique()) == 2:
            grouped_results[name] = get_performance_metrics(group)
        else:
            weight_sum = np.sum(1/group['MHC Allele Prediction - Count'].to_numpy())
            grouped_results[name] = {'count': 0, 'skipped_count': weight_sum, 'eval': False}
    return grouped_results


def merge_pep_sources(results_df, group_by_columns, source_column='Epitope Parent Species'):
    grouped_df_list = []
    for pep_source, group_df in results_df.groupby(group_by_columns):
        if len(group_df['Assay Qualitative Measure'].unique()) == 1:
            group_df[source_column] = 'eval default peptide source'
        grouped_df_list.append(group_df)
    return pd.concat(grouped_df_list)


def merge_mhc_alleles(results_df, group_by_columns):
    grouped_df_list = []
    for pep_source, group_df in results_df.groupby(group_by_columns):
        if len(group_df['Assay Qualitative Measure'].unique()) == 1:
            group_df['MHC Allele Prediction'] = group_df['MHC Class'].apply(
                lambda mhc_class: 'eval default allele I' if mhc_class == 'I' else 'eval default allele II')
        grouped_df_list.append(group_df)
    return pd.concat(grouped_df_list)


def select_evaluation_groups(grouped_results, selected_allele='all', selected_peptide_source='all'):
    results_selection = grouped_results
    if type(list(results_selection.keys())[0]) == tuple:
        if selected_allele != 'all':
            results_selection = {(allele, peptide_source): results for (allele, peptide_source), results in results_selection.items()
                                 if allele == selected_allele}
        if selected_peptide_source != 'all':
            results_selection = {(allele, peptide_source): results for (allele, peptide_source), results in results_selection.items()
                                 if peptide_source == selected_peptide_source}
    else:
        if selected_allele != 'all':
            results_selection = {allele: results for allele, results in results_selection.items() if allele == selected_allele}
        if selected_peptide_source != 'all':
            results_selection = {peptide_source: results for peptide_source, results in results_selection.items() if peptide_source == selected_peptide_source}
    return results_selection


def get_performance_metrics(results_df):
    y_true = results_df['Assay Qualitative Measure'].to_numpy()
    y_scores = results_df['Prediction'].to_numpy()
    sample_weight = 1/results_df['MHC Allele Prediction - Count'].to_numpy()
    AP = average_precision_score(y_true, y_scores, pos_label='Positive', sample_weight=sample_weight)
    ROC_AUC = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    results = {'AP': AP, 'ROC AUC': ROC_AUC, 'count': np.sum(sample_weight), 'skipped_count': 0, 'eval': True}
    return results


def combine_group_results(grouped_results):
    instance_count = sum([results['count'] for results in grouped_results.values()])
    skipped_instance_count = sum([results['skipped_count'] for results in grouped_results.values()])
    corrected_scores = {'count': instance_count, 'skipped_count': skipped_instance_count}
    if instance_count == 0:
        return {**corrected_scores, 'ROC AUC': None, 'AP': None}
    for metric in ['ROC AUC', 'AP']:
        weighted_scores = sum([results[metric] * results['count'] for results in grouped_results.values() if results['eval']])
        corrected_score = weighted_scores/instance_count
        corrected_scores[metric] = corrected_score
    return corrected_scores


def recompute_allele_counts(df):
    grouped_df_list = []
    for _, group_df in df.groupby(['Epitope Description']):
        if len(group_df) != group_df['MHC Allele Prediction - Count'].iloc[0]:
            group_df['MHC Allele Prediction - Count'] = len(group_df)
        grouped_df_list.append(group_df)
    return pd.concat(grouped_df_list)


def get_source_evaluation_dict(prediction_full_df, description):
    source_result_dict = {}
    # sources = ['Human betaherpesvirus 6B', 'Dengue virus', 'Homo sapiens', 'unknown',
    #                    'Human betaherpesvirus 5', 'Severe acute respiratory syndrome coronavirus 2', 'Phleum pratense',
    #                    'Influenza A virus', 'Hepacivirus C', 'Vaccinia virus',
    #                    'Mycobacterium tuberculosis', 'Human gammaherpesvirus 4',
    #                    'Alphapapillomavirus 9', 'Plasmodium falciparum']
    sources = ['Human immunodeficiency virus 1', 'Trypanosoma cruzi', 'Hepacivirus C',
               'Vaccinia virus', 'unknown', 'Human betaherpesvirus 6B', 'Dengue virus',
               'Severe acute respiratory syndrome coronavirus 2', 'Phleum pratense',
               'Human betaherpesvirus 5', 'Homo sapiens']
    # sources = ['Vaccinia virus', 'Dengue virus', 'Human betaherpesvirus 6B', 'Human betaherpesvirus 5', 'Hepacivirus C', 'Homo sapiens']

    # performance on the selected peptide sources
    prediction_source_df = prediction_full_df.loc[prediction_full_df['Epitope Parent Species'].isin(sources)]

    # performance on all peptide sources (not only the selected ones)
    # prediction_source_df = prediction_full_df

    prediction_I_df = prediction_source_df.loc[(prediction_source_df['MHC Class'] == 'I')]
    prediction_II_df = prediction_source_df.loc[(prediction_source_df['MHC Class'] == 'II')]

    eval_list = [('MHC I+II', prediction_source_df)]
    if len(prediction_I_df) > 0:
        eval_list.append(('MHC I', prediction_I_df))
    if len(prediction_II_df) > 0:
        eval_list.append(('MHC II', prediction_II_df))

    for MHC_text, prediction_df in eval_list:
        df_allele_merged = merge_mhc_alleles(prediction_df, ['MHC Allele Prediction'])
        prediction_df_both_merged = merge_pep_sources(df_allele_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])
        prediction_df_both_merged = merge_mhc_alleles(prediction_df_both_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])

        # dataset without default alleles and without default peptide sources
        default_selection = prediction_df_both_merged['MHC Allele Prediction'].isin(['eval default allele I', 'eval default allele II']) | \
                            (prediction_df_both_merged['Epitope Parent Species'] == 'eval default peptide source')
        prediction_df_both_merged_wo_default = recompute_allele_counts(prediction_df_both_merged.loc[~default_selection])

        # detection of both peptide source and MHC allele shortcuts without default alleles + default peptide sources
        allele_pep_source_grouped_results_wo_default = grouped_evaluation(prediction_df_both_merged_wo_default, ['MHC Allele Prediction', 'Epitope Parent Species'])
        allele_pep_source_results_wo_default = combine_group_results(allele_pep_source_grouped_results_wo_default)
        source_result_dict[f'{MHC_text} - All sources/{description}'] = allele_pep_source_results_wo_default

    for source in sources:
        prediction_source_df = prediction_full_df.loc[prediction_full_df['Epitope Parent Species'] == source]
        if len(prediction_source_df) == 0:
            continue
        prediction_I_df = prediction_source_df.loc[(prediction_source_df['MHC Class'] == 'I')]
        prediction_II_df = prediction_source_df.loc[(prediction_source_df['MHC Class'] == 'II')]
        class_I_pos = len(prediction_I_df.loc[prediction_I_df['Assay Qualitative Measure'] == 'Positive'])
        class_I_neg = len(prediction_I_df.loc[prediction_I_df['Assay Qualitative Measure'] == 'Negative'])
        class_II_pos = len(prediction_II_df.loc[prediction_II_df['Assay Qualitative Measure'] == 'Positive'])
        class_II_neg = len(prediction_II_df.loc[prediction_II_df['Assay Qualitative Measure'] == 'Negative'])
        eval_list = [('MHC I+II', prediction_source_df)]

        if class_I_pos > 0 and class_I_neg > 0:
            eval_list.append(('MHC I', prediction_I_df))

        if class_II_pos > 0 and class_II_neg > 0:
            eval_list.append(('MHC II', prediction_II_df))

        for MHC_text, prediction_df in eval_list:
            df_allele_merged = merge_mhc_alleles(prediction_df, ['MHC Allele Prediction'])
            allele_default_selection = df_allele_merged['MHC Allele Prediction'].isin(['eval default allele I', 'eval default allele II'])
            df_allele_merged_wo_default = df_allele_merged.loc[~allele_default_selection]
            count_pos = len(df_allele_merged_wo_default.loc[df_allele_merged_wo_default['Assay Qualitative Measure'] == 'Positive'])
            count_neg = len(df_allele_merged_wo_default.loc[df_allele_merged_wo_default['Assay Qualitative Measure'] == 'Negative'])
            if count_pos > 0 and count_neg > 0:
                df_allele_merged_wo_default = recompute_allele_counts(df_allele_merged_wo_default)
                allele_grouped_results_wo_default = grouped_evaluation(df_allele_merged_wo_default, ['MHC Allele Prediction'])
                results_wo_default = combine_group_results(allele_grouped_results_wo_default)
                source_result_dict[f'{MHC_text} - {source}/{description}'] = results_wo_default
    return source_result_dict

def get_allele_evaluation_dict(prediction_full_df, description):
    source_result_dict = {}
    mhc_alleles_I = ['default allele I', 'HLA-C*07:02', 'HLA-A*02:01', 'HLA-A*11:01', 'HLA-B*44:02', 'HLA-B*07:02', 'HLA-A*29:02',
                     'HLA-A*24:07', 'HLA-B*08:01', 'HLA-A*33:01']
    mhc_alleles_II = ['default allele II', 'HLA-DRA1*01:01-DRB1*16:02', 'HLA-DRA1*01:01-DRB3*02:02', 'HLA-DPA1*01:03-DPB1*02:01',
                      'HLA-DRA1*01:01-DRB1*11:01', 'HLA-DRA1*01:01-DRB1*09:01', 'HLA-DRA1*01:01-DRB1*03:01',
                      'HLA-DRA1*01:01-DRB1*12:01', 'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DRA1*01:01-DRB5*01:01']

    prediction_full_df_subset = prediction_full_df.loc[prediction_full_df['MHC Allele Prediction'].isin(mhc_alleles_I+mhc_alleles_II)]
    prediction_full_df_subset = recompute_allele_counts(prediction_full_df_subset)

    mhc_alleles = mhc_alleles_I + mhc_alleles_II + ['MHC Class I+II', 'MHC Class I', 'MHC Class II']

    for mhc_allele in mhc_alleles:
        if mhc_allele in ['MHC Class I', 'MHC Class II']:
            mhc_class = 'I' if mhc_allele == 'MHC Class I' else 'II'
            prediction_df = prediction_full_df_subset.loc[prediction_full_df_subset['MHC Class'] == mhc_class]
        elif mhc_allele == 'MHC Class I+II':
            prediction_df = prediction_full_df_subset
        else:
            prediction_df = prediction_full_df_subset.loc[prediction_full_df_subset['MHC Allele Prediction'] == mhc_allele]

        if len(prediction_df) == 0:
            continue

        # evaluation with default values and without shortcut detection
        # keep allele counts from subset to merge the per-allele results later on
        # for individual allele results, the allele counts should be recomputed
        # prediction_df = recompute_allele_counts(prediction_df)

        max_count = len(prediction_df['Epitope Description'].unique())

        # detection of MHC allele shortcuts with default alleles while ignoring source shortcuts
        prediction_df_allele_merged = merge_mhc_alleles(prediction_df, ['MHC Allele Prediction'])

        # dataset without default alleles
        default_alleles = ['eval default allele I', 'eval default allele II']

        # Peptide source and MHC allele shortcut detection

        # detection of both peptide source and MHC allele shortcuts with default alleles and default peptide sources
        # first group by MHC alleles and merge single-labeled subsets
        # this ensures that there are no unnecessary merges of peptide sources afterwards
        prediction_df_both_merged = merge_pep_sources(prediction_df_allele_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])
        prediction_df_both_merged = merge_mhc_alleles(prediction_df_both_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])

        # dataset without default alleles and without default peptide sources
        default_selection = prediction_df_both_merged['MHC Allele Prediction'].isin(default_alleles) | \
                            (prediction_df_both_merged['Epitope Parent Species'] == 'eval default peptide source')
        prediction_df_both_merged_wo_default = prediction_df_both_merged.loc[~default_selection]

        pos = len(prediction_df_both_merged_wo_default.loc[prediction_df_both_merged_wo_default['Assay Qualitative Measure'] == 'Positive'])
        neg = len(prediction_df_both_merged_wo_default.loc[prediction_df_both_merged_wo_default['Assay Qualitative Measure'] == 'Negative'])

        if pos > 0 and neg > 0:
            # keep allele counts from subset
            # prediction_df_both_merged_wo_default = recompute_allele_counts(prediction_df_both_merged_wo_default)

            # detection of both peptide source and MHC allele shortcuts without default alleles + default peptide sources
            allele_pep_source_grouped_results_wo_default = grouped_evaluation(prediction_df_both_merged_wo_default, ['MHC Allele Prediction', 'Epitope Parent Species'])
            allele_pep_source_results_wo_default = combine_group_results(allele_pep_source_grouped_results_wo_default)
            allele_pep_source_results_wo_default['skipped_count'] = max_count - allele_pep_source_results_wo_default['count']
            source_result_dict[f'{mhc_allele}/{description}'] = allele_pep_source_results_wo_default

    return source_result_dict

def get_evaluation_dict(prediction_full_df, description='Validation', eval_human=True):
    result_dict = {}
    prediction_I_df = prediction_full_df.loc[(prediction_full_df['MHC Class'] == 'I')]
    prediction_II_df = prediction_full_df.loc[(prediction_full_df['MHC Class'] == 'II')]
    class_I_pos = len(prediction_I_df.loc[prediction_I_df['Assay Qualitative Measure'] == 'Positive'])
    class_I_neg = len(prediction_I_df.loc[prediction_I_df['Assay Qualitative Measure'] == 'Negative'])
    class_II_pos = len(prediction_II_df.loc[prediction_II_df['Assay Qualitative Measure'] == 'Positive'])
    class_II_neg = len(prediction_II_df.loc[prediction_II_df['Assay Qualitative Measure'] == 'Negative'])

    eval_list = [('MHC I+II', prediction_full_df)]
    if class_I_pos > 0 and class_I_neg > 0:
        eval_list.append(('MHC I', prediction_I_df))
    if class_II_pos > 0 and class_II_neg > 0:
        eval_list.append(('MHC II', prediction_II_df))

    for MHC_text, prediction_df in eval_list:
        # evaluation with default values and without shortcut detection
        prediction_df = recompute_allele_counts(prediction_df)
        overall_results = get_performance_metrics(prediction_df)
        max_count = overall_results['count']

        # MHC allele shortcut detection

        # detection of MHC allele shortcuts with default alleles while ignoring source shortcuts
        prediction_df_allele_merged = merge_mhc_alleles(prediction_df, ['MHC Allele Prediction'])

        # dataset without default alleles
        default_alleles = ['eval default allele I', 'eval default allele II']
        allele_default_selection = prediction_df_allele_merged['MHC Allele Prediction'].isin(default_alleles)
        prediction_df_allele_merged_wo_default = recompute_allele_counts(prediction_df_allele_merged.loc[~allele_default_selection])

        # evaluation without default alleles and without shortcut detection
        uncorrected_results_wo_default_alleles = get_performance_metrics(prediction_df_allele_merged_wo_default)
        uncorrected_results_wo_default_alleles['skipped_count'] = max_count - uncorrected_results_wo_default_alleles['count']

        # detection of MHC allele shortcuts without default alleles
        allele_grouped_results_wo_default = grouped_evaluation(prediction_df_allele_merged_wo_default, ['MHC Allele Prediction'])
        allele_results_wo_default = combine_group_results(allele_grouped_results_wo_default)
        allele_results_wo_default['skipped_count'] = max_count - allele_results_wo_default['count']

        # Peptide source and MHC allele shortcut detection

        # detection of both peptide source and MHC allele shortcuts with default alleles and default peptide sources
        # first group by MHC alleles and merge single-labeled subsets
        # this ensures that there are no unnecessary merges of peptide sources afterwards
        prediction_df_both_merged = merge_pep_sources(prediction_df_allele_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])
        prediction_df_both_merged = merge_mhc_alleles(prediction_df_both_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])
        allele_pep_source_grouped_results = grouped_evaluation(prediction_df_both_merged, ['MHC Allele Prediction', 'Epitope Parent Species'])
        allele_pep_source_results = combine_group_results(allele_pep_source_grouped_results)

        # dataset without default alleles and without default peptide sources
        default_selection = prediction_df_both_merged['MHC Allele Prediction'].isin(default_alleles) |\
                            (prediction_df_both_merged['Epitope Parent Species'] == 'eval default peptide source')
        prediction_df_both_merged_wo_default = recompute_allele_counts(prediction_df_both_merged.loc[~default_selection])

        # evaluation without default alleles + peptide sources and without shortcut detection
        uncorrected_results_wo_default = get_performance_metrics(prediction_df_both_merged_wo_default)
        uncorrected_results_wo_default['skipped_count'] = max_count - uncorrected_results_wo_default['count']

        # detection of both peptide source and MHC allele shortcuts without default alleles + default peptide sources
        allele_pep_source_grouped_results_wo_default = grouped_evaluation(prediction_df_both_merged_wo_default, ['MHC Allele Prediction', 'Epitope Parent Species'])
        allele_pep_source_results_wo_default = combine_group_results(allele_pep_source_grouped_results_wo_default)
        allele_pep_source_results_wo_default['skipped_count'] = max_count - allele_pep_source_results_wo_default['count']

        # Peptide source shortcut detection

        # peptide source shortcut detection with default peptide sources
        prediction_df_pep_source_merged = merge_pep_sources(prediction_df, ['Epitope Parent Species'])

        # correcting only source shortcuts while ignoring MHC allele shortcuts
        # dataset without default peptide sources
        pep_source_default_selection = prediction_df_pep_source_merged['Epitope Parent Species'] == 'eval default peptide source'
        prediction_df_pep_source_merged_wo_default = prediction_df_pep_source_merged.loc[~pep_source_default_selection]

        # no peptide source shortcut detection without default peptide sources
        uncorrected_results_wo_default_pep_source = get_performance_metrics(prediction_df_pep_source_merged_wo_default)
        uncorrected_results_wo_default_pep_source['skipped_count'] = max_count - uncorrected_results_wo_default_pep_source['count']

        # peptide source shortcut detection without default peptide sources
        pep_source_grouped_results_wo_default = grouped_evaluation(prediction_df_pep_source_merged_wo_default, ['Epitope Parent Species'])
        pep_source_results_wo_default = combine_group_results(pep_source_grouped_results_wo_default)
        pep_source_results_wo_default['skipped_count'] = max_count - pep_source_results_wo_default['count']


        # Human peptides

        prediction_df_human = prediction_df.loc[prediction_df['Epitope Parent Species'] == 'Homo sapiens']
        if len(prediction_df_human) > 0 and eval_human:
            max_count_human = len(prediction_df_human['Epitope Description'].unique())

            prediction_df_human_recomputed = recompute_allele_counts(prediction_df_human)
            human_pep_overall_results = get_performance_metrics(prediction_df_human_recomputed)
            human_pep_overall_results['skipped_count'] = max_count_human - human_pep_overall_results['count']

            prediction_df_human_allele_merged = merge_mhc_alleles(prediction_df_human, ['MHC Allele Prediction'])

            human_pep_source_allele_grouped_results = grouped_evaluation(prediction_df_human_allele_merged, ['MHC Allele Prediction'])
            human_pep_source_allele_results = combine_group_results(human_pep_source_allele_grouped_results)
            human_pep_source_allele_results['skipped_count'] = max_count_human - human_pep_source_allele_results['count']

            allele_default_selection = prediction_df_human_allele_merged['MHC Allele Prediction'].isin(default_alleles)
            prediction_df_human_allele_merged_wo_default = recompute_allele_counts(prediction_df_human_allele_merged.loc[~allele_default_selection])

            human_pep_overall_results_wo_default = get_performance_metrics(prediction_df_human_allele_merged_wo_default)
            human_pep_overall_results_wo_default['skipped_count'] = max_count_human - human_pep_overall_results_wo_default['count']

            human_pep_source_allele_grouped_results_wo_default = grouped_evaluation(prediction_df_human_allele_merged_wo_default, ['MHC Allele Prediction'])
            human_pep_source_allele_results_wo_default = combine_group_results(human_pep_source_allele_grouped_results_wo_default)
            human_pep_source_allele_results_wo_default['skipped_count'] = max_count_human - human_pep_source_allele_results_wo_default['count']

            human_result_dict = {
                f'{MHC_text} - Human/{description}': human_pep_overall_results,
                f'{MHC_text} - Human - no defaults/{description}': human_pep_overall_results_wo_default,
                f'{MHC_text} - MHC corrected - Human/{description}': human_pep_source_allele_results,
                f'{MHC_text} - MHC corrected - Human - no defaults/{description}': human_pep_source_allele_results_wo_default
            }
        else:
            human_result_dict = {}

        if MHC_text == 'MHC I' or MHC_text == 'MHC II':
            prediction_df_human = prediction_df.loc[prediction_df['Epitope Parent Species'] == 'Homo sapiens']
            human_peptides = list(prediction_df_human['Epitope Description'].unique())
            if MHC_text == 'MHC I':
                peptide_subset = [peptide for peptide in human_peptides if len(peptide) < 15]
            else:
                peptide_subset = [peptide for peptide in human_peptides if len(peptide) >= 15]
            prediction_df_human_subset = prediction_df_human.loc[prediction_df_human['Epitope Description'].isin(peptide_subset)]

            if len(prediction_df_human_subset) > 0 and eval_human:
                max_count_human = len(prediction_df_human_subset['Epitope Description'].unique())

                prediction_df_human_subset_recomputed = recompute_allele_counts(prediction_df_human_subset)
                human_pep_overall_results = get_performance_metrics(prediction_df_human_subset_recomputed)
                human_pep_overall_results['skipped_count'] = max_count_human - human_pep_overall_results['count']

                prediction_df_human_allele_merged = merge_mhc_alleles(prediction_df_human_subset, ['MHC Allele Prediction'])

                human_pep_source_allele_grouped_results = grouped_evaluation(prediction_df_human_allele_merged, ['MHC Allele Prediction'])
                human_pep_source_allele_results = combine_group_results(human_pep_source_allele_grouped_results)
                human_pep_source_allele_results['skipped_count'] = max_count_human - human_pep_source_allele_results['count']

                allele_default_selection = prediction_df_human_allele_merged['MHC Allele Prediction'].isin(default_alleles)
                prediction_df_human_allele_merged_wo_default = recompute_allele_counts(prediction_df_human_allele_merged.loc[~allele_default_selection])
                human_pep_overall_results_wo_default = get_performance_metrics(prediction_df_human_allele_merged_wo_default)
                human_pep_overall_results_wo_default['skipped_count'] = max_count_human - human_pep_overall_results_wo_default['count']

                human_pep_source_allele_grouped_results_wo_default = grouped_evaluation(prediction_df_human_allele_merged_wo_default, ['MHC Allele Prediction'])
                human_pep_source_allele_results_wo_default = combine_group_results(human_pep_source_allele_grouped_results_wo_default)
                human_pep_source_allele_results_wo_default['skipped_count'] = max_count_human - human_pep_source_allele_results_wo_default['count']

                human_subset_result_dict = {
                    f'{MHC_text} - Human subset/{description}': human_pep_overall_results,
                    f'{MHC_text} - Human subset - no defaults/{description}': human_pep_overall_results_wo_default,
                    f'{MHC_text} - MHC corrected - Human subset/{description}': human_pep_source_allele_results,
                    f'{MHC_text} - MHC corrected - Human subset - no defaults/{description}': human_pep_source_allele_results_wo_default
                }
            else:
                human_subset_result_dict = {}
        else:
            human_subset_result_dict = {}

        result_dict = {**result_dict, **human_result_dict, **human_subset_result_dict,
                       f'{MHC_text}/{description}': overall_results,
                       f'{MHC_text} - no source defaults/{description}': uncorrected_results_wo_default_pep_source,
                       f'{MHC_text} - no MHC defaults/{description}': uncorrected_results_wo_default_alleles,
                       f'{MHC_text} - no MHC+source defaults/{description}': uncorrected_results_wo_default,

                       # f'{MHC_text} - MHC corrected/{description}': allele_results,
                       f'{MHC_text} - MHC corrected - no defaults/{description}': allele_results_wo_default,

                       f'{MHC_text} - source corrected - no defaults/{description}': pep_source_results_wo_default,

                       f'{MHC_text} - MHC+source corrected/{description}': allele_pep_source_results,
                       f'{MHC_text} - MHC+source corrected - no defaults/{description}': allele_pep_source_results_wo_default,
                    }
    return result_dict


def get_custom_evaluation_dict(prediction_full_df, description, eval_options):
    prediction_I_df = prediction_full_df.loc[(prediction_full_df['MHC Class'] == 'I')]
    prediction_II_df = prediction_full_df.loc[(prediction_full_df['MHC Class'] == 'II')]
    result_dict = {}
    for MHC_text, prediction_df in [('MHC I+II', prediction_full_df), ('MHC I', prediction_I_df), ('MHC II', prediction_II_df)]:
        grouped_results = grouped_evaluation(prediction_df, eval_options['group_by_columns'])
        grouped_results = select_evaluation_groups(grouped_results,
                                                   selected_allele=eval_options['selected_allele'],
                                                   selected_peptide_source=eval_options['selected_peptide_source'])
        allele_pep_source_results = combine_group_results(grouped_results)
        result_dict[f'{MHC_text} - custom/{description}'] = allele_pep_source_results
    return result_dict

