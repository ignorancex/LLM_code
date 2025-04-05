"""
Helper functions for use with our topic heatmap visualization Jupyter notebook

Authors: Lexie Li, George H. Chen
"""
import matplotlib
import numpy as np
import os
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sb
import warnings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('seaborn')
warnings.filterwarnings('ignore')


class MidpointNormalize(colors.Normalize):
    # https://matplotlib.org/3.1.0/gallery/userdemo/colormap_normalizations_custom.html
    # For adjusting mid values of the heatmaps
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def get_shorter_name(vocab, dataset):
    '''
        Get better feature names for plotting the heatmaps.
    '''
    new_vocab = []
    if dataset == "METABRIC":
        for v in vocab:
            if "NOT_IN_OSLOVAL_" in v:
                new_v = v.replace("NOT_IN_OSLOVAL_", "")
            else:
                new_v = v

            new_vocab.append(new_v)

    elif dataset.startswith("SUPPORT"):
        vmap = {"num.co": "num.comorbidities",
                "ca_metastatic": "cancer_metastatic",
                "ca_no": "cancer_no",
                "ca_yes": "cancer_yes",
                "wblc": "wbc_count",
                "crea": "serum_creatinine",
                "sod": "serum_sodium",
                "hrt": "heart_rate",
                "resp": "respiration_rate",
                "temp": "temperature_celcius",
                "meanbp": "mean_blood_pressure"}
        for v in vocab:
            for vpre in vmap.keys():
                if v.startswith(vpre):
                    new_v = v.replace(vpre, vmap[vpre])
                    break
                else:
                    new_v = v
            new_vocab.append(new_v)

    elif dataset == "UNOS":
        vmap = {"INIT_AGE": "AGE",
                "AGE_DON": "AGE_DONOR",
                "PRAMR": "MOST_RECENT_PRA",
                "CREAT_TRR": "CREATININE",
                "HGT_CM_CALC": "HEIGHT_CM",
                "HGT_CM_DON_CALC": "HEIGHT_CM_DONOR",
                "PREV_TX": "PREVIOUS_TRANSPLANT",
                "BMI_DON_CALC": "BMI_DONOR",
                "DIAL_PRIOR_TX": "DIALYSIS_HISTORY",
                "PRAPK": "PEAK_PRA",
                "LV_EJECT": "LV_EJECT_FRACTION",
                "INFECT_IV_DRUG_TRR": "INFECTION_REQUIRING_IV_DRUG",
                "AMIS": "A_LOCUS_MISMATCH_LEVEL",
                "WGT_KG_CALC": "WEIGHT_KG",
                "WGT_KG_DON_CALC": "WEIGHT_KG_DONOR",
                "TBILI": "DONOR_TERMINAL_TOTAL_BILIRUBIN",
                "BMIS": "B_LOCUS_MISMATCH_LEVEL",
                "DIAB": "DIABETES",
                "BMI_CALC": "BMI",
                "ABO": "BLOOD_GROUP",
                "DAYS_STAT1A": "DAYS_IN_STATUS_1A",
                "DAYS_STAT1": "DAYS_IN_STATUS_1",
                "IABP_TRR": "IABP",
                "ECMO_TRR": "ECMO_TRR",
                "HIST_DIABETES_DON": "DONOR_DIABETES_HISTORY",
                "DAYS_STAT2": "DAYS_IN_STATUS_2",
                "HEP_C_ANTIBODY_DON": "HEP_C_ANTIBODY_DONOR",
                "DRMIS": "DR_LOCUS_MISMATCH_LEVEL",
                "VAD_TAH_TRR": "VAD_TAH",
                "DAYS_STAT1B": "DAYS_IN_STATUS_1B",
                "ABO_DON": "BLOOD_GROUP_DONOR",
                "CREAT_DON": "CREATININE_DONOR",
                "HLAMIS": "HLA_MISMATCH_LEVEL",
                "CLIN_INFECT_DON": "DONOR_CLINICAL_INFECTION",
                "ISCHTIME": "ISCHEMIC_TIME_HOURS",
                "ABO_MAT": "ABO_MATCH_LEVEL",
                "VENTILATOR_TRR": "VENTILATOR",
                "GENDER_DON": "GENDER_DONOR"}
        for v in vocab:
            for vpre in vmap.keys():
                if v.startswith(vpre):
                    new_v = v.replace(vpre, vmap[vpre])
                    break
                else:
                    new_v = v
            new_v = new_v.replace("_TRR", "")
            new_v = new_v.replace("_DON_", "_DONOR_")
            new_v = new_v.replace("_DON(", "_DONOR(")
            new_v = new_v.replace("_MAT_", "_MATCH_LEVEL_")
            new_v = new_v.replace("_MAT(", "_MATCH_LEVEL(")

            new_vocab.append(new_v)

    elif dataset == "Ich":
        for v in vocab:
            if "merged_others" in v:
                start_i = v.index("merged_others")
                end_i = start_i + len("merged_others")
                new_vocab.append(v[:end_i])
            else:
                new_v = v.replace('_date:', ':')
                new_vocab.append(new_v)

    return np.array(new_vocab)


def argsort_vocab(vocabulary, topic_distributions, score_func, aggregate_func,
                  reverse=False, max_words=None, verbose=False):
    score_dict = dict()
    for v_i, v in enumerate(vocabulary):
        if "BIN#" in v:
            try:
                base_feature = re.search(r'(\w+)\(BIN#\d\):', v).group(1)
            except BaseException:
                base_feature = v[:(v.index('BIN#')-1)]
        elif "_" in v:
            base_feature = "_".join(v.split("_")[:-1])
        else:
            base_feature = v

        score = score_func(topic_distributions[v_i])

        if base_feature in score_dict:
            score_dict[base_feature] = aggregate_func(score_dict[base_feature],
                                                      score)
        else:
            score_dict[base_feature] = score

    sorted_key_val = sorted(score_dict.items(), key=lambda pair: pair[1],
                            reverse=reverse)

    new_order = []
    if verbose:
        print('Top scores (across row):')
    for curr_base, score in sorted_key_val:
        if verbose:
            print(curr_base, ':', score)
        block_to_add = []
        for v_i, v in enumerate(vocabulary):
            if "BIN#" in v:
                try:
                    base_feature = re.search(r'(\w+)\(BIN#\d\):', v).group(1)
                except BaseException:
                    base_feature = v[:(v.index('BIN#')-1)]
            elif "_" in v:
                base_feature = "_".join(v.split("_")[:-1])
            else:
                base_feature = v

            if curr_base == base_feature:
                block_to_add.append(v_i)
        if max_words is not None \
                and len(new_order) + len(block_to_add) > max_words:
            print()
            print(len(block_to_add))
            print([vocabulary[_] for _ in block_to_add])
            print()
            break
        new_order += block_to_add
    new_order = np.array(new_order)
    return new_order


cdict = {'red':   ((0.0,  1.0, 1.0),
                   (0.9,  0.9, 0.9),
                   (1.0,  0.6, 0.6)),
         'green': ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),
         'blue':  ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0))}
modified_red_color_map = \
    colors.LinearSegmentedColormap('modified_red_color_map',
                                   segmentdata=cdict, N=256)


def heatmap_plot_topic_reordered(
        model, dataset, topic_distributions, beta, vocabulary,
        sort_by_beta=True, sort_by_feature_name=True, logscale=False,
        saveto="", show_plot=False, max_words=80,
        min_df=0.02, max_df=0.5, remove_chart=True,
        word_weighting='standard', word_ranking='average',
        use_log_deviations=False, use_AFT=False, use_background_topic=True):
    if use_background_topic:
        # actual beta has a 0 at the end when using a background topic!
        beta = np.array(list(beta) + [0.])

    if sort_by_beta:
        if not use_AFT:
            topic_order = np.argsort(-beta)
        else:
            topic_order = np.argsort(beta)
        topic_distributions = topic_distributions[topic_order]
        beta = beta[topic_order]
    print('Beta', ['%f' % b for b in beta])

    # remove stop words
    df = np.loadtxt(os.path.join('dataset',
                                 'doc_freq_%s.txt' % dataset))
    df_filter = (df < min_df) | (df > max_df)
    if np.any(df_filter):
        topic_distributions = topic_distributions[:, ~df_filter]
        vocabulary = vocabulary[~df_filter]
        df = df[~df_filter]
    if remove_chart:
        chart_mask = np.array([word.lower().startswith('chart:')
                               for word in vocabulary],
                              dtype=bool)
        if len(chart_mask) > 0 and np.any(chart_mask):
            topic_distributions = topic_distributions[:, ~chart_mask]
            vocabulary = vocabulary[~chart_mask]
            df = df[~chart_mask]
    topic_distributions /= topic_distributions.sum(axis=1)[:, np.newaxis]

    if word_weighting == 'tfidf':
        log_vocab_probs = np.log(topic_distributions)
        scores = topic_distributions \
            * (log_vocab_probs
               - np.mean(log_vocab_probs, axis=0)[np.newaxis, :])
    elif word_weighting == 'idf':
        scores = topic_distributions * np.log(1 / df)
    else:
        scores = topic_distributions

    nan_mask = np.isnan(scores)
    if np.any(nan_mask):
        if not use_log_deviations:
            scores[nan_mask] = np.nanmin(scores)
        else:
            scores[nan_mask] = 0

    scores = scores.transpose()

    if sort_by_feature_name:
        con_vocab_ids = []
        cat_vocab_ids = []
        for v_i, v in enumerate(vocabulary):
            if "BIN#" in v:
                con_vocab_ids.append(v_i)
            else:
                cat_vocab_ids.append(v_i)

        cat_sorted_args = np.argsort(vocabulary[cat_vocab_ids])
        con_sorted_args = np.argsort(vocabulary[con_vocab_ids])

        cat_scores = scores[cat_vocab_ids][cat_sorted_args]
        con_scores = scores[con_vocab_ids][con_sorted_args]
        scores = np.vstack((cat_scores, con_scores))

        vocabulary = np.concatenate(
            (vocabulary[cat_vocab_ids][cat_sorted_args],
             vocabulary[con_vocab_ids][con_sorted_args]))

    # sort by probabilities
    if word_ranking == 'max':
        new_order = argsort_vocab(vocabulary, scores, np.max,
                                  max, reverse=True, max_words=max_words)
        print(len(new_order))
        vocabulary = vocabulary[new_order]
        scores = scores[new_order]

    if word_ranking == 'range':
        # still sort by max probability first (and then we do another sort)
        new_order = argsort_vocab(vocabulary, scores, np.max,
                                  max, reverse=True)
        vocabulary = vocabulary[new_order]
        scores = scores[new_order]

        # sort by peak-to-peak
        new_order = argsort_vocab(vocabulary, scores, np.ptp,
                                  max, reverse=True, max_words=max_words)
        vocabulary = vocabulary[new_order]
        scores = scores[new_order]

    elif word_ranking == 'average':
        # still sort by max probability first (and then we do another sort)
        new_order = argsort_vocab(vocabulary, scores, np.max,
                                  max, reverse=True)
        vocabulary = vocabulary[new_order]
        scores = scores[new_order]

        new_order = argsort_vocab(vocabulary, scores, np.mean,
                                  max, reverse=True, max_words=max_words)
        vocabulary = vocabulary[new_order]
        scores = scores[new_order]

    vocabulary = get_shorter_name(vocabulary, dataset)

    with plt.style.context('seaborn-dark'):
        fig = plt.figure(figsize=(len(beta)*0.7, len(vocabulary)/3*0.7))
        ax = fig.gca()
        ax.xaxis.tick_top()
        ax.set_xlabel('X LABEL')
        ax.xaxis.set_label_position('top')
        if not logscale:
            if not use_log_deviations:
                sb.heatmap(scores, cmap=modified_red_color_map)
            else:
                sb.heatmap(scores,
                           cmap='RdBu_r',
                           norm=MidpointNormalize(
                               midpoint=0,
                               vmin=-np.max(np.abs(topic_distributions)),
                               vmax=np.max(np.abs(topic_distributions))))
        else:
            sb.heatmap(scores, cmap="Reds",
                       norm=colors.LogNorm(vmin=scores.min(),
                                           vmax=scores.max()))
        plt.ylabel('Feature')
        if not use_AFT:
            plt.xlabel('Cox regression coefficient')
        else:
            plt.xlabel('AFT regression coefficient')
        plt.yticks(np.arange(len(vocabulary)) + 0.5,
                   [_v.lower() for _v in vocabulary],
                   rotation='horizontal')
        plt.xticks(np.arange(len(beta)) + 0.5, [" %.2f " % _b for _b in beta])
        plt.savefig(saveto+"_heatmap.pdf", bbox_inches='tight')
        if not show_plot:
            plt.close()


def print_top_words(probs, betas, features, transcript_path, n_top=50):

    with open(transcript_path, "w") as transcript:

        topic_order = np.argsort(-betas)

        for topic_i in range(len(betas)):
            print("Topic #{} (Beta={})".format(
                topic_i, betas[topic_order[topic_i]]), file=transcript)

            curr_probs = probs[topic_order[topic_i]]
            curr_order = np.argsort(-curr_probs)
            for word_i, word_id in enumerate(curr_order[:100]):
                print(features[word_id],
                      curr_probs[word_id],
                      np.exp(curr_probs[word_id]), file=transcript)

            print("---\n", file=transcript)


def plot_topic_heatmaps(dataset, model, experiment_id,
                        root_dir=None, vis_output_dir="plots",
                        word_ranking='max',
                        word_weighting='standard',
                        action_list=None,
                        use_distributions=True,
                        use_AFT=False):
    '''
        Action list specify plots that the user would like to generate. 
        If None, default to all topic heatmap.
    '''
    if root_dir is None:
        root_dir = os.path.join("dataset", dataset, model, experiment_id)

    outer_iter_i = 0
    saved_outputs_fname = \
        "{}_{}_{}_{}_beta.pickle".format(model, dataset, experiment_id,
                                         outer_iter_i)

    with open(os.path.join(root_dir, saved_outputs_fname), "rb") as file:
        beta_info = pickle.load(file)

    if use_distributions: # use topic_deviations for survscholar
        vocab_probs = beta_info['topic_distributions']
    else:
        vocab_probs = beta_info['topic_deviations']

    n_topics = vocab_probs.shape[0]
    n_vocab = vocab_probs.shape[1]
    print("Topic output loaded: {} topics, {} words".format(n_topics, n_vocab))

    heatmap_plot_topic_reordered(
        model, dataset, vocab_probs, beta_info['beta'], beta_info['vocabulary'],
        saveto=os.path.join(vis_output_dir, "{}_{}_{}_{}_{}_{}".format(
            model, dataset, experiment_id, outer_iter_i,
            word_weighting, word_ranking)),
        logscale=False,
        show_plot=False,
        word_ranking=word_ranking, word_weighting=word_weighting,
        use_log_deviations=not use_distributions,
        use_AFT=use_AFT)
