'''
Preprocessing script for SUPPORT datasets to create experiment-ready .npy files

Authors: Lexie Li, George H. Chen
'''
import os
import numpy as np
import pandas as pd


def process_support(mode):
    '''
    mode could be one of {'discretized', 'original', 'cox'}
    '''
    support_dzclasses = ['ARF_MOSF', 'COPD_CHF_Cirrhosis', 'Cancer', 'Coma']
    for dzclass in support_dzclasses:
        print('Preprocessing SUPPORT_{} dataset, mode = {}'.format(dzclass, mode))
        os.makedirs('SUPPORT_{}'.format(dzclass), exist_ok=True)

        data_df = pd.read_csv('support_{}.csv'.format(dzclass), delimiter=',')
        print('Dimensions of incoming {} dataset:'.format(dzclass),
              data_df.shape)
        data_df.dropna(inplace=True)
        assert(np.sum(data_df.isna().values) == 0)

        feature_vectors_df = data_df.drop(columns=['d.time', 'death'])
        labels_df = data_df[['d.time', 'death']]

        cat_features = []
        universal_features = []
        cat_features_prefixes = dict()
        for cat_feature in ['sex', 'race', 'ca']:
            if len(np.unique(feature_vectors_df.loc[:, cat_feature])) == 1:
                # this handles a special case: feature 'ca' takes the same
                # value for all subjects for the support_cancer dataset
                universal_features.append(cat_feature)
            else:
                cat_features.append(cat_feature)
                cat_features_prefixes[cat_feature] = cat_feature

        feature_vectors_df = feature_vectors_df.drop(
            columns=universal_features)

        # one-hot encode categorical variables:
        # the difference between mode 'original' and mode 'cox' is that:
        # mode cox drops one reference column for each categorical variable
        # during one-hot encoding; this is required for unregularized Cox
        # models to avoid numerical issues during optimization
        if mode == 'original':
            feature_vectors_df = pd.get_dummies(
                feature_vectors_df,
                columns=cat_features,
                prefix=cat_features_prefixes,
                drop_first=False)

        elif mode == 'cox':
            feature_vectors_df = pd.get_dummies(
                feature_vectors_df,
                columns=cat_features,
                prefix=cat_features_prefixes,
                drop_first=True)

        elif mode == 'discretized':
            # we also do not remove reference columns for mode 'discretized'
            feature_vectors_df = pd.get_dummies(
                feature_vectors_df,
                columns=cat_features,
                prefix=cat_features_prefixes,
                drop_first=False)

            n_sample, n_feature = feature_vectors_df.shape

            continuous_features = [
                'age',
                'meanbp',
                'hrt',
                'resp',
                'temp',
                'wblc',
                'sod',
                'crea',
                'num.co']
            continuous_features_quantiles = np.linspace(0, 1, 6)

            for curr_feature in continuous_features:
                curr_vals = feature_vectors_df[curr_feature]
                curr_quantile_edges = list(np.quantile(
                    curr_vals[curr_vals.notnull()],
                    continuous_features_quantiles))
                discretized_vals = np.digitize(
                    curr_vals, bins=curr_quantile_edges[:-1])

                discretized_bin_label = 1
                for discretized_id in np.unique(discretized_vals):
                    curr_lo = curr_quantile_edges[discretized_id - 1]
                    curr_hi = curr_quantile_edges[discretized_id]
                    new_feature_name = curr_feature + '(BIN#{}):{}-{}'.format(
                        discretized_bin_label, np.round(
                            curr_lo, decimals=2), np.round(
                            curr_hi, decimals=2))

                    new_feature_vals = (
                        discretized_vals == discretized_id).astype(
                        np.int32)
                    feature_vectors_df[new_feature_name] = new_feature_vals
                    discretized_bin_label += 1

            feature_vectors_df = feature_vectors_df.drop(
                columns=continuous_features)
            # note here that we don't have to remove the missingness flags
            # because we removed all samples with missing info for the SUPPORT
            # dataset; we decided to do this because SUPPORT datasets have very
            # few samples with missing values

        feature_vectors_df = feature_vectors_df.astype('float64')
        labels_df = labels_df.astype('float64')
        assert(np.sum(feature_vectors_df.isna().values) == 0)
        assert(np.sum(labels_df.isna().values) == 0)

        # there is no full or empty column
        try:
            assert(
                np.sum(
                    np.sum(
                        feature_vectors_df.values,
                        axis=0) /
                    feature_vectors_df.shape[0] == 1) == 0)
            assert(
                np.sum(
                    np.sum(
                        feature_vectors_df.values,
                        axis=0) /
                    feature_vectors_df.shape[0] == 0) == 0)
        except BaseException:
            full_col_ids = np.where(
                np.sum(
                    feature_vectors_df.values,
                    axis=0) /
                feature_vectors_df.shape[0] == 1)
            print('this column is always one for all samples:',
                  feature_vectors_df.columns[full_col_ids])
            assert(False)

        if mode == 'discretized':
            print('Discretized dataset dimensions: ', feature_vectors_df.shape)
            # These will be loaded for experiments
            np.save(
                'SUPPORT_{}/X_discretized.npy'.format(dzclass),
                feature_vectors_df.values)
            np.save(
                'SUPPORT_{}/Y_discretized.npy'.format(dzclass),
                labels_df.values)
            np.savetxt(
                'SUPPORT_{}/F_discretized.txt'.format(dzclass),
                feature_vectors_df.columns,
                fmt='%s')

        elif mode == 'original':
            print(
                'Non-discretized dataset dimensions: ',
                feature_vectors_df.shape)

            np.save(
                'SUPPORT_{}/X.npy'.format(dzclass),
                feature_vectors_df.values)
            np.save('SUPPORT_{}/Y.npy'.format(dzclass), labels_df.values)
            np.savetxt(
                'SUPPORT_{}/F.txt'.format(dzclass),
                feature_vectors_df.columns,
                fmt='%s')

        elif mode == 'cox':
            print('Cox dataset dimensions: ', feature_vectors_df.shape)

            np.save(
                'SUPPORT_{}/X_cox.npy'.format(dzclass),
                feature_vectors_df.values)
            np.save('SUPPORT_{}/Y_cox.npy'.format(dzclass), labels_df.values)
            np.savetxt('SUPPORT_{}/F_cox.txt'.format(dzclass),
                       feature_vectors_df.columns, fmt='%s')
        else:
            raise NotImplementedError


if __name__ == '__main__':
    process_support(mode='original')
    process_support(mode='discretized')
    process_support(mode='cox')

    for dataset in [
            'SUPPORT_ARF_MOSF',
            'SUPPORT_COPD_CHF_Cirrhosis',
            'SUPPORT_Cancer',
            'SUPPORT_Coma']:
        X = np.load(os.path.join(dataset, 'X_discretized.npy'))
        df = (X >= 1).mean(axis=0)
        np.savetxt('doc_freq_{}.txt'.format(dataset), df)
