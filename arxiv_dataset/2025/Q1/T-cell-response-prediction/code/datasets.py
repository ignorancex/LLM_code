from ast import literal_eval
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from packages.mhcnames import compact_allele_name


class TcellData(object):
    def __init__(self, params):
        self.params = params

        train_data = pd.read_csv(params['train_file'], index_col=0, low_memory=False)
        eval_data = pd.read_csv(params['eval_file'], index_col=0, low_memory=False)

        if 'tsne_subset' in self.params and self.params['tsne_subset']:
            vis_sources = ['Human betaherpesvirus 6B', 'Vaccinia virus', 'Dengue virus',
                           'Human alphaherpesvirus 2', 'Homo sapiens',]  # 'Phleum pratense',
            vis_alleles_I = ['default allele I', 'HLA-C*07:02', 'HLA-A*02:01', 'HLA-A*11:01', 'HLA-B*44:02', 'HLA-B*07:02',
                             'HLA-A*29:02', 'HLA-A*24:07', 'HLA-B*08:01', 'HLA-A*33:01', 'default allele II',
                              'HLA-DRA1*01:01-DRB1*16:02', 'HLA-DRA1*01:01-DRB3*02:02', 'HLA-DPA1*01:03-DPB1*02:01',
                             'HLA-DRA1*01:01-DRB1*11:01', 'HLA-DRA1*01:01-DRB1*09:01', 'HLA-DRA1*01:01-DRB1*03:01',
                             'HLA-DRA1*01:01-DRB1*12:01', 'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DRA1*01:01-DRB5*01:01']
            train_data = train_data.loc[train_data['MHC Allele Prediction'].isin(vis_alleles_I) & train_data['Epitope Parent Species'].isin(vis_sources)]
            eval_data = eval_data.loc[eval_data['MHC Allele Prediction'].isin(vis_alleles_I) & eval_data['Epitope Parent Species'].isin(vis_sources)]

        class_I_II_peptides = {
            'AEEDEREISVPAEIL', 'AEQFKQKALGLLQTA', 'DVKFPGGGQIVGGVY', 'GTITVEELKKLLEQW', 'HVRLLSYRGDPLVFK', 'KQFLSASYEFQREF',
            'KRYFRPLLRAWSLGL', 'LPAPNYTFALWRVSA', 'MYENYIVPEDKREMW', 'QLMPDDYSNTHSTRY', 'RFFKAVNFREGK', 'RGKPGIYRFVAPGER',
            'RHTPVNSWLGNIIMF', 'TIFKIRMYVGGVEHR', 'TVATRDGKLPATQLR', 'VFTDNSSPPVVPQSF', 'VPYFVRVQGLLRFCA', 'VQAWKSKKTPMGFSY'}

        train_data = train_data.loc[~train_data['Epitope Description'].isin(class_I_II_peptides)]
        eval_data = eval_data.loc[~eval_data['Epitope Description'].isin(class_I_II_peptides)]

        if self.params['mhc_class'] != 'I+II':
            eval_data = eval_data.loc[eval_data['MHC Class'] == self.params['mhc_class']]

        if 'permute_labels' in params and params['permute_labels'] == 'full_init':
            eval_data = self.permute_labels(eval_data)
            train_data = self.permute_labels(train_data)

        train_data_full = train_data.copy()
        eval_data_full = eval_data.copy()

        if self.params['train_mhc_class'] == 'mhc_class':
            train_mhc_class = self.params['mhc_class']
        else:
            train_mhc_class = self.params['train_mhc_class']

        train_data_full = train_data_full if train_mhc_class == 'I+II' \
            else train_data_full.loc[train_data_full['MHC Class'] == train_mhc_class]

        if params['select_pep_sources'] != 'All':
            source_selection = [params['select_pep_sources'],]
            train_data = train_data.loc[train_data['Epitope Parent Species'].isin(source_selection)]
            eval_data = eval_data.loc[eval_data['Epitope Parent Species'].isin(source_selection)]

        if params['select_pep_sources'] == 'All' and params['pretrain_model']:
            source_selection = [params['pep_source_train_selection'],]
            eval_data = eval_data.loc[eval_data['Epitope Parent Species'].isin(source_selection)]

        if params['select_alleles'] != 'MHC Class I+II' and params['select_alleles'] != 'All':
            if params['select_alleles'] == 'MHC Class I' or params['select_alleles'] == 'MHC Class II':
                mhc_class = 'I' if params['select_alleles'] == 'MHC Class I' else 'II'
                train_data = train_data.loc[train_data['MHC Class'] == mhc_class]
                eval_data = eval_data.loc[eval_data['MHC Class'] == mhc_class]
            else:
                train_data = train_data.loc[train_data['MHC Allele Prediction'] == params['select_alleles']]
                eval_data = eval_data.loc[eval_data['MHC Allele Prediction'] == params['select_alleles']]

        if 'permute_labels' in params and params['permute_labels'] == 'full':
            eval_data = self.permute_labels(eval_data)
        elif 'permute_labels' in params and params['permute_labels'] == 'per_allele':
            eval_data = self.permute_labels_per_allele(eval_data)

        self.eval_data = eval_data.copy()
        self.eval_data_full = eval_data_full

        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.amino_acid_to_id = {self.amino_acids[i]: i for i in range(len(self.amino_acids))}
        self.amino_acid_to_names = {'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic acid', 'C': 'Cysteine',
                                    'Q': 'Glutamine', 'E': 'Glutamic acid', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
                                    'L': 'Leucine', 'K': 'Lysine', 'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline',
                                    'S': 'Serine', 'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'}
        self.amino_acid_names = [self.amino_acid_to_names[aa] for aa in self.amino_acids]
        # self.pseudoseq_dict = self.get_pseudoseq_dict(train_data)

        # only consider the top 20 peptide sources for adversarial debiasing
        pep_source_list = train_data_full['Epitope Parent Species'].value_counts().iloc[:15].index.to_list()
        pep_source_idx = list(range(len(pep_source_list) + 1))
        self.pep_source_dict = dict(zip(pep_source_list + ['default peptide source'], pep_source_idx))
        train_data['Peptide Source'] = train_data['Epitope Parent Species'].apply(
            lambda x: x if x in pep_source_list else 'default peptide source')
        train_data_full['Peptide Source'] = train_data_full['Epitope Parent Species'].apply(
            lambda x: x if x in pep_source_list else 'default peptide source')
        self.eval_data['Peptide Source'] = self.eval_data['Epitope Parent Species'].apply(
            lambda x: x if x in pep_source_list else 'default peptide source')
        self.eval_data_full['Peptide Source'] = self.eval_data_full['Epitope Parent Species'].apply(
            lambda x: x if x in pep_source_list else 'default peptide source')

        allele_list = sorted(list(train_data_full['MHC Allele Prediction'].unique()))
        allele_idx = list(range(len(allele_list)))
        self.allele_dict = dict(zip(allele_list, allele_idx))
        self.label_per_allele, self.label_per_source, self.allele_freq_pos, self.allele_freq_neg, self.pep_source_freq_pos, \
        self.pep_source_freq_neg = self.get_baseline_freq(train_data_full)

        def inverse_sigmoid(score):
            return torch.log(0.00001 + score/(1-score+0.00001))

        self.allele_freq_pos_unnormalized = inverse_sigmoid(self.allele_freq_pos)
        self.allele_freq_neg_unnormalized = inverse_sigmoid(self.allele_freq_neg)
        self.pep_source_freq_pos_unnormalized = inverse_sigmoid(self.pep_source_freq_pos)
        self.pep_source_freq_neg_unnormalized = inverse_sigmoid(self.pep_source_freq_neg)

        # token indices for Transformer models
        self.padding_token_idx = len(self.amino_acid_to_id)
        self.prediction_token_idx = len(self.amino_acid_to_id) + 1
        self.pep_start_token_idx = len(self.amino_acid_to_id) + 2
        self.pep_end_token_idx = len(self.amino_acid_to_id) + 2
        self.allele_padding_token_idx = len(self.allele_dict)

        self.train_data = train_data.drop_duplicates(subset=['Epitope Description'])

        dataset_dict = {'BagOfAA': BagOfAADataset,
                        'OneHotPeptide': PeptideDataset}
        TcellDataset = dataset_dict[params['features']]

        if self.params['pretrain_model']:
            if 'pretraining_data' in self.params and self.params['pretraining_data'] == 'selected_source':
                # pre-train on human peptides from both MHC classes
                base_pep_source = self.params['pep_source_train_selection']
            elif 'pretraining_data' in self.params and self.params['pretraining_data'] == 'wo_selected_source':
                # exclude human peptides from the other MHC class
                base_pep_source = f'non-human-II' if self.params['mhc_class'] == 'I' else 'non-human-I'
            else:
                base_pep_source = 'all'
        elif self.params['combine_sources']:
            base_pep_source = 'non-human'
        else:
            base_pep_source = self.params['pep_source_train_selection']

        # print('Base peptide source: ', base_pep_source)

        train_data_selection = self.train_data if train_mhc_class == 'I+II'\
            else self.train_data.loc[self.train_data['MHC Class'] == train_mhc_class]

        if 'permute_labels' in params and params['permute_labels'] == 'full':
            train_data_selection = self.permute_labels(train_data_selection)
        elif 'permute_labels' in params and params['permute_labels'] == 'per_allele':
            train_data_selection = self.permute_labels_per_allele(train_data_selection)

        if self.params['batch_size'] > len(train_data_selection)*0.5:
            self.params['batch_size'] = round(len(train_data_selection) * 0.5) - 1
            print(f'Set batch size to {round(len(train_data_selection) * 0.5) - 1} since there are only {len(train_data_selection)} training examples.')

        # print('train data set selection size: ', len(train_data_selection))
        # print(train_data_selection['Assay Qualitative Measure'].value_counts())
        # print('eval data set size: ', len(self.eval_data))
        # print(self.eval_data['Assay Qualitative Measure'].value_counts())

        train_dataset = TcellDataset(self, train_data_selection, pep_source=base_pep_source, sample=self.params['sample_size'])
        self.train_dataset_size = len(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True,
                                           collate_fn=train_dataset.collate_fn)
        # print('train dataset size', len(train_dataset))
        if self.params['combine_sources'] or self.params['pretrain_model']:
            finetune_pep_source = self.params['pep_source_train_selection'] if self.params['select_pep_sources'] == 'All' else 'all'
            train_dataset_human = TcellDataset(self, self.train_data, mhc_class=self.params['mhc_class'], pep_source=finetune_pep_source)
            self.train_dataloader_human = DataLoader(train_dataset_human, batch_size=self.params['batch_size_human'], shuffle=True, drop_last=True,
                                               collate_fn=train_dataset_human.collate_fn)

        eval_dataset = TcellDataset(self, self.eval_data)
        eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False,
                                     collate_fn=eval_dataset.collate_fn)
        self.eval_batch = next(iter(eval_dataloader))

        if self.params['evaluate_pep_sources'] or self.params['evaluate_alleles']:
            eval_dataset_full = TcellDataset(self, self.eval_data_full)
            eval_dataloader_full = DataLoader(eval_dataset_full, batch_size=len(eval_dataset_full), shuffle=False,
                                         collate_fn=eval_dataset_full.collate_fn)
            self.eval_batch_full = next(iter(eval_dataloader_full))

        eval_data_unique = self.eval_data.drop_duplicates(subset=['Epitope Description'])
        eval_dataset_unique = TcellDataset(self, eval_data_unique, mhc_class=self.params['mhc_class'])

        eval_dataloader_unique = DataLoader(eval_dataset_unique, batch_size=len(eval_dataset_unique), shuffle=False,
                                     collate_fn=eval_dataset_unique.collate_fn)
        self.eval_batch_unique = next(iter(eval_dataloader_unique))

        peptide_sample = train_data_selection['Epitope Description'].sample(frac=self.params['train_eval_frac'])
        self.train_eval_data = train_data_selection.loc[train_data_selection['Epitope Description'].isin(peptide_sample)]
        self.train_eval_data = self.train_eval_data.copy()
        train_eval_data_unique = self.train_eval_data.drop_duplicates(subset=['Epitope Description'])
        train_eval_dataset = TcellDataset(self, self.train_eval_data)
        train_eval_dataloader = DataLoader(train_eval_dataset, batch_size=len(self.train_eval_data), shuffle=False,
                                           collate_fn=train_eval_dataset.collate_fn)
        self.train_eval_batch = next(iter(train_eval_dataloader))
        train_eval_dataset_unique = TcellDataset(self, train_eval_data_unique, mhc_class=self.params['mhc_class'])
        train_eval_dataloader_unique = DataLoader(train_eval_dataset_unique, batch_size=len(train_eval_data_unique), shuffle=False,
                                           collate_fn=train_eval_dataset_unique.collate_fn)
        self.train_eval_batch_unique = next(iter(train_eval_dataloader_unique))

    def permute_labels(self, df):
        df_unique = df.drop_duplicates(subset=['Epitope Description'])
        num_labels = len(df_unique)
        pos_weight = (df_unique['Assay Qualitative Measure'] == 'Positive').sum()/len(df_unique)
        permuted_labels = random.choices(['Positive', 'Negative'], weights=[pos_weight, 1-pos_weight], k=num_labels)
        peptide_list = df_unique['Epitope Description'].to_list()
        permuted_labels_dict = dict(zip(peptide_list, permuted_labels))
        permuted_df = df.copy()
        permuted_df['Assay Qualitative Measure'] = df['Epitope Description'].apply(lambda peptide: permuted_labels_dict[peptide])
        return permuted_df

    def permute_labels_per_allele(self, df):
        # first assign new labels per allele
        grouped_df_list = []
        for mhc_allele, group_df in df.groupby(['MHC Allele Prediction']):
            permuted_group_df = self.permute_labels(group_df)
            grouped_df_list.append(permuted_group_df)
        permuted_df = pd.concat(grouped_df_list)

        # if one peptide has several alleles, the assigned labels need to be consistent
        # randomly reassign them if they are not consistent
        permuted_labels_dict = {}
        for peptide, group_df in permuted_df.groupby(['Epitope Description']):
            labels = group_df['Assay Qualitative Measure'].values

            if len(set(labels)) > 1:
                pos_fraction = np.sum(labels == 'Positive') / len(labels)
                new_label = random.choices(['Positive', 'Negative'], k=1, weights=[pos_fraction, 1-pos_fraction])[0]
                permuted_labels_dict[peptide] = new_label
            else:
                permuted_labels_dict[peptide] = labels[0]

        permuted_df['Assay Qualitative Measure'] = permuted_df['Epitope Description'].apply(lambda peptide: permuted_labels_dict[peptide])

        return permuted_df

    def get_pseudoseq_dict(self, train_data):
        pseudoseq_NetMHC_df = pd.read_csv(self.params['pseudoseq_file'], index_col=0, sep=' ', header=None)
        # there seem to be a couple of alleles with two sequences, keep the first one
        pseudoseq_NetMHC_df = pseudoseq_NetMHC_df[~pseudoseq_NetMHC_df.index.duplicated(keep='first')]
        pseudoseq_NetMHC_dict = pseudoseq_NetMHC_df.T.to_dict(orient='list')
        pseudoseq_dict = {allele: pseudoseq_NetMHC_dict[self.NetMHC_allele_names(allele)][0]
                          for allele in train_data['MHC Allele Name'].unique()}
        return pseudoseq_dict

    def normalize_embeddings(self, embeddings, normalization_data):
        embedding_mean = np.mean(normalization_data, axis=0)
        embedding_std = np.std(normalization_data, axis=0)
        embedding_std[embedding_std == 0] = 1
        embeddings_normalized = (embeddings - embedding_mean) / embedding_std
        return embeddings_normalized

    def NetMHC_allele_names(self, allele):
        allele = compact_allele_name(allele)\
            .replace('DRB1', 'DRB1_').replace('DRB3', 'DRB3_').replace('DRB4', 'DRB4_').replace('DRB5', 'DRB5_')
        if 'DRB' not in allele:
            allele = 'HLA-' + allele
        return allele

    def get_input_dimension(self):
        return self.train_eval_batch[self.params['feature_selector']].shape[1]

    def parse_allele_vectors(self, df):
        allele_id_lists = df['MHC Allele Prediction - List'].apply(
            lambda allele_list: [self.allele_dict[allele] for allele in literal_eval(allele_list)]).to_list()

        allele_vectors = []
        allele_vectors_ones = []
        for allele_ids in allele_id_lists:
            multi_hot_vector = np.zeros(len(self.allele_dict))
            multi_hot_vector[allele_ids] = 1./len(allele_ids)
            multi_hot_vector_ones = np.zeros(len(self.allele_dict))
            multi_hot_vector_ones[allele_ids] = 1
            allele_vectors.append(multi_hot_vector)
            allele_vectors_ones.append(multi_hot_vector_ones)

        return allele_id_lists, allele_vectors, allele_vectors_ones

    def get_baseline_freq(self, df):
        df_pos = df.loc[df['Assay Qualitative Measure'] == 'Positive']
        df_neg = df.loc[df['Assay Qualitative Measure'] == 'Negative']

        num_pep_pos = len(df_pos['Epitope Description'].unique())
        num_pep_neg = len(df_neg['Epitope Description'].unique())

        allele_count_pos = (df_pos['MHC Allele Prediction'].value_counts()).to_dict()
        allele_count_neg = (df_neg['MHC Allele Prediction'].value_counts()).to_dict()
        for allele in self.allele_dict.keys():
            allele_count_pos.setdefault(allele, 0)
            allele_count_neg.setdefault(allele, 0)

        pep_source_count_pos = (df_pos.drop_duplicates(subset=['Epitope Description'])['Peptide Source'].value_counts()).to_dict()
        pep_source_count_neg = (df_neg.drop_duplicates(subset=['Epitope Description'])['Peptide Source'].value_counts()).to_dict()
        for pep_source in self.pep_source_dict.keys():
            pep_source_count_pos.setdefault(pep_source, 0)
            pep_source_count_neg.setdefault(pep_source, 0)

        allele_freq_pos_vector = np.zeros(len(self.allele_dict))
        allele_freq_neg_vector = np.zeros(len(self.allele_dict))
        for allele, allele_id in self.allele_dict.items():
            allele_freq_pos_vector[allele_id] = allele_count_pos[allele]/num_pep_pos
            allele_freq_neg_vector[allele_id] = allele_count_neg[allele]/num_pep_neg

        pep_source_freq_pos_vector = np.zeros(len(self.pep_source_dict))
        pep_source_freq_neg_vector = np.zeros(len(self.pep_source_dict))
        for pep_source, pep_source_id in self.pep_source_dict.items():
            pep_source_freq_pos_vector[pep_source_id] = pep_source_count_pos[pep_source]/num_pep_pos
            pep_source_freq_neg_vector[pep_source_id] = pep_source_count_neg[pep_source]/num_pep_neg

        label_per_allele = {}
        for allele in self.allele_dict.keys():
            label_per_allele[allele] = [allele_count_pos[allele], allele_count_neg[allele]]

        label_per_source = {}
        for pep_source in self.pep_source_dict.keys():
            label_per_source[pep_source] = [pep_source_count_pos[pep_source], pep_source_count_neg[pep_source]]

        return label_per_allele, label_per_source, \
               torch.tensor(allele_freq_pos_vector, dtype=torch.float), torch.tensor(allele_freq_neg_vector, dtype=torch.float), torch.tensor(pep_source_freq_pos_vector, dtype=torch.float), torch.tensor(pep_source_freq_neg_vector, dtype=torch.float)


class BagOfAADataset(Dataset):
    def __init__(self, tcelldata, df, mhc_class='I+II', pep_source='all', sample='none'):
        if mhc_class != 'I+II':
            df = df.loc[df['MHC Class'] == mhc_class]
        if pep_source != 'all':
            df = df.loc[df['Epitope Parent Species'] == pep_source]

        if sample != 'none':
            df = df.sample(frac=sample)
            print('Sample')

            # make sure that the batch size is not larger than the whole dataset
            # otherwise no training takes place because all incomplete batches are dropped
            # print(len(df))
            assert len(df) >= tcelldata.params['batch_size']

        self.collate_fn = None
        self.tcelldata = tcelldata
        labels = df['Assay Qualitative Measure'].apply(lambda x: 1 if x == 'Positive' else 0).to_numpy()
        peptide_embeddings = np.vstack(df['Epitope Description'].apply(
            lambda peptide: self.bag_of_features(peptide)).to_list())

        if tcelldata.params['normalize_features']:
            peptide_normalization_data = np.vstack(
                tcelldata.train_data['Epitope Description'].apply(
                    lambda peptide: self.bag_of_features(peptide)).to_list())

            peptide_embeddings = tcelldata.normalize_embeddings(peptide_embeddings, peptide_normalization_data)

        self.labels = labels
        self.peptide_embeddings = peptide_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        peptide_embedding = torch.tensor(self.peptide_embeddings[idx])

        return {'peptide': peptide_embedding,
                'label': label}

    def bag_of_features(self, aa_sequence):
        encoding = np.zeros(len(self.tcelldata.amino_acids), dtype=np.float32)
        for aa in aa_sequence:
            encoding[self.tcelldata.amino_acids.index(aa)] += 1
        return np.array(encoding)


class PeptideDataset(Dataset):
    def __init__(self, tcelldata, df, mhc_class='I+II', mhc_allele='all', pep_source='all', sample='none'):
        if mhc_class != 'I+II':
            df = df.loc[df['MHC Class'] == mhc_class]
        if mhc_allele != 'all':
            df = df.loc[df['MHC Allele Prediction'] == mhc_allele]
        if pep_source != 'all':
            if pep_source == 'non-human':
                df = df.loc[df['Epitope Parent Species'] != 'Homo sapiens']
            elif pep_source == 'non-human-I':
                df = df.loc[~((df['Epitope Parent Species'] == 'Homo sapiens') & (df['MHC Class'] == 'I'))]
            elif pep_source == 'non-human-II':
                df = df.loc[~((df['Epitope Parent Species'] == 'Homo sapiens') & (df['MHC Class'] == 'II'))]
            else:
                df = df.loc[df['Epitope Parent Species'] == pep_source]
        if sample != 'none':
            df = df.sample(frac=sample)
            df = pd.concat([df, df])  # ensure that the dataset is not too small
            # make sure that the batch size is not larger than the whole dataset
            # otherwise no training takes place because all incomplete batches are dropped
            assert len(df) >= tcelldata.params['batch_size']

        labels = df['Assay Qualitative Measure'].apply(lambda x: 1 if x == 'Positive' else 0).to_numpy()

        mhc_presentation = df['MHC Allele Prediction - Min Percentile Rank'].to_numpy()

        peptide_seq_idxs = df['Epitope Description'].apply(
            lambda peptide_seq: [tcelldata.pep_start_token_idx,]+[tcelldata.amino_acid_to_id[amino_acid] for amino_acid in peptide_seq]+[tcelldata.pep_end_token_idx,]).to_list()

        pep_source_ids = df['Peptide Source'].apply(
            lambda pep_source: tcelldata.pep_source_dict[pep_source]).to_list()

        pep_source_indicator = df['Peptide Source'].apply(
            # lambda pep_source: 1 if pep_source == 'Human alphaherpesvirus 2' else 0).to_list()
            lambda pep_source: 1 if pep_source == 'Vaccinia virus' else 0).to_list()

        allele_ids, allele_vectors, allele_vectors_ones = tcelldata.parse_allele_vectors(df)

        self.tcelldata = tcelldata
        self.labels = labels
        self.mhc_presentation = mhc_presentation
        self.peptide_seq_idxs = peptide_seq_idxs
        self.pep_source_ids = pep_source_ids
        self.pep_source_indicator = pep_source_indicator
        self.allele_vectors = allele_vectors
        self.allele_vectors_ones = allele_vectors_ones

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        presentation_score = self.mhc_presentation[idx]
        peptide_seq_idxs = self.peptide_seq_idxs[idx]
        pep_source_id = self.pep_source_ids[idx]
        pep_source_indicator = self.pep_source_indicator[idx]
        allele_vectors_ones = self.allele_vectors_ones[idx]

        return {'peptide': peptide_seq_idxs,
                'allele': None,
                'pep_source': pep_source_id,
                'pep_source_indicator': pep_source_indicator,
                'label': label,
                'allele_vectors_ones': allele_vectors_ones,
                'mhc_presentation': presentation_score,
                }

    def collate_fn(self, batch):
        peptide_seq_list = [torch.tensor(batch_element['peptide']) for batch_element in batch]
        peptide_seq_padded = pad_sequence(peptide_seq_list, batch_first=True, padding_value=self.tcelldata.padding_token_idx)

        peptide_lengths = torch.tensor([len(batch_element['peptide']) for batch_element in batch])
        padding_mask = peptide_seq_padded == self.tcelldata.padding_token_idx

        labels = torch.tensor([batch_element['label'] for batch_element in batch], dtype=torch.float)
        presentation_scores = torch.tensor([batch_element['mhc_presentation'] for batch_element in batch], dtype=torch.float)
        allele_vectors_ones = torch.stack([torch.tensor(batch_element['allele_vectors_ones'], dtype=torch.float) for batch_element in batch])

        pep_sources = torch.tensor([batch_element['pep_source'] for batch_element in batch], dtype=torch.long)

        pep_source_indicator = torch.tensor([batch_element['pep_source_indicator'] for batch_element in batch], dtype=torch.long)

        return {
                'padding_mask': padding_mask,
                'peptide': peptide_seq_padded,
                'peptide_lengths': peptide_lengths,
                'label': labels,
                'mhc_presentation': presentation_scores,
                'allele_vectors_ones': allele_vectors_ones,
                'pep_source': pep_sources,
                'pep_source_indicator': pep_source_indicator,
                }
