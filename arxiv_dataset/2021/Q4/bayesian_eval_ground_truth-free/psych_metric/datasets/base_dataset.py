"""
Base Dataset class to be inherited by other dataset classes
Overwrite these methods
"""
import ast
import csv

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class BaseDataset(object):

    def split_train_val_test(self):
        raise NotImplementedError

    @staticmethod
    def str_to_array(s):
        if isinstance(s, str):
            s = ast.literal_eval(s)
        return np.array(s)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def display(self, i):
        raise NotImplementedError

    def _check_dataset(self, dataset, dsets):
        """Checks if the dataset is in the provided set. Raises error if not.

        Params
        ------
        dataset : str
            string representing the dataset identity.
        dsets : set
            set of strings that is compared to for evaluating dataset's validity.
        """
        if dataset not in dsets:
            raise ValueError(str(dataset) + ' is not an acceptable dataset. Use only the following datasets: ' + str(dsets))

    def read_csv_to_dict(self, csvpath, sep=',', key_column=0, value_column=1, key_dtype=str):
        """Read the csv as a dict where key value pair is first two columns."""
        #TODO I do not see this being used often, perhaps make another subclass of BaseDataset, and all that use this extend that class.
        csv_dict = {}
        with open(csvpath, 'r') as f:
            csv_reader = csv.reader(f, delimiter=sep)

            if key_dtype != str:
                for row in csv_reader:
                    csv_dict[key_dtype(row[key_column])] = row[value_column]
            else:
                # No need to cast the row key value.
                for row in csv_reader:
                    csv_dict[row[key_column]] = row[value_column]
        return csv_dict

    def add_ground_truth(self, ground_truth, ground_truth_header='ground_truth', sample_id='sample_id', inplace=False):
        """ Add the ground truth labels to every sample (row) of the main
        dataframe in its own column.

        Parameters
        ----------
        ground_truth : pandas.DataFrame
            Dataframe of the ground truth,
        inplace : bool, optional
            Update dataframe in place if True, otherwise return the updated
            dataframe
        sample_id : str
            the column name that identifies the sample to match to its
            corresponding ground truth value.

        Returns
        -------
        pd.DataFrame
            DataFrame with a ground truth labels column returned if inplace is
            False, otherwise returns None.
        """
        if isinstance(ground_truth, pd.DataFrame):
            # converts to dict first, may or may not be efficent.
            ground_truth_dict = {}
            for i in range(len(ground_truth)):
                ground_truth_dict[ground_truth['sample_id'][i]] = ground_truth['ground_truth'][i]
            ground_truth = ground_truth_dict
        elif not isinstance(ground_truth, dict):
            raise TypeError('`ground_truth` must be a dictionary of sample_ids to ground truth labels or a pd.DataFrame containing sample_id and ground_truth columns, not of type ' + str(type(ground_truth)))

        if self.is_in_annotation_list_format():
            ground_truth_col = self.df[sample_id].apply(lambda x: ground_truth[x] if x in ground_truth else None)
        elif isinstance(self.df, pd.SparseDataFrame):
            # TODO, create ground truth column where the index of df is the sample_id.
            ground_truth_col = pd.Series(self.df[sample_id]).apply(lambda x: ground_truth[x] if x in ground_truth else self.df.default_fill_value)
        else:
            raise TypeError('Cannot add ground truth to a dataframe with a non-standard format.')

        if inplace:
            self.df[ground_truth_header] = ground_truth_col
        else:
            df_copy = self.df.copy()
            df_copy[ground_truth_header] = ground_truth_col
            return df_copy


    def encode_labels(self, columns=None, column_labels=None, by_index=False):
        """Initializes a label encoder and fits it to the expected labels of
        the data column.

        This creates a label encoder and encodes a column in the dataframe with
        that label encoder.

        Parameters
        ----------
        column : list, optional
            list of str or int objects representing the columns to encode. If
            column is not included in dictionary keys, but is in `columns`, then
            the column will encode but using only the unique values seen from
            the data (will not include missing values).
        column_labels : dict, optional
           dict of column names to target labels for fitting the label encoder
           for that column.
        by_index : bool, optional
            Accesses column by column index if True, by column label if False

        Returns
        -------
        {str : sklearn.preprocessing.LabelEncoder}
            column name to LabelEncoder for the labels to their numeric
            representations.
        """
        label_encoders = dict()

        if columns is None and column_labels is None:
            raise ValueError('need to provide either the columns or column_labels parameters.')

        # if column_labels is provided, encode them
        if column_labels is not None:
            for column in column_labels.keys():
                label_encoder = LabelEncoder()
                label_encoder.fit(list(column_labels[column]))

                if by_index:
                    self.df.iloc[:,column] = label_encoder.transform(self.df.iloc[:,column])
                else:
                    self.df[column] = label_encoder.transform(self.df[column])

                label_encoders[column] = label_encoder

        # if columns is provided, encode them if not already encoded
        if columns is not None:
            if column_labels is not None:
                # Only encode the columns not in column_labels
                columns = set(columns) - set(column_labels.keys())

            for column in columns:
                label_encoder = LabelEncoder()

                if by_index:
                    self.df.iloc[:,column] = label_encoder.fit_transform(self.df.iloc[:,column])
                else:
                    self.df[column] = label_encoder.fit_transform(self.df[column])

                label_encoders[column] = label_encoder

        return label_encoders

    #TODO rename this to `sparse_matrix_to_annotation_list` for clarity
    def sparse_matrix_to_annotation_list(self, missing_value=None, inplace=False):
        """Lossy conversion of df as a sparse dataframe into a annotation list
        equivalent dataframe format, where the rows are the different instance
        of annotations by individual annotators and the columns are 'sample_id',
        'worker_id', and 'worker_label'.

        Parameters
        ----------
        missing_value :
            The missing_value of the sparse matrix that represents when an
            annotator did not annotate a sample. By default this uses the
            pandas.SparseDataFrame.default_fill_value.
        inplace : bool
            Will update the pandas.DataFrame inplace if True, otherwise returns
            the resulting dataframe.

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations in an annotation list format.

        """
        # Check if the current dataframe is a sparse matrix
        if not isinstance(self.df, pd.SparseDataFrame):
            # Assumes df will not be a sparse dataframe if in any other format.
            return None

        if missing_value is None:
            missing_value = self.df.default_fill_value

        if 'ground_truth' in self.df.columns:
            num_workers = len(self.df.columns) - 1
            list_df = pd.DataFrame(np.empty((len(self.df.index)*num_workers, 4)), columns=['sample_id', 'worker_id', 'worker_label', 'ground_truth'])
        else:
            num_workers = len(self.df.columns)
            list_df = pd.DataFrame(np.empty((len(self.df.index)*num_workers, 3)), columns=['sample_id', 'worker_id', 'worker_label'])

        # Place the correct value in the resulting dataframe list
        if 'ground_truth' in self.df.columns:
            for sample_idx in range(len(self.df.index)):
                for worker_idx in range(num_workers):
                    list_df.iloc[(sample_idx*num_workers) + worker_idx] = [sample_idx, worker_idx, self.df.iloc[sample_idx, worker_idx], self.df['ground_truth'].iloc[sample_idx]]
        else:
            for sample_idx in range(len(self.df.index)):
                for worker_idx in range(num_workers):
                    list_df.iloc[(sample_idx*num_workers) + worker_idx] = [sample_idx, worker_idx, self.df.iloc[sample_idx, worker_idx]]

        # remove all rows with missing values for labels from dataframe list.
        list_df = list_df[~list_df.eq(missing_value).any(1)]

        if not inplace:
            return list_df
        self.df = list_df

    def is_in_annotation_list_format(self):
        """Checks if df is in annotation list format."""
        # NOTE assumes that df is always a pd.DataFrame if in this list format.
        return isinstance(self.df, pd.DataFrame) and 'sample_id' in self.df.columns and 'worker_id' in self.df.columns and 'worker_label' in self.df.columns

    def annotation_list_to_sparse_matrix(self, fill_value=None, inplace=False):
        """Convert provided dataframe into a matrix format, possibly sparse.

        Parameters
        ----------
        fill_value : optional
            The value to be used as the fill value for the sparse matrix. If not
            provided then `None` will be used.
        inplace : bool, optional
            Will overwrite the current df attribute if True, outputs the sparse
            dataframe equivalent if False. Overwriting the current df may result
            in loss of data, such as ground truth, features, etc. Use with
            caution.

        Returns
        -------
            If inplace is False, then pd.SparseDataFrame equivalent of self.df,
            otherwise None. If df is not an annoation list, returns None.
        """
        # Check if the current dataframe is in annotation list format
        if not self.is_in_annotation_list_format():
            return None

        unique_sample_ids, unique_sample_ids_inverse = np.unique(self.df['sample_id'], return_inverse=True)
        unique_worker_ids, unique_worker_ids_inverse = np.unique(self.df['worker_id'], return_inverse=True)

        # Create a dictionary for mapping of all workers to their matrix index
        worker_to_matrix_idx = {worker:i for i, worker in enumerate(unique_worker_ids)}

        # TODO Store visited workers per each sample to detect multiple samplings
        # NOTE unnecesary, can check if value is fill_value at matrix spot prior to filling, this informs of duplicates and will then require whatever action to be taken. Could default for now to using the last seen label value of the worker for that sample.

        # TODO Put the existing annotations into the matrix, by each sample
        if 'ground_truth' in self.df.columns:
            # Create the initial dense matrix and fill it's entries with the missing_value
            matrix = np.full((len(unique_sample_ids), len(unique_worker_ids) + 1), None)

            for i, sample_id in enumerate(unique_sample_ids):
                # Get worker's labels for this sample
                sample_instances = np.nonzero(unique_sample_ids_inverse == i)[0]
                # Put each worker's label into the sample_id row of matrix
                for sample_idx in sample_instances:
                    # TODO handle multiple labels of a sample from one worker
                    #if matrix[i][worker_to_matrix_idx[self.df['worker_id'].iloc[sample_idx]]] != missing_value
                    matrix[i, worker_to_matrix_idx[self.df['worker_id'].iloc[sample_idx]]] = self.df['worker_label'].iloc[sample_idx]
                    matrix[i, -1] = self.df['ground_truth'].iloc[sample_idx]
        else:
            # Create the initial dense matrix and fill it's entries with the missing_value
            matrix = np.full((len(unique_sample_ids), len(unique_worker_ids)), None)

            for i, sample_id in enumerate(unique_sample_ids):
                # Get worker's labels for this sample
                sample_instances = np.nonzero(unique_sample_ids_inverse == i)[0]
                # Put each worker's label into the sample_id row of matrix
                for sample_idx in sample_instances:
                    # TODO handle multiple labels of a sample from one worker
                    #if matrix[i][worker_to_matrix_idx[self.df['worker_id'].iloc[sample_idx]]] != missing_value
                    matrix[i, worker_to_matrix_idx[self.df['worker_id'].iloc[sample_idx]]] = self.df['worker_label'].iloc[sample_idx]

        # Make the matrix a sparse dataframe.
        matrix = pd.DataFrame(matrix, columns=unique_worker_ids, index=unique_sample_ids).to_sparse(fill_value)

        if inplace:
            self.df = matrix
            return None
        else:
            return matrix


    def truth_inference_survey_format(self, inplace=False):
        """Creates the two dictionaries the Truth Inference Survey's code
        expects from the dataframe in list format into the two.

        Returns
        -------
            Tuple of a dictionary of sample identifiers as keys and values as list of
            annotator id and their annotation, and a dictionary of annotator
            identifeirs as keys and values as list of samples.
        """
        # NOTE may want this to handle more standard format versions of data.
        # If in sparse matrix format, convert to annotation list
        if inplace:
            if isinstance(self.df, pd.SparseDataFrame):
                self.convert_to_annotation_list(inplace=inplace)
            df = self.df
        else:
            df = self.sparse_matrix_to_annotation_list() if isinstance(self.df, pd.SparseDataFrame) else self.df.copy()

        samples_to_annotations = dict()
        annotators_to_samples = dict()

        for index, row in df.iterrows():
            # add to samples_to_annotations
            if row['sample_id'] not in samples_to_annotations:
                samples_to_annotations[row['sample_id']] = [[row['worker_id'], row['worker_label']]]
            else:
                samples_to_annotations[row['sample_id']].append([row['worker_id'], row['worker_label']])

            # add to annotators_to_samples
            if row['worker_id'] not in annotators_to_samples:
                annotators_to_samples[row['worker_id']] = [[row['sample_id'], row['worker_label']]]
            else:
                annotators_to_samples[row['worker_id']].append([row['sample_id'], row['worker_label']])

        return samples_to_annotations, annotators_to_samples

    def statistics_summary(self, filepath=None):
        """Calculate and return the dataset statistics, such as total number of
        annotations, annoators, samples, mean and 5 quantiles of both
        annotations per sample and samples per annotator.

        Parameters
        ----------
        filepath : str
            If filepath is provided, then the summary statistics and other
            defining characteristics of the data (ie. task_type, id, and
            data_filepath) to the designated filepath as a csv.
        """
        # TODO implement dataset analysis function(s) for finding the
        raise NotImplementedError
        # add domain of dataset, such as language, face images, etc.
