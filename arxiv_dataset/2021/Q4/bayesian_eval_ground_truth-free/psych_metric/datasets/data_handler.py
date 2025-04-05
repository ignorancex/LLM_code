# TODO this needs handled in the __init__ files/setup.py installation of the package.
from psych_metric import datasets

def dataset_exists(dataset, collection=None):
    """Returns true if the provided string identifier matches a dataset in
    some collections.
    """
    if collection is None:
        # checks if the dataset exists in any collection
        return isinstance(dataset, str) and (dataset in datasets.Snow2008.datasets or dataset in datasets.TruthSurvey2017.datasets or dataset == 'trec-rf10-data' or dataset in datasets.Ipeirotis2010.datasets or dataset in datasets.FacialBeauty.datasets or dataset in datasets.CrowdLayer.datasets or dataset in datasets.FirstImpressions.datasets)
    elif collection == 'truth_survey_2017':
        # Check if dataset in truth_survey_2017
        return dataset in datasets.TruthSurvey2017.datasets
    elif collection == 'snow_2008':
        # Check if dataset in snow_2008
        return dataset in datasets.Snow2008.datasets
    elif collection == 'ipeirotis_2010':
        return dataset in datasets.Ipeirotis2010.datasets
    elif collection == 'facial_beauty':
        return dataset in datasets.FacialBeauty.datasets
    elif collection == 'crowd_layer':
        return dataset in datasets.CrowdLayer.datasets
    elif collection == 'first_impressions':
        return dataset in datasets.FirstImpressions.datasets

    return dataset == 'trec-rf10-data'

def load_dataset(dataset, *args, **kwargs):
    """
    Handler function for calling the correct datahandler class for a given
    dataset.

    Parameters
    ----------
    dataset : str
        name of dataset represented as a string.

    Returns
    -------
    psych_metric.dataset.base_dataset
        A dataset object corresponding to the given dataset str requested.
    """
    #Snow 2008
    if dataset_exists(dataset, 'snow_2008'):
        return datasets.Snow2008(dataset, *args, **kwargs)

    # Truth Survey 2017
    elif dataset_exists(dataset, 'truth_survey_2017'):
        return datasets.TruthSurvey2017(dataset, *args, **kwargs)

    # TREC 2010
    elif dataset_exists(dataset, 'trec-fr10'):
        return datasets.TRECRelevancy2010(*args, **kwargs)

    # Ipeirotis
    elif dataset_exists(dataset, 'ipeirotis_2010'):
        return datasets.Ipeirotis2010(dataset, *args, **kwargs)

    # Facial Beauty
    elif dataset_exists(dataset, 'facial_beauty'):
        return datasets.FacialBeauty(dataset, *args, **kwargs)

    # First Impressions
    # elif dataset_exists(dataset, 'first_impressions'):
    #    return datasets.FirtImpressions(dataset, *args, **kwargs)

    # CrowdLayer
    elif dataset_exists(dataset, 'crowd_layer'):
        return datasets.CrowdLayer(dataset, *args, **kwargs)

    # TODO Google fact evaluation, low priority.
