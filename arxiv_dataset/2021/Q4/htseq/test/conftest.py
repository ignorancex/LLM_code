import os
import pytest


def get_data_folder():
    cwd = os.getcwd().rstrip('/')
    if cwd.endswith('example_data'):
        return cwd+'/'
    else:
        return cwd+'/example_data/'


def get_docs_folder():
    cwd = os.getcwd().rstrip('/')
    if cwd.endswith('doc'):
        return cwd+'/'
    elif cwd.endswith('example_data'):
        return cwd[:-len('example_data')]+'doc/'
    else:
        return cwd+'/doc/'


# Same as fixtures
@pytest.fixture(scope="module")
def data_folder():
    return get_data_folder()


@pytest.fixture(scope="module")
def docs_folder():
    return get_docs_folder()



