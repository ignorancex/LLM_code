from src.TPP.dataloader.generic.generic_dataset import generic_dataloader


def get_dataloader():
    '''
    Required by dataloader_zoo() to further load data and create dataset objects.
    '''
    return generic_dataloader()
