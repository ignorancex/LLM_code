from src.TPP.dataloader.ifl.ifl_dataset import ifl_dataloader


def get_dataloader():
    '''
    Required by dataloader_zoo() to further load data and create dataset objects.
    '''
    return ifl_dataloader()