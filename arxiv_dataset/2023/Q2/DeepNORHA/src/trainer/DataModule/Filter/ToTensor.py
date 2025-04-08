
import torch

from .Filter import Filter


class ToTensor(Filter):
    '''
    Transform np.array to tensor
    '''
    
    def __init__(self, type=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._type = type
        # print('tensor1')

    def apply_one(self,  image, i, prepared):
        '''
        Format the input image for the network
        '''
        # print("TO TENSOR",image.shape)
        
        # if len(image.shape) == 3:
        #     image = image.transpose(2, 0, 1)
        image = torch.tensor(image)

        type_i = self._type if type(self._type) == str else self._type[i]
        if type_i == 'float':
            image = image.float()
        elif type_i == 'long':
            image = image.long()
        # print('tensor2')
        return torch.unsqueeze(image, dim=0)