
import torch

from .Filter import Filter


class ToTensorPairDifferenceVolume(Filter):
    '''
    Transform np.array to tensor
    '''
    
    def __init__(self, type=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._type = type

    def apply_one(self,  image, i, prepared):
        '''
        Format the input image for the network
        '''
        # print("TO TENSOR",image[1].shape)
        left_volume, right_volume = image[1],image[2]
        image = image[0]
        imagepair = image[1,:,:,:]
        image = image[0,:,:,:]

        # print("TO TENSOR1",image.shape)
        type_i = self._type if type(self._type) == str else self._type[i]
        if type_i == 'float':
            image = image.float()
        elif type_i == 'long':
            image = image.long()

        
        # print("TO TENSOR2",imagepair.shape)
        type_i = self._type if type(self._type) == str else self._type[i]
        if type_i == 'float':
            imagepair = imagepair.float()
        elif type_i == 'long':
            imagepair = imagepair.long()

        # return torch.unsqueeze(image, dim=0)
        # print('totensorpair')
        # return torch.cat((image,imagepair), 0)
        return [torch.cat((torch.unsqueeze(image, 0),torch.unsqueeze(imagepair, 0)), 0),left_volume, right_volume]