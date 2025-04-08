import torch
import torch.nn as nn


class EncoderDecoderNetwork(nn.Module):
    '''
    Abstract class representing an encoder/decoder network.
    It implements the basic methods of a model that takes an image as an input, and export another image
    with the same size of the input image.
    '''

    def __init__(self, num_input_channels, num_output_channels, **args):
        ''' 
        Constructor.
        -------

        '''
        super().__init__()

        # set the name of the model
        self.name = 'encoder-decoder-model'

        # Activation function to produce the scores
        self.final_activation = getattr(nn, args.get('activation_function','Identity'))()

        # Number of channels in the input
        self.num_input_channels = num_input_channels
        # Number of channels in the output
        self.num_output_channels = num_output_channels

        # setup default configuration
        self.rep_dim = args.get('rep_dim', True)
        self.use_batchnorm = args.get('batch_norm', True)
        self.activation_unit = args.get('activation_unit','ReLU')
        self.hipo_diff = args.get('hipo_diff', True)
        self.fc_out = args.get('fc_out', True)
        self.VolumeDifference = args.get('VolumeDifference', True)

    @property
    def device(self):
        return next(self.parameters()).device

    def test(self, image):
        '''
        Predict a formatted output from an image.
        Use this method in test time to get the expected output from the image.
        '''
        # print('test EncoderDecoderNetwork')
        # expand a dimension to simulate a batch
    
        if image.shape[0] == 1:
            image = torch.unsqueeze(image, dim=0).to(self.device)
            output = self.forward(image)
            self.predict(image)
            return output[0].detach().cpu()

        else:
            output = self.forward(image.to(self.device))
            return output

    def predict(self, image):
        # print('predict EncoderDecoderNetwork')
        output = self.forward(image) #torch.Size([12, 1, 64, 64, 64])
        # output = self.predict(image) #torch.Size([12, 1, 64, 64, 64])
        return output



