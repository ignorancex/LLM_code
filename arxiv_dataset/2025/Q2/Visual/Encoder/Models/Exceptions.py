
# This is a class of custom exceptions

###############################################################################
    
class TwoManyConvolutionalLayersException(Exception):

    def __init__(self, dimensionX, dimensionY):
        self.dimensionX = dimensionX
        self.dimensionY = dimensionY
    
    def __str__(self):
        message = 'The image dimensions at the end of the convolution layers are:' + \
                  ' ' + str(self.dimensionX) + ' and' + \
                  ' ' + str(self.dimensionY) + ',' + \
                  ' which are < 1.'
                  
        return message
            
class FlattenedNumberOfNeuronsLessThanFCMiddleLayersException(Exception):

    def __init__(self, flattenedDimensionsAtEndOfConvolutions, numberOfNeuronsAtOutputOfFirstFCMiddleLayer):
        self.flattenedDimensionsAtEndOfConvolutions      = flattenedDimensionsAtEndOfConvolutions
        self.numberOfNeuronsAtOutputOfFirstFCMiddleLayer = numberOfNeuronsAtOutputOfFirstFCMiddleLayer
    
    def __str__(self):
        message = 'The flattened dimensions at end of convolutions is' + \
                  ' ' + str(self.flattenedDimensionsAtEndOfConvolutions) + ' and ' + \
                  'the number of neurons at the output of the first FC middle layer is' + \
                  ' ' + str(self.numberOfNeuronsAtOutputOfFirstFCMiddleLayer) + '.' + \
                  'The first should be bigger than the second.'
                  
        return message
        
class ConcatenatedNeuronsLessThanFCFinalLayersException(Exception):

    def __init__(self, numberOfConcatenatedNeurons, numberOfNeuronsAtOutputOfFirstFCFinalLayer):
        self.numberOfConcatenatedNeurons                = numberOfConcatenatedNeurons
        self.numberOfNeuronsAtOutputOfFirstFCFinalLayer = numberOfNeuronsAtOutputOfFirstFCFinalLayer
    
    def __str__(self):
        message = 'The dimension after the concatenation with the depth and image ratio is' + \
                  ' ' + str(self.numberOfConcatenatedNeurons) + ' and ' + \
                  'the number of neurons at the output of the first FC final layer is' + \
                  ' ' + str(self.numberOfNeuronsAtOutputOfFirstFCFinalLayer) + '.' + \
                  'The first should be bigger than the second.'
                  
        return message