class ToFloatMNIST(object):
    """Convert ndarrays in sample values to integers."""

    def __call__(self, image):
        return (image / 255).astype('f4')
