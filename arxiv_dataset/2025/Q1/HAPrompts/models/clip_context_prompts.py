def food_101():
    """
        CLIP template for food-101 dataset
    """
    templates = [
        'a photo of {}, a type of food.',
    ]
    return templates


def ucf_101():
    templates = [
        'a photo of a person {}.',
        'a video of a person {}.',
        'a example of a person {}.',
        'a demonstration of a person {}.',
        'a photo of the person {}.',
        'a video of the person {}.',
        'a example of the person {}.',
        'a demonstration of the person {}.',
        'a photo of a person using {}.',
        'a video of a person using {}.',
        'a example of a person using {}.',
        'a demonstration of a person using {}.',
        'a photo of the person using {}.',
        'a video of the person using {}.',
        'a example of the person using {}.',
        'a demonstration of the person using {}.',
        'a photo of a person doing {}.',
        'a video of a person doing {}.',
        'a example of a person doing {}.',
        'a demonstration of a person doing {}.',
        'a photo of the person doing {}.',
        'a video of the person doing {}.',
        'a example of the person doing {}.',
        'a demonstration of the person doing {}.',
        'a photo of a person during {}.',
        'a video of a person during {}.',
        'a example of a person during {}.',
        'a demonstration of a person during {}.',
        'a photo of the person during {}.',
        'a video of the person during {}.',
        'a example of the person during {}.',
        'a demonstration of the person during {}.',
        'a photo of a person performing {}.',
        'a video of a person performing {}.',
        'a example of a person performing {}.',
        'a demonstration of a person performing {}.',
        'a photo of the person performing {}.',
        'a video of the person performing {}.',
        'a example of the person performing {}.',
        'a demonstration of the person performing {}.',
        'a photo of a person practicing {}.',
        'a video of a person practicing {}.',
        'a example of a person practicing {}.',
        'a demonstration of a person practicing {}.',
        'a photo of the person practicing {}.',
        'a video of the person practicing {}.',
        'a example of the person practicing {}.',
        'a demonstration of the person practicing {}.',
    ]
    return templates


def cub_200():
    """
        CLIP template for birdsnap dataset
        we may reuse it for CUB-200
    """
    templates = [
        'a photo of a {}, a type of bird.',
    ]
    return templates


def sun_324(): 
    """
        CLIP template for SUN-397/SUN-324 dataset
        in our case, we filter the classes from 397 down to 324
        due to ambiguity in some hierarchical labels
    """
    templates = [
        'a photo of a {}.',
        'a photo of the {}.',
    ]
    return templates


def imagenet():
    """
        CLIP ensemble template for imagenet
    """
    templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return templates


def select_template(opts):
    assert opts.prompt == 'clip', f'not recognize prompt: {opts.prompt}'
    if opts.dataset == 'food-101': return food_101()
    elif opts.dataset == 'ucf-101': return ucf_101()
    elif opts.dataset == 'cub-200': return cub_200()
    elif opts.dataset == 'sun-324': return sun_324()
    elif opts.dataset == 'imagenet': return imagenet()
    else:
        raise NotImplementedError()
