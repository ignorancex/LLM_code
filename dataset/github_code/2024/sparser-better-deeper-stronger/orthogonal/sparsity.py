import torch

def er_densities(epsi, shapes):
    ''' Get densities for layers of given shapes.
        shapes: list[shape]
        Works for both: linear - shape has 2 entries, and convolutional - shape has 4 entries. '''
    densities_tensors = [epsi * torch.sum(torch.Tensor(shape)) / torch.prod(torch.Tensor(shape)) for shape in shapes]
    return [x.item() for x in densities_tensors]

def overal_density(epsi, shapes):
    ''' Compute overall density, if we applied er-based pruning with given epsi.
        shapes: list[shape] '''
    top = 0.0
    bottom = 0.0
    for shape in shapes:
        top += epsi * torch.sum(torch.Tensor(shape))
        bottom += torch.prod(torch.Tensor(shape))
    top = top.item()
    bottom = bottom.item()
    if top > bottom:
        print(f'Warning: encountered too large density: {round(top / bottom, 6)}. Consider making epsi lower.')
    return top / bottom

def er_layerwise_densities(shapes, target_ovr_density):
    ''' Find densities per layer given a target overall density.
        shapes: list[shape] 
        function performs bin search over the epsi parameter of er. '''
    epsi_lo = 0.0
    epsi_hi = 1e9 # some very large value
    acceptable_err = 0.01
    
    e_density = overal_density((epsi_lo + epsi_hi) / 2.0, shapes)
    while abs(e_density - target_ovr_density) > acceptable_err:
        if e_density > target_ovr_density:
            epsi_hi = (epsi_lo + epsi_hi) / 2.0
        else:
            epsi_lo = (epsi_lo + epsi_hi) / 2.0
        e_density = overal_density((epsi_lo + epsi_hi) / 2.0, shapes)
    
    target_epsi = (epsi_lo + epsi_hi) / 2.0
    result = er_densities(target_epsi, shapes)

    print(f'Found densities: {[round(x, 6) for x in result]}')
    print(f'Overall density: {round(overal_density(target_epsi, shapes), 6)}, when asked for {target_ovr_density}.')

    return result


if __name__ == '__main__':
    example_shapes = [(100, 784)]
    for _ in range(5):
        example_shapes.append((100, 100))
    example_shapes.append((10, 100))
    er_layerwise_densities(example_shapes, 0.1)
    ''' Outputs something like:
    Found densities: [0.073508, 0.130385, 0.130385, 0.130385, 0.130385, 0.130385, 0.717118]
    Overall density: 0.100459, when asked for 0.1.
    '''