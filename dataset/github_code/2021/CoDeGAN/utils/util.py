import torch
import torchvision.transforms as transforms
import torch.distributions.categorical as categorical
from   torch import autograd


# sample noises
def sample_noise(batch_size, zn_dim, zc_dim, device='cpu'):
    set_index = categorical.Categorical\
        (torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    zn      = torch.randn(batch_size, zn_dim)
    m_zeros = torch.zeros(batch_size, zc_dim)
    zc_idx  = [set_index.sample().view(-1,1) for _ in range(batch_size)]
    zc_idx  = torch.cat(zc_idx, dim=0)
    zc      = m_zeros.scatter_(1, zc_idx, 1)
    z       = torch.cat((zn, zc), dim=1)
    if device != 'cpu':
        return z.to(device), zn.to(device), zc_idx.squeeze().to(device)

    return z, zn, zc_idx.squeeze()


# sample real data
def sample_real_data(batch_size, real_datasets, class_num=10):
    sampled_data   = []
    sampled_labels = []
    for i in range(class_num):
        batch_index  = list(torch.utils.data.WeightedRandomSampler \
            (weights=[1 for _ in range(real_datasets[i][0].size(0))], \
            num_samples=batch_size//class_num, replacement=False))

        sampled_data.append(  real_datasets[i][0][batch_index])
        sampled_labels.append(real_datasets[i][1][batch_index])

    sampled_data   = torch.cat(sampled_data, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    return [sampled_data, sampled_labels]


# calculate gradient penalty
def calc_gradient_penalty(real_data, generated_data, D):
    LAMBDA = 10
    device = real_data.device
    b_size = real_data.size()[0]
    alpha  = torch.rand(b_size, 1, 1, 1)
    alpha  = alpha.expand_as(real_data)
    alpha  = alpha.to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = autograd.Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    prob_interpolated = D(interpolated)
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                        create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(b_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# image augmentation
def train_transform(image, detach=False):
    if image.shape[1]==3:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.)/ 2.),
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.Lambda(lambda x: x * 2. - 1.),
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.)/ 2.),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x * 2. - 1.),
        ])


    image_tran = transform(image)
    if detach:
        return image_tran.detach()
    
    return image_tran\

# image augmentation
def train_transform_simple(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.)),
    ])

    return transform(image).detach()

def sample_noise_with_label(batch_size, zn_dim, zc_dim, device, labels):
    zn = torch.randn(batch_size, zn_dim).to(device)
    zc = labels * torch.ones((batch_size, 1)).long().to(device)
    zc_dix = zc.float()
    m_zeros = torch.zeros(batch_size, zc_dim).to(device)
    zc = m_zeros.scatter_(1, zc, 1)
    z = torch.cat((zn, zc), dim=1).to(device)
    return z, zc_dix


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]