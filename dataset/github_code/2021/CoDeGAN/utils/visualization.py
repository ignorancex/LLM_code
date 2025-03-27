import torch
import torchvision


# visualize generated images
def sample_fake_image(G, zn_dim, zc_dim, device):
    fake_image = []
    for labels in range(zc_dim):
        z_fc = torch.randn(10, zn_dim)
        label_f = labels * torch.ones((10, 1)).long()
        m_zeros = torch.zeros(10, zc_dim)
        label_f = m_zeros.scatter_(1, label_f, 1)
        z_f = torch.cat((z_fc, label_f), dim=1).to(device)
        fake_image.append(G(z_f))
    return torch.cat(fake_image, dim=0)

def visualize_result(epoch, model_dir, G, zn_dim, zc_dim, device):
    stack_imgs = sample_fake_image(G, zn_dim, zc_dim, device)
    torchvision.utils.save_image(stack_imgs, \
         model_dir + '/gen_classes{}.png'.format(epoch), nrow=zc_dim, normalize=True)
