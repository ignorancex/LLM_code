from surgeon_pytorch import Inspect
import torch


def perceptual_similarity_loss(fake_im, real_im, model, perceptual_layers):
    fake_im = fake_im.expand(-1, 3, -1, -1)  # .to("cuda:1")
    real_im = real_im.expand(-1, 3, -1, -1)  # .to("cuda:1")

    f1_real, f2_real, f3_real, f4_real, f5_real = vgg_16_pretrained(real_im, model)  # extract features from vgg
    f1_fake, f2_fake, f3_fake, f4_fake, f5_fake = vgg_16_pretrained(fake_im,
                                                                    model)  # extract features from vgg (1st,2nd, 3rd, 4th, 5th pool)

    if perceptual_layers == 'all':
        m1 = torch.mean(torch.mean((f1_real - f1_fake) ** 2))  # mse difference
        m2 = torch.mean(torch.mean((f2_real - f2_fake) ** 2))
        m3 = torch.mean(torch.mean((f3_real - f3_fake) ** 2))
        m4 = torch.mean(torch.mean((f4_real - f4_fake) ** 2))
        m5 = torch.mean(torch.mean((f5_real - f5_fake) ** 2))

        perceptual_loss = (m1 + m2 + m3 + m4 + m5)  # .to("cuda:0")

    elif perceptual_layers == '1-2':
        m1 = torch.mean(torch.mean((f1_real - f1_fake) ** 2))  # mse difference
        m2 = torch.mean(torch.mean((f2_real - f2_fake) ** 2))

        perceptual_loss = (m1 + m2)  # .to("cuda:0")

    elif perceptual_layers == '4-5':
        m4 = torch.mean(torch.mean((f4_real - f4_fake) ** 2))
        m5 = torch.mean(torch.mean((f5_real - f5_fake) ** 2))

        perceptual_loss = (m4 + m5)  # .to("cuda:0")

    elif perceptual_layers == '2-4':

        m2 = torch.mean(torch.mean((f2_real - f2_fake) ** 2))
        m4 = torch.mean(torch.mean((f4_real - f4_fake) ** 2))

        perceptual_loss = (m2 + m4)  # .to("cuda:0")

    return perceptual_loss


# EXTRACT THE POOLING LAYERS FROM THE PRETRAINED VGG-16
def vgg_16_pretrained(image, model):
    wrapped_model = Inspect(model, layer=['features.4', 'features.9', 'features.16', 'features.23', 'features.30'])
    _, [p1, p2, p3, p4, p5] = wrapped_model(image)

    return p1, p2, p3, p4, p5


if __name__ == "__main__":
    a = torch.ones((2,3,256,256))
    print(a.shape)
    b = a.expand(-1,3,-1,-1)
    print(b.shape)