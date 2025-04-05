"""
Implementation of Base GAN models.
"""
import torch

from torch_mimicry.nets.basemodel import basemodel
from torch_mimicry.modules import losses
from torch.autograd import Variable

class BaseGenerator(basemodel.BaseModel):
    r"""
    Base class for a generic unconditional generator model.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.nz = nz
        self.ngf = ngf
        self.bottom_width = bottom_width
        self.loss_type = loss_type

    def generate_images(self, num_images, device=None):
        r"""
        Generates num_images randomly.

        Args:
            num_images (int): Number of images to generate
            device (torch.device): Device to send images to.

        Returns:
            Tensor: A batch of generated images.
        """
        if device is None:
            device = self.device

        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise)

        return fake_images

    def compute_gan_loss(self, output):
        r"""
        Computes GAN loss for generator.

        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).

        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        # Compute loss and backprop
        if self.loss_type == "gan":
            errG = losses.minimax_loss_gen(output)

        elif self.loss_type == "ns":
            errG = losses.ns_loss_gen(output)

        elif self.loss_type == "hinge":
            errG = losses.hinge_loss_gen(output)

        elif self.loss_type == "wasserstein":
            errG = losses.wasserstein_loss_gen(output)

        else:
            raise ValueError("Invalid loss_type {} selected.".format(
                self.loss_type))

        return errG

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        # Compute output logit of D thinking image real
        output = netD(fake_images)

        # Compute loss
        errG = self.compute_gan_loss(output=output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class BaseDiscriminator(basemodel.BaseModel):
    r"""
    Base class for a generic unconditional discriminator model.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.ndf = ndf
        self.loss_type = loss_type

    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """
        # Compute loss for D
        if self.loss_type == "gan" or self.loss_type == "ns":
            errD = losses.minimax_loss_dis(output_fake=output_fake,
                                           output_real=output_real)

        elif self.loss_type == "hinge":
            errD = losses.hinge_loss_dis(output_fake=output_fake,
                                         output_real=output_real)

        elif self.loss_type == "wasserstein":
            errD = losses.wasserstein_loss_dis(output_fake=output_fake,
                                               output_real=output_real)

        else:
            raise ValueError("Invalid loss_type selected.")

        return errD

    def compute_probs(self, output_real, output_fake):
        r"""
        Computes probabilities from real/fake images logits.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.

        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        D_x = torch.sigmoid(output_real).mean().item()
        D_Gz = torch.sigmoid(output_fake).mean().item()

        return D_x, D_Gz

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()
        real_images, real_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce logits for real images
        output_real = self.forward(real_images)

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        # Produce logits for fake images
        output_fake = self.forward(fake_images)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
    def advtrain_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one adv_training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()
        real_images, real_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce logits for real images
        output_real = self.forward(real_images)

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        # Produce logits for fake images
        output_fake = self.forward(fake_images)

        #compute the adversarial samples of real and fake images.
        t=1
        real_value=torch.mean(output_real)
        fake_value=torch.mean(output_fake)
        fake_imgs_adv=fake_images.clone()
        real_imgs_adv=real_images.clone()
        real_imgs_adv=Variable(real_imgs_adv,requires_grad=True)
        fake_imgs_adv=Variable(fake_imgs_adv,requires_grad=True)
        #real_grad=Variable(real_grad,requires_grad=True)
        fake_output= self.forward(fake_imgs_adv)
        fake_output=fake_output.mean()
        fake_adv_loss = torch.abs(fake_output-real_value)
        #print(fake_adv_loss)
        #print(fake_adv_loss.requires_grad)
        #print(fake_imgs_adv.requires_grad)
        fake_grad=torch.autograd.grad(fake_adv_loss,fake_imgs_adv)
        fake_imgs_adv=fake_imgs_adv-fake_grad[0].clamp(-1*t,t)
        fake_imgs_adv=fake_imgs_adv.clamp(-1,1)
        real_output= self.forward(real_imgs_adv)
        real_output=real_output.mean()
        real_adv_loss = torch.abs(real_output-fake_value)
        real_grad=torch.autograd.grad(real_adv_loss,real_imgs_adv)
        real_imgs_adv=real_imgs_adv-real_grad[0].clamp(-1*t,t)
        fake_adv_validity= self.forward(fake_imgs_adv.detach())
        real_adv_validity = self.forward(real_imgs_adv)
        real_imgs_adv=real_imgs_adv.clamp(-1,1) 

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=real_adv_validity,
                                     output_fake=fake_adv_validity)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=real_adv_validity,
                                       output_fake=fake_adv_validity)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
