
import torch
import torch.nn.functional as F
import random
import torchvision
from tqdm import tqdm
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

from core.config import DEVICE


class LossObject:
    def __init__(self, loss_function, at_channels=None, at_pixels=None):
        self.loss_function = loss_function
        self.at_channels = at_channels
        self.at_pixels = at_pixels

    def get_loss(self, net_output):
        #print("mean net output: {}".format(torch.mean(net_output)))
        return self.loss_function(net_output)

    def __call__(self, net_output):
        #print("self.at_channels: {}".format(self.at_channels))
        loss = self.get_loss(net_output)
        if self.at_channels is not None:
            loss_out = 0
            if isinstance(self.at_channels, int):
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                #return torch.mean(loss_out)
            elif len(self.at_channels.shape) == 0:
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                #return torch.mean(loss_out)
            #print("at channels: {}".format(self.at_channels))
            else:
                for channel in self.at_channels:
                    loss_out = loss_out + loss[:, channel, :, :]
            if self.at_pixels is not None:
                loss_out_ = torch.clone(loss_out)
                loss_out = 0
                for pixel in self.at_pixels:
                    if len(pixel) == 3:
                        y = pixel[1]
                        x = pixel[2]
                    elif len(pixel) == 2:
                        y = pixel[0]
                        x = pixel[1]
                    else:
                        raise RuntimeError("invalid pixel data")
                    loss_out = loss_out + loss_out_[:, y, x]
            return torch.mean(loss_out)

            #return torch.mean(loss_out)
        elif self.at_pixels is not None:
            loss_out = 0
            for pixel in self.at_pixels:
                if len(pixel) == 3:
                    y = pixel[1]
                    x = pixel[2]
                elif len(pixel) == 2:
                    y = pixel[0]
                    x = pixel[1]
                else:
                    raise RuntimeError("invalid pixel data")
                loss_out = loss_out + loss[:, :, y, x]
            return torch.mean(loss_out)

        return torch.mean(loss)

class LossObject_Target:
    def __init__(self, loss_function, target, at_channels=None, at_pixels=None):
        self.loss_function = loss_function
        self.target = target
        self.target_internally = target
        self.at_channels = at_channels
        self.at_pixels = at_pixels

    def get_loss(self, net_output):
        #print(torch.mean(self.target_internally))
        return self.loss_function(net_output, self.target_internally.to("cuda"))

    def __call__(self, net_output):
        loss = self.get_loss(net_output)#self.loss_function(net_output, self.target.to("cuda"))
        if self.at_channels is not None:
            loss_out = 0
            if isinstance(self.at_channels, int):
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                return torch.mean(loss_out)
            elif len(self.at_channels.shape) == 0:
                loss_out = loss_out + loss[:, self.at_channels, :, :]
                return torch.mean(loss_out)
            for channel in self.at_channels:
                loss_out = loss_out + torch.mean(loss[:, channel, :, :])
            return torch.mean(loss_out)
        elif self.at_pixels is not None:
            loss_out = 0
            for pixel in self.at_pixels:
                if len(pixel) == 3:
                    y = pixel[1]
                    x = pixel[2]
                elif len(pixel) == 2:
                    y = pixel[0]
                    x = pixel[1]
                else:
                    raise RuntimeError("invalid pixel data")
                loss_out = loss_out + loss[:, :, y, x]
            return torch.mean(loss_out)

        return torch.mean(loss)





class OptimGoal_ForwardHook:
    def __init__(self, hook_point, loss_object, weight=-1.0, forward_func=None):
        #self.handle = hook_point.register_forward_hook(self.hook_function)
        self.hook_point = hook_point
        self.loss_object = loss_object
        self.hook_output = None
        self.weight = weight
        self.forward_func = forward_func

    def set_handle(self):
        self.handle = self.hook_point.register_forward_hook(self.hook_function)

    def remove_handle(self):
        self.hook_output = None
        self.handle.remove()

    def run_forward(self, visualization):
        if self.forward_func is not None:
            self.forward_func(visualization)


    def hook_function(self, module, input_, output):
        self.hook_output = output

    def compute_loss(self, visualization):
        self.run_forward(visualization)
        output = self.hook_output
        #print("output.shape: {}".format(output.shape))
        loss = self.loss_object(output)
        #print("loss: {}".format(loss))
        #self.hook_output.detach()

        return loss*self.weight



def hook_gan_single_vector(module, input_, output):
    #print("hook input gan shape: {}".format(output.shape))
    #print("output shape: {}".format(output.shape))
    #out = torch.ones_like(output)*hook_input_gan.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    out = torch.ones_like(output)*hook_input_gan.unsqueeze(dim=-1).unsqueeze(dim=-1)
    #print("out shape: {}".format(out.shape))
    return out#F.relu(hook_input_gan)


hook_input_gan = None
def hook_gan(module, input_, output):
    #print("hook input gan shape: {}".format(hook_input_gan.shape))
    #print("output shape: {}".format(output.shape))
    #return output
    return hook_input_gan#F.relu(hook_input_gan)


def optim_gan(initial_gan_activations, forward_function_gan_batch_input, forward_hook_gan, forward_function_seg_net, optim_goals,
              from_single_vector=False, num_steps=512,
              regularization='full', normalization_preprocessor=torch.nn.Identity(),
              num_darkening_steps=0):
    global hook_input_gan

    #regularization = 'none'

    batch_size = initial_gan_activations.shape[0]

    for optim_goal in optim_goals:
        optim_goal.set_handle()

    #start_activations =

    #input_seg_net = input_seg_net.to("cuda")
    initial_gan_activations = initial_gan_activations.to("cuda")

    if from_single_vector:
        hook_input_gan = initial_gan_activations#[0, :, 5, 5]#torch.randn((input_.shape[1])).to("cuda")
        gan_handle = forward_hook_gan.register_forward_hook(hook_gan_single_vector)
    else:
        hook_input_gan = initial_gan_activations#+torch.randn_like(input_)#*0.05
        gan_handle = forward_hook_gan.register_forward_hook(hook_gan)

    #input_ = torch.randn_like(input_)*3.0
    #image_f = lambda : F.relu(input_)
    hook_input_gan.requires_grad=True
    optimizer = torch.optim.Adam([hook_input_gan], lr=5e-2)
    #optimizer = torch.optim.SGD([hook_input_gan], lr=0.1, momentum=0.9)

    #image_f = WorkaroundPointerToInput(input_)

    upsample = torch.nn.Identity()#torch.nn.Upsample(size=input_seg_net.shape[-1], mode="bilinear", align_corners=True)

    #upsample = torch.nn.Upsample(224, mode="bilinear", align_corners=True)

    for i in tqdm(range(1, max((num_steps,)) + 1), disable=False, ascii=True):
        optimizer.zero_grad()
        #print(torch.mean(hook_input_gan))

        jitter = 1
        SCALE = 1.1

        #new_val = torchvision.transforms.functional.normalize(image_f(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        new_val = forward_function_gan_batch_input(batch_size)
        new_val = normalization_preprocessor(new_val)

        if regularization == 'full':
            #new_val = image_f()
            #new_val = normalize(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            #new_val = F.pad(x, (2, 2, 2, 2), mode='reflect')#value=0.5, mode='reflect')

            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
            #rotation_values = list(range(-10, 11)) + 5 * [0]

            rotation_values = list(range(-5, 5+1))
            rotation_idx = random.randint(0, len(rotation_values)-1)
            rotate_by = rotation_values[rotation_idx]
            #min_value = -5
            #max_value = 5
            #rotate_by = random.randint(min_value, max_value)# + current_rot
            #scale_values = [1 + (i - 5) / 50. for i in range(11)]
            #scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_value_idx = random.randint(0, len(scale_values)-1)
            new_size = int(new_val.shape[-1]*scale_values[scale_value_idx])
            new_val = torchvision.transforms.functional.resize(new_val, new_size)
            new_val = torchvision.transforms.functional.rotate(new_val, angle=rotate_by, interpolation=2)
            #new_val = (new_val - model.normalization_mean) / model.normalization_std
            #new_val = transform_inception(new_val)
            #new_val = transform_classification(new_val)

            new_val = upsample(new_val)
            #new_val = torch.clamp(new_val, min=0.0, max=1.0)
        elif regularization == 'jitter_only':
            #new_val = image_f()
            #new_val = normalize(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            pass
            #new_val = image_f()
            #new_val = normalize(new_val)
            #new_val = input_
        else:
            raise RuntimeError("regularization input is not known")

        forward_function_seg_net(new_val)

        loss = torch.tensor(0.0)

        for optim_goal in optim_goals:
            loss = loss + optim_goal.compute_loss(new_val)#*0.0

        if i < num_darkening_steps:
            loss = loss + torch.mean(new_val)
        #print(loss)

        loss.backward()
        optimizer.step()

    for optim_goal in optim_goals:
        optim_goal.remove_handle()

    gan_out = forward_function_gan_batch_input(batch_size).cpu().detach()
    gan_handle.remove()
    return gan_out.cpu().detach()
    #return hook_input_gan.cpu().detach()



def total_variation_loss(image):
    # Calculate the total variation loss
    # Total Variation (TV) loss is the sum of the absolute differences for neighboring pixel-values in the input images.
    # It measures the total amount of variation in the image.

    # Shift one pixel to the right and one pixel down
    shift_right = image[:, :, :, 1:]
    shift_down = image[:, :, 1:, :]

    # Compute the differences
    diff_right = image[:, :, :, :-1] - shift_right
    diff_down = image[:, :, :-1, :] - shift_down

    # Compute the total variation loss
    total_variation_loss = torch.mean(torch.abs(diff_right)) + torch.mean(torch.abs(diff_down))

    return total_variation_loss


def gaussian_blur_kernel(kernel_size=10, sigma=1.0):
    # Create a Gaussian blur kernel
    kernel = torch.tensor([[
        [np.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
        for j in range(kernel_size)]
        for i in range(kernel_size)]
        for _ in range(3)]).float()
    kernel = kernel / torch.sum(kernel)  # Normalize the kernel

    return kernel



def general_optim(input_, forward_functions, optim_goals, num_steps=512, fft_magic_fac=4.0, start_from_input=False, regularization='full', use_original_color_data=False,
        use_fft=True, use_decorrelate=True, gan_gen_prior=None, input_to_rgb_transform=True, normalization_preprocessor=torch.nn.Identity(),
        mean_color=None, std_color=None):
    input_ = input_.to("cuda")
    if mean_color is not None:
        mean_color = mean_color.to("cuda")
    if std_color is not None:
        std_color = std_color.to("cuda")

    print("mean color: {}".format(mean_color))
    #raise RuntimeError

    #raise RuntimeError

    blur_kernel = gaussian_blur_kernel().to("cuda").unsqueeze(dim=0)

    from lucent.optvis import param
    #transform_inception = transform.preprocess_inceptionv1()
    #transform_classification = transform.normalize()

    for optim_goal in optim_goals:
        optim_goal.set_handle()

    if not start_from_input:
        param_f = lambda: param.image(input_.shape[-1], decorrelate=use_decorrelate, fft=use_fft, batch=input_.shape[0], fft_magic_fac=fft_magic_fac,
                                use_original_color_data=use_original_color_data, gan_gen_prior=gan_gen_prior, input_to_rgb_transform=input_to_rgb_transform)
        #param_f = lambda: param.cppn(128)
        # param_f is a function that should return two things
        # params - parameters to update, which we pass to the optimizer
        # image_f - a function that returns an image as a tensor
        params, image_f = param_f()

        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
        optimizer = optimizer(params)
    else:
        input_ = torch.randn_like(input_)*3.0
        image_f = lambda : F.relu(input_)
        input_.requires_grad=True
        optimizer = torch.optim.Adam([input_], lr=5e-2)

        #image_f = WorkaroundPointerToInput(input_)


    upsample = torch.nn.Upsample(size=input_.shape[-1], mode="bilinear", align_corners=True)



    use_regularization = True
    for i in tqdm(range(1, max((num_steps,)) + 1), disable=True):
        optimizer.zero_grad()

        jitter = 1
        SCALE = 1.1

        if start_from_input:
            new_val = image_f()
        else:
            new_val = image_f()
        #new_val = torchvision.transforms.functional.normalize(image_f(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if regularization == 'full':
            #new_val = image_f()
            new_val = normalization_preprocessor(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            #new_val = F.pad(x, (2, 2, 2, 2), mode='reflect')#value=0.5, mode='reflect')

            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
            #rotation_values = list(range(-10, 11)) + 5 * [0]

            rotation_values = list(range(-5, 5+1))
            rotation_idx = random.randint(0, len(rotation_values)-1)
            rotate_by = rotation_values[rotation_idx]
            #min_value = -5
            #max_value = 5
            #rotate_by = random.randint(min_value, max_value)# + current_rot
            #scale_values = [1 + (i - 5) / 50. for i in range(11)]
            #scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_values = [SCALE ** (n/10.) for n in range(-10, 11)]
            scale_value_idx = random.randint(0, len(scale_values)-1)
            new_size = int(new_val.shape[-1]*scale_values[scale_value_idx])
            new_val = torchvision.transforms.functional.resize(new_val, new_size)
            new_val = torchvision.transforms.functional.rotate(new_val, angle=rotate_by, interpolation=2)
            #new_val = (new_val - model.normalization_mean) / model.normalization_std
            #new_val = transform_inception(new_val)
            #new_val = transform_classification(new_val)

            #new_val = upsample(new_val)
        elif regularization == 'jitter_only':
            #new_val = image_f()
            #new_val = normalize(new_val)
            new_val = normalization_preprocessor(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)#value=0.5, mode='reflect')
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            new_val = normalization_preprocessor(new_val)
            #pass
            #new_val = image_f()
            #new_val = normalize(new_val)
            #new_val = input_
        else:
            raise RuntimeError("regularization input is not known")



        for forward_function in forward_functions:
            forward_function(new_val)

        loss = 0.0
        #loss = F.l1_loss(torch.mean(new_val, dim=[0, 2, 3]), mean_color) - 0.6*torch.mean(new_val, dim=[0, 2, 3])[2] - 0.4*torch.mean(new_val, dim=[0, 2, 3])[0] + 0.2*torch.mean(new_val) + 1.5*total_variation_loss(new_val)
        #loss = -1.0*torch.mean(new_val, dim=[0, 2, 3])[2] - 0.8*torch.mean(new_val, dim=[0, 2, 3])[0] + 0.8*torch.mean(new_val) + 0.2*total_variation_loss(new_val)
        #loss = 2.0*F.l1_loss(torch.mean(new_val, dim=[0, 2, 3]), mean_color) + 0.2*total_variation_loss(new_val)
        #loss = -0.3*torch.mean(new_val, dim=[0, 2, 3])[2] - 0.2*torch.mean(new_val, dim=[0, 2, 3])[0] + 0.4*torch.mean(new_val) + 0.2*total_variation_loss(new_val)

        """
        loss = 0.3*total_variation_loss(new_val) + 1.5*F.l1_loss(torch.mean(new_val, dim=[0, 2, 3]), 0.9*mean_color) + F.l1_loss(torch.std(new_val, dim=[0, 2, 3]), 0.9*std_color)# + 0.7*torch.mean(new_val)# + torch.argmin(new_val) - torch.argmax(new_val)
        color_stability_loss = 5.0*total_variation_loss(F.avg_pool2d(new_val, kernel_size=60, stride=60))

        #loss = 0.2*total_variation_loss(new_val) + 1.5*F.l1_loss(torch.mean(new_val, dim=[0, 2, 3]), 0.8*mean_color) + 1.5*F.l1_loss(torch.std(new_val, dim=[0, 2, 3]), std_color)# + 0.7*torch.mean(new_val)# + torch.argmin(new_val) - torch.argmax(new_val)
        #color_stability_loss = 5.0*total_variation_loss(F.avg_pool2d(new_val, kernel_size=60, stride=60))

        loss += color_stability_loss# + min_loss + max_loss
        """
        for optim_goal in optim_goals:
            loss = loss + optim_goal.compute_loss(new_val)
        #print(loss)

        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    input_.cpu()
    new_val.cpu()

    for optim_goal in optim_goals:
        optim_goal.remove_handle()
    if start_from_input:
        return image_f().cpu().detach()
    else:
        return image_f().cpu().detach()#image_f().cpu().detach()#x.cpu().detach()





def load_decoder():

    model_from = get_model("version_299").eval()#.cpu()

    #layer = "down_blocks.down block 3.resnet_blocks.resnet_block 2.identity_hook_before_relu"
    layer = "down_blocks.down block 2.resnet_blocks.resnet_block 3"

    return_nodes = {
        # node_name: user-specified key for output dict
        layer: 'out',
    }

    model_from = create_feature_extractor(model_from.model.encoder, return_nodes).eval()

    decoder = Decoder.load_from_checkpoint("/home/digipath2/projects/xai/vector_visualization/logs/version_299/" + layer + "/lightning_logs/version_0/checkpoints/epoch=9-step=21170.ckpt",
                                        feature_extractor=model_from, default_noise_size=512, normalization=nn.Identity()).eval().to(DEVICE)
    decoder.decoder.default_noise_size = 600

    return decoder






def dot_cossim_loss(x, target):
    x_vec = torch.mean(x, dim=[2, 3])
    #return torch.mean(x_vec*target) * torch.mean(F.cosine_similarity(x_vec, target))**4

    return torch.mean(F.cosine_similarity(x_vec, target)) + 0.25*torch.mean(x_vec*target)*torch.mean(F.cosine_similarity(x_vec, target))

    #return torch.mean(F.cosine_similarity(x_vec, target)) - torch.mean(x_vec)



def activation_vec_vis(seg_net, layer_hook_seg_net, target_vectors, num_steps=512, mean_color=None, std_color=None):
    #num_steps = 256

    optimization_visualizations = []
    #for i, target_vector in enumerate(target_vectors):
    b_data = torch.randn((target_vectors.shape[0], 3, 224, 224)).to("cuda")
    #mean_col = mean_color[i].unsqueeze(dim=0)
    #std_col = std_color[i].unsqueeze(dim=0)
    #target_vec = target_vectors.unsqueeze(dim=0)

    #loss_function = lambda x, target: F.l1_loss(torch.mean(x, dim=[2, 3]), 2.0*target)
    loss_function = dot_cossim_loss
    loss_object = LossObject_Target(loss_function, target=target_vectors)
    optim_goal = OptimGoal_ForwardHook(layer_hook_seg_net, loss_object, weight=-1.0)

    optim_vis = general_optim(b_data, [seg_net.forward],
                        [optim_goal], num_steps=num_steps, use_fft=True, use_decorrelate=True,
                        mean_color=None, std_color=None, gan_gen_prior=None)#load_decoder().noise_forward)

    #optimization_visualizations.append(optim_vis[0])
        #decoder = load_decoder()
    #initial_activation = access_activations_forward_hook([target_vectors[:, :128].float().to("cuda")], decoder.forward, decoder.decoder.decoder.up_blocks[-2])
    #optim_vis = optim_gan(None, initial_activation, decoder.forward_no_inputs, decoder.decoder.decoder.up_blocks[-2], seg_net.forward,
    #                      [optim_goal], from_single_vector=False, num_steps=num_steps)

    #return torch.stack(optimization_visualizations, dim=0)

    return optim_vis


    """
    b_data = torch.randn((target_vectors.shape[0], 3, 600, 600)).to("cuda")
    gan = get_model("cond_SLN600").eval().to("cuda")

    #loss_function = lambda x, target: F.l1_loss(torch.mean(x, dim=[2, 3]), 2.0*target)
    loss_function = dot_cossim_loss
    loss_object = LossObject_Target(loss_function, target=target_vectors)
    optim_goal = OptimGoal_ForwardHook(layer_hook_seg_net, loss_object, weight=-1.0)

    optim_vis = optim_gan(None, b_data, gan.forward, gan.identity_layer_for_input_hook, seg_net.forward,
                          [optim_goal], from_single_vector=False, num_steps=num_steps)

    return optim_vis
    """