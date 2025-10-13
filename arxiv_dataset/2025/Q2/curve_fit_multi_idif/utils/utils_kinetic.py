import torch
from utils.utils_torch import torch_conv, torch_conv_batch

# Use of step: https://dsp.stackexchange.com/questions/26079/why-does-the-amplitude-of-a-discrete-convolution-depend-on-the-time-step
def PET_2TC_KM(input_aorta_TAC, input_portal_TAC, t, k1, k2, k3, k4, Vb, alpha, step=0.1):


    a = alpha * input_aorta_TAC + (1-alpha) * input_portal_TAC
    # Extended two-tissue compartment model to include k4

    # Calculate b1 and b2:
    b1 = 0.5 * ((k2 + k3 + k4) - torch.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))
    b2 = 0.5 * ((k2 + k3 + k4) + torch.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))
    
    # Compute exponential decay terms
    e1 = torch.exp(-b1 * t)
    e2 = torch.exp(-b2 * t)

    d = (k1 / (b2 - b1)) * ((k3 + k4 - b1)*e1 + (b2 - k3 - k4)*e2)

    c = torch_conv(a, d) * step

    PET = (1 - Vb) * c + Vb * a
    PET.requires_grad_()
    return PET

def PET_2TC_KM_batch(input_aorta_TAC, input_portal_TAC, t, k1, k2, k3, Vb, alpha, step=0.1):

    a = alpha * input_aorta_TAC + (1-alpha) * input_portal_TAC  #(1, h*w, 600)

    e = torch.multiply(k2+k3, t)
    b = k1 / (k2+k3) * (k3 + k2*torch.exp(-e))         # 2TC irreversible
    c = torch_conv_batch(a, b) * step

    # The permutations are required to get the expected shape
    Vb = Vb.permute((1, 0, 2))
    PET = (1-Vb) * c + Vb * a

    PET = PET.permute((0, 2, 1))
    PET.requires_grad_()
    return PET

