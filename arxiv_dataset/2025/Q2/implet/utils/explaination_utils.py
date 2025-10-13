import torch
from captum.attr import Occlusion, InputXGradient, IntegratedGradients, Saliency, GradientShap, DeepLift, Lime, \
    KernelShap, GuidedBackprop
def get_xai_ref(xai_name):
    if xai_name == 'DeepLift':
        xai_ref = DeepLift
    elif xai_name == 'IntegratedGradients':
        xai_ref = IntegratedGradients
    elif xai_name == 'InputXGradient':
        xai_ref = InputXGradient
    elif xai_name == 'Saliency':
        xai_ref = Saliency
    elif xai_name == 'GradientShap':
        xai_ref = GradientShap
    elif xai_name == 'Lime':
        xai_ref = Lime
    elif xai_name == 'Occlusion':
        xai_ref = Occlusion
    elif xai_name == 'KernelShap':
        xai_ref = KernelShap
    elif xai_name == 'GuidedBackprop':
        xai_ref = GuidedBackprop
    return xai_ref


def explain(xia_method, xai_name, input, predicted_label, sliding_window=(1, 10), baselines=None):
    if xai_name == "Occlusion":
        exp = xia_method.attribute(input, target=predicted_label, sliding_window_shapes=sliding_window)  # for Occlusion
    elif xai_name == "GradientShap":
        exp = xia_method.attribute(input, baselines=baselines, target=predicted_label)
    elif xai_name in ["Lime", "KernelShap"]:
        exp = torch.zeros(input.shape).float().cuda()
        for i in range(input.shape[0]):
            exp[i] = xia_method.attribute(input[i:i + 1], target=predicted_label)
    else:
        exp = xia_method.attribute(input, target=predicted_label)
    return exp