## Code Snippet to obtain the different perturbation vectors - Gradient, FGSM, DeepFool

## For Gradient vectors
normed_grad = data.grad[i] ## gradient associated with sample data point

## For FGSM vectors
normed_grad = step_epsilon * torch.sign(data.grad[i])
step_adv = data[i] + normed_grad
adv = step_adv - data[i] 
adv = torch.clamp(adv, -eps, eps) # eps = epsilon, CIFAR10 = 8/255

## For DeepFool attack vectors 
r, loop_i, label_orig, label_pert, pert_image = deepfool(data[i], net, num_classes=10, overshoot=0.02, max_iter=10) # store the attack vector r

## Store the list of perturbation vectors of required sample size in a numpy array which will be used in svd-uap.py to obtain the SVD-UAP vector.
