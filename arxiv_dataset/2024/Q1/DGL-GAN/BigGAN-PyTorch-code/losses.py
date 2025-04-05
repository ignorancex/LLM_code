import torch
import torch.nn.functional as F

class hinge_pos(torch.autograd.Function):
  """
  Activation function for hinge loss
      if x >= 1, y = 1,
      elif 0 < x < 1, y = x,
      else, y = 1
  """
  @staticmethod
  def forward(ctx, input):
    input[input >= 1] = 1
    input[input <= 0] = 1

    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

class hinge_neg(torch.autograd.Function):
  """
  Activation function for hinge loss
      if x <= -1, y = 1,
      elif -1 < x < 0, y = -x,
      else, y = 1
  """
  @staticmethod
  def forward(ctx, input):
    input = input * (-1)
    input[input >= 1] = 1
    input[input <= 0] = 1

    return input

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output



# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2

def loss_dcgan_dis_smooth(dis_fake, dis_real, dis_fake_bar, dis_real_bar):
  L1 = torch.mean(torch.sigmoid(dis_real_bar) * F.softplus(-dis_real))
  L2 = torch.mean((1 - torch.sigmoid(dis_fake_bar)) * F.softplus(dis_fake))
  return L1, L2

def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss

# Hinge Loss
def loss_hinge_dis_smooth(dis_fake, dis_real, dis_fake_bar, dis_real_bar):
  loss_real = torch.mean(torch.sigmoid(dis_real_bar) * F.relu(1. - dis_real))
  loss_fake = torch.mean((1 - torch.sigmoid(dis_fake_bar)) * F.relu(1. + dis_fake))
  return loss_real, loss_fake

# Gradient penalty
def d_r1_loss(dis_real, real_img):
    grad_real, = torch.autograd.grad(
        outputs=dis_real.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

# Hinge Loss
def loss_hinge_dis_smoothCus(dis_fake, dis_real, dis_fake_bar, dis_real_bar):
  pos, neg = hinge_pos.apply, hinge_neg.apply
  loss_real = torch.mean(pos(dis_real_bar) * F.relu(1. - dis_real))
  loss_fake = torch.mean(neg(dis_fake_bar) * F.relu(1. + dis_fake))
  return loss_real, loss_fake

def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

def loss_hinge_gen_L1(dis_fake, tea_feas, stu_feas, lamina=0.05):
  loss = -torch.mean(dis_fake)
  l1loss = torch.nn.L1Loss(reduction='None')
  L1 = torch.mean(l1loss(stu_feas[0], tea_feas[0]))
  for t_fea, s_fea in zip(tea_feas[1:], stu_feas[1:]):
    L1 += torch.mean(l1loss(s_fea, t_fea))
  total_loss = loss + lamina * L1
  return total_loss, loss, L1

def loss_hinge_genDual(dis_fake, dis_fake_bar):
  loss = -torch.mean(dis_fake)
  dual = -torch.mean(dis_fake_bar)
  return loss+0.1*dual, loss, dual

# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
