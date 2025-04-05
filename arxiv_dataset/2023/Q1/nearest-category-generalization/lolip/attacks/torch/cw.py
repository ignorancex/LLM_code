"""
Implementation of L_infty Carlini-Wagner attack based on the L2 implementation
in FoolBox v1.9 (with many dependencies on that pakage)
https://github.com/bethgelab/foolbox
"""

from foolbox.criteria import Misclassification
import foolbox as fb
import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from ..base import TorchAttackModel

class CWL2Attack(TorchAttackModel):
  def __init__(self, model_fn, n_classes, eps, norm, batch_size,
               clip_min=None, clip_max=None, device=None):
    self.model_fn = model_fn
    self.eps = eps
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.clip_min = clip_min
    self.clip_max = clip_max
    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device=device

  def perturb(self, X, y, eps=None, is_img_data=True):
    dataset = data_utils.TensorDataset(
        self._preprocess_x(X, is_img_data=is_img_data), torch.from_numpy(y).long())

    return self.perturb_ds(dataset, eps=eps, is_img_data=is_img_data)

  def perturb_ds(self, ds, eps=None, is_img_data=True):
    loader = torch.utils.data.DataLoader(ds,
        batch_size=self.batch_size, shuffle=False, num_workers=2)
    if self.clip_min is None:
      self.clip_min = -1e10
    if self.clip_max is None:
      self.clip_max = 1e10

    fmodel = fb.PyTorchModel(self.model_fn, bounds=(self.clip_min, self.clip_max), device=self.device)
    attack = fb.attacks.L2CarliniWagnerAttack(
      binary_search_steps=9,
      stepsize=1e-2,
      steps=10000,
      abort_early=True,
      initial_const=1e-3,
    )

    ret = []
    for [x, y] in tqdm(loader, desc="Attacking (CWL2)"):
      #x, y = x.cpu().numpy(), y.cpu().numpy()
      x, y = x.to(self.device), Misclassification(y.to(self.device))
      adv_x = attack(model=fmodel, inputs=x, criterion=y, epsilons=None)
      ret.append(adv_x[0].detach().cpu().numpy())
    ret = np.concatenate(ret, axis=0)

    if len(ret.shape) == 4 and is_img_data:
      ret = ret.transpose(0, 2, 3, 1)
    return ret
