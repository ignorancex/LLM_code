from loss_utils import MammalContrastiveCriterion
import torch

criterion = MammalContrastiveCriterion()

a = torch.ones(size=(8, 1024, 1, 1))
c = torch.ones(size=(4, 1024))


loss = criterion(a, c, concept_flag=0)
print(loss)